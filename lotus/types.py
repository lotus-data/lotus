from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import pandas as pd
from litellm.types.utils import ChatCompletionTokenLogprob
from pydantic import BaseModel


################################################################################
# LM related
################################################################################
@dataclass
class LMOutput:
    outputs: list[str]
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


@dataclass
class LMStats:
    @dataclass
    class TotalUsage:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        total_cost: float = 0.0
        # Cached tokens (prompt cache hits) - charged at lower rate
        cached_prompt_tokens: int = 0
        # Tokens used to create the cache (prompt cache writes) - one-time cost
        cache_creation_tokens: int = 0

        def __sub__(self, other: "LMStats.TotalUsage") -> "LMStats.TotalUsage":
            return LMStats.TotalUsage(
                prompt_tokens=self.prompt_tokens - other.prompt_tokens,
                completion_tokens=self.completion_tokens - other.completion_tokens,
                total_tokens=self.total_tokens - other.total_tokens,
                total_cost=self.total_cost - other.total_cost,
                cached_prompt_tokens=self.cached_prompt_tokens - other.cached_prompt_tokens,
                cache_creation_tokens=self.cache_creation_tokens - other.cache_creation_tokens,
            )

        def __add__(self, other: "LMStats.TotalUsage") -> "LMStats.TotalUsage":
            return LMStats.TotalUsage(
                prompt_tokens=self.prompt_tokens + other.prompt_tokens,
                completion_tokens=self.completion_tokens + other.completion_tokens,
                total_tokens=self.total_tokens + other.total_tokens,
                total_cost=self.total_cost + other.total_cost,
                cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
                cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            )

    # Usage stats if there was no caching
    virtual_usage: TotalUsage = field(default_factory=TotalUsage)
    # Actual usage with caching applied
    physical_usage: TotalUsage = field(default_factory=TotalUsage)

    cache_hits: int = 0
    operator_cache_hits: int = 0

    def __add__(self, other: "LMStats") -> "LMStats":
        return LMStats(
            virtual_usage=self.virtual_usage + other.virtual_usage,
            physical_usage=self.physical_usage + other.physical_usage,
            cache_hits=self.cache_hits + other.cache_hits,
            operator_cache_hits=self.operator_cache_hits + other.operator_cache_hits,
        )


@dataclass
class LogprobsForCascade:
    tokens: list[list[str]]
    confidences: list[list[float]]


@dataclass
class LogprobsForFilterCascade:
    true_probs: list[float]
    tokens: list[list[str]]
    confidences: list[list[float]]


################################################################################
# Semantic operation outputs
################################################################################
@dataclass
class SemanticMapPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


@dataclass
class SemanticMapOutput:
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


@dataclass
class SemanticExtractPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]
    explanations: list[str | None]


@dataclass
class SemanticExtractOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]
    explanations: list[str | None]


@dataclass
class SemanticFilterPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]


@dataclass
class RawOutputs:
    predictions: list[list[bool]]
    raw_outputs: list[list[str]]
    explanations: list[list[str | None]]
    logprobs: list[list[list[ChatCompletionTokenLogprob]]] | None = None


@dataclass
class SemanticFilterOutput:
    outputs: list[bool]
    _raw_outputs: RawOutputs
    stats: dict[str, Any] | None = None

    @property
    def raw_outputs(self) -> list[str]:
        # Backward compatibility: flatten if single run, or return first run?
        # Feedback says: "check if the lists only have 1 item and return that item"
        if len(self._raw_outputs.raw_outputs) > 0 and len(self._raw_outputs.raw_outputs[0]) == 1:
            return [runs[0] for runs in self._raw_outputs.raw_outputs]
        # Fallback if multiple runs: return just the first run's data or handle differently?
        # For now, let's return the simplified single-run view required by existing tests
        return [runs[0] for runs in self._raw_outputs.raw_outputs]

    @property
    def explanations(self) -> list[str | None]:
        if len(self._raw_outputs.explanations) > 0 and len(self._raw_outputs.explanations[0]) == 1:
            return [runs[0] for runs in self._raw_outputs.explanations]
        return [runs[0] for runs in self._raw_outputs.explanations]

    @property
    def logprobs(self) -> list[list[ChatCompletionTokenLogprob]] | None:
        if self._raw_outputs.logprobs:
            if len(self._raw_outputs.logprobs) > 0 and len(self._raw_outputs.logprobs[0]) == 1:
                return [runs[0] for runs in self._raw_outputs.logprobs]
            return [runs[0] for runs in self._raw_outputs.logprobs]
        return None



@dataclass
class SemanticAggOutput:
    outputs: list[str]


class LongContextStrategy(Enum):
    """Enumeration of available document long_context strategies for semantic aggregation."""

    TRUNCATE = auto()
    CHUNK = auto()


@dataclass
class SemanticJoinOutput:
    join_results: list[tuple[int, int, str | None]]
    filter_outputs: list[bool]
    all_raw_outputs: list[str]
    all_explanations: list[str | None]
    stats: dict[str, Any] | None = None


class ProxyModel(Enum):
    HELPER_LM = "helper_lm"
    EMBEDDING_MODEL = "embedding_model"


class CascadeArgs(BaseModel):
    recall_target: float = 0.8
    precision_target: float = 0.8
    sampling_percentage: float = 0.1
    failure_probability: float = 0.2
    map_instruction: str | None = None
    map_examples: pd.DataFrame | None = None
    proxy_model: ProxyModel = ProxyModel.HELPER_LM

    # Filter cascade args
    cascade_IS_weight: float = 0.9
    cascade_num_calibration_quantiles: int = 50

    # Join cascade args
    min_join_cascade_size: int = 100
    cascade_IS_max_sample_range: int = 200
    cascade_IS_random_seed: int | None = None

    # to enable pandas
    class Config:
        arbitrary_types_allowed = True


@dataclass
class SemanticTopKOutput:
    indexes: list[int]
    stats: dict[str, Any] | None = None


################################################################################
# RM related
################################################################################


@dataclass
class RMOutput:
    distances: list[list[float]]
    indices: list[list[int]]


################################################################################
# Reranker related
################################################################################
@dataclass
class RerankerOutput:
    indices: list[int]


################################################################################
# Serialization related
################################################################################
class SerializationFormat(Enum):
    JSON = "json"
    XML = "xml"
    DEFAULT = "default"


################################################################################
# Utility
################################################################################
@dataclass
class UsageLimit:
    prompt_tokens_limit: float = float("inf")
    completion_tokens_limit: float = float("inf")
    total_tokens_limit: float = float("inf")
    total_cost_limit: float = float("inf")


################################################################################
# Exception related
################################################################################
class LotusException(Exception):
    """Base class for all Lotus exceptions."""

    pass


class LotusUsageLimitException(LotusException):
    """Exception raised when the usage limit is exceeded."""

    pass


################################################################################
# Reasoning Strategy
################################################################################
class ReasoningStrategy(Enum):
    DEFAULT = auto()
    COT = auto()
    ZS_COT = auto()
    FEW_SHOT = auto()
