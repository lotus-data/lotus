from dataclasses import dataclass, field
from enum import Enum
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
        cache_hits: int = 0
        operator_cache_hits: int = 0

    total_usage: TotalUsage = field(default_factory=TotalUsage)


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


@dataclass
class SemanticExtractOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]


@dataclass
class SemanticFilterPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]


@dataclass
class SemanticFilterOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]
    stats: dict[str, Any] | None = None
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


@dataclass
class SemanticAggOutput:
    outputs: list[str]


@dataclass
class SemanticJoinOutput:
    join_results: list[tuple[int, int, str | None]]
    filter_outputs: list[bool]
    all_raw_outputs: list[str]
    all_explanations: list[str | None]
    stats: dict[str, Any] | None = None


class CascadeArgs(BaseModel):
    recall_target: float = 0.8
    precision_target: float = 0.8
    sampling_percentage: float = 0.1
    failure_probability: float = 0.2
    map_instruction: str | None = None
    map_examples: pd.DataFrame | None = None

    # Filter cascade args
    cascade_IS_weight: float = 0.5
    cascade_num_calibration_quantiles: int = 50

    # Join cascade args
    min_join_cascade_size: int = 100
    cascade_IS_max_sample_range: int = 250
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
