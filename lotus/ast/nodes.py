"""Pydantic node models for LOTUS AST pipelines.

Each node is a BaseModel storing operator arguments with a __call__ method for execution.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

import lotus
from lotus.types import (
    CascadeArgs,
    LongContextStrategy,
    ReasoningStrategy,
    SemanticExtractPostprocessOutput,
    SemanticMapPostprocessOutput,
)


class BaseNode(BaseModel):
    """Base class for all AST nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame | Any:
        """Execute this node on the input DataFrame."""
        raise NotImplementedError(f"{type(self).__name__}.__call__ not implemented")

    def _log_execution(self, input_rows: int, output_rows: int | None = None) -> None:
        """Log node execution details."""
        if output_rows is not None:
            lotus.logger.debug(f"{type(self).__name__}: input={input_rows} rows, output={output_rows} rows")
        else:
            lotus.logger.debug(f"{type(self).__name__}: input={input_rows} rows")


# ------------------------------------------------------------------
# Source Node
# ------------------------------------------------------------------


class SourceNode(BaseNode):
    """Source node representing input data."""

    key: str = "default"
    df: pd.DataFrame | None = None

    def __call__(self, df: pd.DataFrame | None = None, **context: Any) -> pd.DataFrame:
        """Return the source DataFrame."""
        lotus.logger.debug(f"SourceNode: loading source '{self.key}'")
        if df is not None:
            lotus.logger.debug(f"SourceNode: loaded {len(df)} rows from provided df")
            return df
        if self.df is not None:
            lotus.logger.debug(f"SourceNode: loaded {len(self.df)} rows from bound df")
            return self.df
        raise ValueError(f"No DataFrame provided for source '{self.key}'")


# ------------------------------------------------------------------
# Semantic Operator Nodes
# ------------------------------------------------------------------


class SemFilterNode(BaseNode):
    """Semantic filter node."""

    user_instruction: str
    return_raw_outputs: bool = False
    return_explanations: bool = False
    return_all: bool = False
    default: bool = True
    suffix: str = "_filter"
    examples: pd.DataFrame | None = None
    helper_examples: pd.DataFrame | None = None
    strategy: ReasoningStrategy | None = None
    cascade_args: CascadeArgs | None = None
    return_stats: bool = False
    safe_mode: bool = False
    progress_bar_desc: str = "Filtering"
    additional_cot_instructions: str = ""

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        lotus.logger.debug(f"SemFilterNode: filtering {len(df)} rows with instruction: {self.user_instruction[:50]}...")
        result = df.sem_filter(
            self.user_instruction,
            return_raw_outputs=self.return_raw_outputs,
            return_explanations=self.return_explanations,
            return_all=self.return_all,
            default=self.default,
            suffix=self.suffix,
            examples=self.examples,
            helper_examples=self.helper_examples,
            strategy=self.strategy,
            cascade_args=self.cascade_args,
            return_stats=self.return_stats,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            additional_cot_instructions=self.additional_cot_instructions,
        )
        if isinstance(result, tuple):
            lotus.logger.debug(f"SemFilterNode: output {len(result[0])} rows (with stats)")
        else:
            lotus.logger.debug(f"SemFilterNode: output {len(result)} rows")
        return result


class SemMapNode(BaseNode):
    """Semantic map node."""

    user_instruction: str
    system_prompt: str | None = None
    postprocessor: Callable[[list[str], Any, bool], SemanticMapPostprocessOutput] | None = None
    return_explanations: bool = False
    return_raw_outputs: bool = False
    suffix: str = "_map"
    examples: pd.DataFrame | None = None
    strategy: ReasoningStrategy | None = None
    safe_mode: bool = False
    progress_bar_desc: str = "Mapping"
    model_kwargs: dict[str, Any] | None = None

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemMapNode: mapping {len(df)} rows with instruction: {self.user_instruction[:50]}...")
        kwargs: dict[str, Any] = {}
        if self.postprocessor is not None:
            kwargs["postprocessor"] = self.postprocessor
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)

        result = df.sem_map(
            self.user_instruction,
            system_prompt=self.system_prompt,
            return_explanations=self.return_explanations,
            return_raw_outputs=self.return_raw_outputs,
            suffix=self.suffix,
            examples=self.examples,
            strategy=self.strategy,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            **kwargs,
        )
        lotus.logger.debug(f"SemMapNode: output {len(result)} rows")
        return result


class SemExtractNode(BaseNode):
    """Semantic extract node."""

    input_cols: list[str]
    output_cols: dict[str, str | None]
    extract_quotes: bool = False
    postprocessor: Callable[[list[str], Any, bool], SemanticExtractPostprocessOutput] | None = None
    return_raw_outputs: bool = False
    safe_mode: bool = False
    progress_bar_desc: str = "Extracting"
    return_explanations: bool = False
    strategy: ReasoningStrategy | None = None

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemExtractNode: extracting from {len(df)} rows, input_cols={self.input_cols}")
        kwargs: dict[str, Any] = {}
        if self.postprocessor is not None:
            kwargs["postprocessor"] = self.postprocessor

        result = df.sem_extract(
            self.input_cols,
            self.output_cols,
            extract_quotes=self.extract_quotes,
            return_raw_outputs=self.return_raw_outputs,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            return_explanations=self.return_explanations,
            strategy=self.strategy,
            **kwargs,
        )
        lotus.logger.debug(f"SemExtractNode: output {len(result)} rows with cols {list(self.output_cols.keys())}")
        return result


class SemAggNode(BaseNode):
    """Semantic aggregation node."""

    user_instruction: str
    all_cols: bool = False
    suffix: str = "_output"
    group_by: list[str] | None = None
    safe_mode: bool = False
    progress_bar_desc: str = "Aggregating"
    long_context_strategy: LongContextStrategy | None = LongContextStrategy.CHUNK

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemAggNode: aggregating {len(df)} rows with instruction: {self.user_instruction[:50]}...")
        result = df.sem_agg(
            self.user_instruction,
            all_cols=self.all_cols,
            suffix=self.suffix,
            group_by=self.group_by,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            long_context_strategy=self.long_context_strategy,
        )
        lotus.logger.debug(f"SemAggNode: output {len(result)} rows")
        return result


class SemTopKNode(BaseNode):
    """Semantic top-k node."""

    user_instruction: str
    K: int
    method: str = "quick"
    strategy: ReasoningStrategy | None = None
    group_by: list[str] | None = None
    cascade_threshold: float | None = None
    return_stats: bool = False
    safe_mode: bool = False
    return_explanations: bool = False

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        lotus.logger.debug(
            f"SemTopKNode: selecting top {self.K} from {len(df)} rows with instruction: {self.user_instruction[:50]}..."
        )
        result = df.sem_topk(
            self.user_instruction,
            K=self.K,
            method=self.method,
            strategy=self.strategy,
            group_by=self.group_by,
            cascade_threshold=self.cascade_threshold,
            return_stats=self.return_stats,
            safe_mode=self.safe_mode,
            return_explanations=self.return_explanations,
        )
        if isinstance(result, tuple):
            lotus.logger.debug(f"SemTopKNode: output {len(result[0])} rows (with stats)")
        else:
            lotus.logger.debug(f"SemTopKNode: output {len(result)} rows")
        return result


class SemJoinNode(BaseNode):
    """Semantic join node.

    Supports three modes for specifying the right DataFrame:
    1. right_pipeline: Pipeline object to execute first
    2. right_source_node: Source node to join with
    3. right_df: direct DataFrame reference
    """

    right_source_node: SourceNode | None = None
    right_pipeline: Any = None  # Actually Pipeline, but avoid circular import
    right_df: pd.DataFrame | None = None

    join_instruction: str
    return_explanations: bool = False
    how: str = "inner"
    suffix: str = "_join"
    examples: pd.DataFrame | None = None
    strategy: ReasoningStrategy | None = None
    default: bool = True
    cascade_args: CascadeArgs | None = None
    return_stats: bool = False
    safe_mode: bool = False
    progress_bar_desc: str = "Join comparisons"

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        # Get right DataFrame from context (resolved by Run)
        resolved_right_df = context.get("right_df")
        if resolved_right_df is None:
            raise ValueError("Right DataFrame not provided in context. This should be resolved by Run.")

        lotus.logger.debug(f"SemJoinNode: joining {len(df)} left rows with {len(resolved_right_df)} right rows")
        lotus.logger.debug(f"SemJoinNode: instruction: {self.join_instruction[:50]}...")
        result = df.sem_join(
            resolved_right_df,
            self.join_instruction,
            return_explanations=self.return_explanations,
            how=self.how,
            suffix=self.suffix,
            examples=self.examples,
            strategy=self.strategy,
            default=self.default,
            cascade_args=self.cascade_args,
            return_stats=self.return_stats,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
        )
        lotus.logger.debug(f"SemJoinNode: output {len(result)} rows")
        return result


class SemSimJoinNode(BaseNode):
    """Semantic similarity join node.

    Supports two modes for specifying the right DataFrame:
    1. right_source_node: Source node to join with
    2. right_pipeline: Pipeline object to execute first
    2. right_df: direct DataFrame reference
    """

    right_source_node: SourceNode | None = None
    right_pipeline: Any = None  # Actually Pipeline, but avoid circular import
    right_df: pd.DataFrame | None = None

    left_on: str
    right_on: str
    K: int
    lsuffix: str = ""
    rsuffix: str = ""
    score_suffix: str = ""
    keep_index: bool = False

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        # Get right DataFrame from context (resolved by Run)
        resolved_right_df = context.get("right_df")
        if resolved_right_df is None:
            raise ValueError("Right DataFrame not provided in context. This should be resolved by Run.")

        lotus.logger.debug(
            f"SemSimJoinNode: joining {len(df)} left rows with {len(resolved_right_df)} right rows, K={self.K}"
        )
        result = df.sem_sim_join(
            resolved_right_df,
            left_on=self.left_on,
            right_on=self.right_on,
            K=self.K,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            score_suffix=self.score_suffix,
            keep_index=self.keep_index,
        )
        lotus.logger.debug(f"SemSimJoinNode: output {len(result)} rows")
        return result


class SemSearchNode(BaseNode):
    """Semantic search node."""

    col_name: str
    query: str
    K: int | None = None
    n_rerank: int | None = None
    return_scores: bool = False
    suffix: str = "_sim_score"

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemSearchNode: searching {len(df)} rows for query: {self.query[:50]}...")
        result = df.sem_search(
            self.col_name,
            self.query,
            K=self.K,
            n_rerank=self.n_rerank,
            return_scores=self.return_scores,
            suffix=self.suffix,
        )
        lotus.logger.debug(f"SemSearchNode: output {len(result)} rows")
        return result


class SemIndexNode(BaseNode):
    """Semantic index node."""

    col_name: str
    index_dir: str

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemIndexNode: indexing column '{self.col_name}' from {len(df)} rows to {self.index_dir}")
        result = df.sem_index(self.col_name, self.index_dir)
        lotus.logger.debug(f"SemIndexNode: indexed {len(result)} rows")
        return result


class SemClusterByNode(BaseNode):
    """Semantic cluster node."""

    col_name: str
    ncentroids: int
    return_scores: bool = False
    return_centroids: bool = False
    niter: int = 20
    verbose: bool = False

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        lotus.logger.debug(f"SemClusterByNode: clustering {len(df)} rows into {self.ncentroids} clusters")
        result = df.sem_cluster_by(
            self.col_name,
            ncentroids=self.ncentroids,
            return_scores=self.return_scores,
            return_centroids=self.return_centroids,
            niter=self.niter,
            verbose=self.verbose,
        )
        if isinstance(result, tuple):
            lotus.logger.debug(f"SemClusterByNode: output {len(result[0])} rows (with centroids)")
        else:
            lotus.logger.debug(f"SemClusterByNode: output {len(result)} rows")
        return result


class SemDedupNode(BaseNode):
    """Semantic deduplication node."""

    col_name: str
    threshold: float

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemDedupNode: deduplicating {len(df)} rows with threshold={self.threshold}")
        result = df.sem_dedup(self.col_name, self.threshold)
        lotus.logger.debug(f"SemDedupNode: output {len(result)} rows")
        return result


class SemPartitionByNode(BaseNode):
    """Semantic partition node."""

    partition_fn: Callable[[pd.DataFrame], list[int]]

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"SemPartitionByNode: partitioning {len(df)} rows")
        result = df.sem_partition_by(self.partition_fn)
        lotus.logger.debug(f"SemPartitionByNode: output {len(result)} rows")
        return result


# ------------------------------------------------------------------
# Pandas Operator Nodes
# ------------------------------------------------------------------


class PandasFilterNode(BaseNode):
    """Pandas boolean filter node."""

    predicate: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        lotus.logger.debug(f"PandasFilterNode: filtering {len(df)} rows")
        mask = self.predicate(df)
        result = df[mask]
        lotus.logger.debug(f"PandasFilterNode: output {len(result)} rows")
        return result


class PandasOpNode(BaseNode):
    """Generic pandas operation node.

    Supports Pipeline references in arguments via pipeline_args dict.
    During execution, Run resolves these Pipeline references to DataFrames.
    """

    op_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    is_attr: bool = False  # True if this is a property access, not a method call
    # Pipeline references by arg index: {"_pipeline_arg_0": Pipeline, ...}
    pipeline_args: dict[str, Any] | None = None
    # Pipeline references in kwargs: {"_pipeline_kwarg_other": Pipeline, ...}
    pipeline_kwargs: dict[str, Any] | None = None

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame | Any:
        if self.is_attr:
            lotus.logger.debug(f"PandasOpNode: accessing attribute '{self.op_name}' on {len(df)} rows")
            return getattr(df, self.op_name)

        # Get resolved args/kwargs from context (resolved by Run)
        resolved_args = context.get("resolved_args", self.args)
        resolved_kwargs = context.get("resolved_kwargs", self.kwargs or {})

        lotus.logger.debug(f"PandasOpNode: calling '{self.op_name}' on {len(df)} rows")
        method = getattr(df, self.op_name)
        result = method(*resolved_args, **resolved_kwargs)

        if hasattr(result, "__len__") and not isinstance(result, str):
            lotus.logger.debug(f"PandasOpNode: output {len(result)} rows/items")
        else:
            lotus.logger.debug(f"PandasOpNode: output type={type(result).__name__}")

        # Return the result (may be DataFrame, Series, or other)
        return result if result is not None else df


class PandasAssignNode(BaseNode):
    """Node for column assignment operations.

    Supports two modes:
    1. Single column assignment: column + value
    2. Multiple assignments: assignments dict (like pandas assign())

    Also supports Pipeline references that are resolved lazily during execution:
    - value_pipeline: Pipeline for single column mode (resolved by Run)
    - assignment_pipelines: dict of column -> Pipeline for multi-column mode
    """

    # Single column mode
    column: str | None = None
    value: Any = None
    # Multi-column mode (like pandas assign)
    assignments: dict[str, Any] | None = None
    # Pipeline references for lazy resolution (resolved by Run before calling)
    value_pipeline: Any = None  # Pipeline for single column mode
    assignment_pipelines: dict[str, Any] | None = None  # column -> Pipeline

    def __call__(self, df: pd.DataFrame, **context: Any) -> pd.DataFrame:
        result = df.copy()

        # Get resolved values from context (populated by Run for Pipeline references)
        resolved_value = context.get("resolved_value", self.value)
        resolved_assignments = context.get("resolved_assignments", {})

        if self.assignments or self.assignment_pipelines:
            # Multi-column assignment mode
            # Merge static assignments with resolved pipeline assignments
            all_assignments = dict(self.assignments or {})
            all_assignments.update(resolved_assignments)

            cols = list(all_assignments.keys())
            lotus.logger.debug(f"PandasAssignNode: assigning {len(cols)} columns to {len(df)} rows: {cols}")
            for col, val in all_assignments.items():
                if callable(val):
                    result[col] = val(result)
                else:
                    result[col] = val
        elif self.column is not None:
            # Single column assignment mode
            lotus.logger.debug(f"PandasAssignNode: assigning column '{self.column}' to {len(df)} rows")
            # Use resolved_value if we had a value_pipeline, otherwise use self.value
            val = resolved_value if self.value_pipeline is not None else self.value
            if callable(val):
                result[self.column] = val(result)
            else:
                result[self.column] = val

        lotus.logger.debug(f"PandasAssignNode: output {len(result)} rows")
        return result


class ApplyFnNode(BaseNode):
    """Node for a callable that takes only Pipeline results (no 'self' DataFrame).

    Used for pd.concat, custom combiners, etc. Run executes each pipeline
    found in args/kwargs recursively, then calls fn with resolved values.

    Args and kwargs may contain Pipeline instances or nested structures (lists, tuples, dicts)
    containing Pipelines. _resolve_pipeline_structure will recursively detect and execute them.
    """

    fn: Any  # Callable, e.g. pd.concat
    args: tuple[Any, ...] = ()  # Positional arguments (may contain Pipelines or nested structures)
    kwargs: dict[str, Any] | None = None  # Keyword arguments (may contain Pipelines or nested structures)

    def __call__(self, df: pd.DataFrame | None = None, **context: Any) -> Any:
        """Execute the function with resolved Pipeline arguments.

        Args:
            df: Ignored (no 'self' DataFrame for ApplyFnNode)
            **context: Must contain 'resolved_fn_args' and 'resolved_fn_kwargs'
        """
        # Resolved values injected by Run; df is ignored
        resolved = context.get("resolved_fn_args", ())
        resolved_kwargs = context.get("resolved_fn_kwargs", {})
        lotus.logger.debug(
            f"ApplyFnNode: calling {self.fn} with {len(resolved)} args and {len(resolved_kwargs)} kwargs"
        )
        result = self.fn(*resolved, **resolved_kwargs)
        if hasattr(result, "__len__") and not isinstance(result, str):
            lotus.logger.debug(f"ApplyFnNode: output {len(result)} rows/items")
        else:
            lotus.logger.debug(f"ApplyFnNode: output type={type(result).__name__}")
        return result
