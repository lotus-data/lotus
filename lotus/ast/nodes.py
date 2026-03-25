"""Pydantic node models for LOTUS LazyFrame.

Each node is a BaseModel storing operator arguments with a ``__call__`` method
for execution.  Nodes also expose two display hooks consumed by
``LazyFrame.show()``:

* ``signature()`` — one-line human-readable summary for tree display.
* ``child_lfs()`` — labeled nested LazyFrame references for tree display.

Nodes that hold LazyFrame references resolve them inside ``__call__`` using
the ``resolver`` callable passed in via ``**context`` by the runner.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

import lotus
from lotus.types import (
    CascadeArgs,
    LongContextStrategy,
    ProxyModel,
    ReasoningStrategy,
    SemanticExtractPostprocessOutput,
    SemanticMapPostprocessOutput,
)

Resolver = Callable[[Any], Any]


def _no_resolver(ref: Any) -> Any:
    """Default resolver: pass through plain values, raise on LazyFrame / SourceNode."""
    from .lazyframe import LazyFrame

    if isinstance(ref, LazyFrame):
        raise RuntimeError(f"Cannot resolve {type(ref).__name__} without a runner. " "Pass a resolver to the node.")
    if isinstance(ref, SourceNode):
        if ref.df is not None:
            return ref.df
        raise ValueError("SourceNode has no DataFrame specified. Pass a resolver to the node.")
    if isinstance(ref, pd.DataFrame):
        return ref
    if isinstance(ref, list):
        return [_no_resolver(v) for v in ref]
    if isinstance(ref, tuple):
        return tuple(_no_resolver(v) for v in ref)
    if isinstance(ref, dict):
        return {k: _no_resolver(v) for k, v in ref.items()}
    return ref


def _truncate(text: str, max_len: int = 50) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


# ------------------------------------------------------------------
# Base
# ------------------------------------------------------------------


class BaseNode(BaseModel):
    """Base class for all AST nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimizable_params: frozenset[str] | None = None

    def __call__(self, df: pd.DataFrame | None = None, resolver: Resolver = _no_resolver, **context: Any) -> Any:
        """Execute this node on the input DataFrame.

        Args:
            df: The current DataFrame flowing through the lazyframe (or None for
                source/apply nodes that do not require it).
            resolver: A recursive callable that materialises LazyFrame and
                SourceNode references.  Supplied by the runner; the default
                raises on any unresolved reference.
            **context: Extra keyword arguments forwarded by the runner.
                Nodes may inspect these for optional behaviour (e.g.
                ``cascade_stats_collector``).
        """
        raise NotImplementedError(f"{type(self).__name__}.__call__ not implemented")

    # -- Optimization hooks ---------------------------------------------

    @staticmethod
    def _split_optimizable_param(param_name: str) -> tuple[Any, ...]:
        parts: list[Any] = []
        token: list[str] = []
        i = 0
        while i < len(param_name):
            ch = param_name[i]
            if ch == ".":
                if token:
                    parts.append("".join(token))
                    token = []
                i += 1
                continue
            if ch == "[":
                if token:
                    parts.append("".join(token))
                    token = []
                close = param_name.find("]", i + 1)
                if close == -1:
                    raise ValueError(f"Invalid optimizable parameter path: {param_name!r}")
                key_expr = param_name[i + 1 : close].strip()
                if not key_expr:
                    raise ValueError(f"Invalid optimizable parameter path: {param_name!r}")
                try:
                    key = ast.literal_eval(key_expr)
                except (SyntaxError, ValueError):
                    if key_expr.lstrip("-").isdigit():
                        key = int(key_expr)
                    else:
                        key = key_expr
                parts.append(key)
                i = close + 1
                continue
            token.append(ch)
            i += 1

        if token:
            parts.append("".join(token))

        return tuple(part for part in parts if part != "")

    @staticmethod
    def _has_nested_param(container: Any, path: tuple[Any, ...]) -> bool:
        current = container
        for part in path:
            if isinstance(current, BaseModel):
                if not isinstance(part, str):
                    return False
                if part not in type(current).model_fields:
                    return False
                current = getattr(current, part)
            elif isinstance(current, (list, tuple)):
                if not isinstance(part, int) or part < 0 or part >= len(current):
                    return False
                current = current[part]
            elif isinstance(current, dict):
                if part not in current:
                    return False
                current = current[part]
            else:
                return False
        return True

    @staticmethod
    def _resolve_nested_param(container: Any, path: tuple[Any, ...]) -> Any:
        current = container
        for part in path:
            if isinstance(current, BaseModel):
                if not isinstance(part, str):
                    raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
                if part not in type(current).model_fields:
                    raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
                current = getattr(current, part)
            elif isinstance(current, (list, tuple)):
                if not isinstance(part, int) or part < 0 or part >= len(current):
                    raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
                current = current[part]
            elif isinstance(current, dict):
                if part not in current:
                    raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
                current = current[part]
            else:
                raise ValueError(f"Cannot resolve nested optimizable parameter through {type(current).__name__}")
        return current

    @staticmethod
    def _apply_nested_param(container: Any, path: tuple[Any, ...], value: Any) -> Any:
        if not path:
            return value

        part, rest = path[0], path[1:]
        if isinstance(container, BaseModel):
            if not isinstance(part, str):
                raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
            if part not in type(container).model_fields:
                raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
            updated_child = BaseNode._apply_nested_param(getattr(container, part), rest, value)
            return container.model_copy(update={part: updated_child})

        if isinstance(container, list):
            if not isinstance(part, int) or part < 0 or part >= len(container):
                raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
            updated_list = list(container)
            updated_list[part] = BaseNode._apply_nested_param(updated_list[part], rest, value)
            return updated_list

        if isinstance(container, tuple):
            if not isinstance(part, int) or part < 0 or part >= len(container):
                raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
            updated_list = list(container)
            updated_list[part] = BaseNode._apply_nested_param(updated_list[part], rest, value)
            return tuple(updated_list)

        if isinstance(container, dict):
            if part not in container:
                raise ValueError(f"Unknown nested optimizable parameter: {part!r}")
            updated_dict = dict(container)
            updated_dict[part] = BaseNode._apply_nested_param(updated_dict[part], rest, value)
            return updated_dict

        raise ValueError(f"Cannot apply nested optimizable parameter through {type(container).__name__}")

    def supports_optimizable_param(self, param_name: str) -> bool:
        try:
            parts = self._split_optimizable_param(param_name)
        except ValueError:
            return False
        if not parts:
            return False

        root = parts[0]
        if not isinstance(root, str):
            return False
        if root not in type(self).model_fields:
            return False
        if len(parts) == 1:
            return True

        container = getattr(self, root, None)
        if container is None:
            return False
        return self._has_nested_param(container, parts[1:])

    def resolve_optimizable_param_value(self, param_name: str) -> Any:
        parts = self._split_optimizable_param(param_name)
        if not parts:
            raise ValueError("optimizable parameter path is empty")

        root = parts[0]
        if not isinstance(root, str):
            raise ValueError(f"Unknown optimizable parameter: {param_name!r}")
        if root not in type(self).model_fields:
            raise ValueError(f"Unknown optimizable parameter: {param_name!r}")
        if len(parts) == 1:
            return getattr(self, root)

        container = getattr(self, root, None)
        if container is None:
            raise ValueError(f"Optimizable parameter root {root!r} is None")
        return self._resolve_nested_param(container, parts[1:])

    def apply_optimizable_param_value(self, param_name: str, value: Any) -> BaseNode:
        parts = self._split_optimizable_param(param_name)
        if not parts:
            raise ValueError("optimizable parameter path is empty")

        root = parts[0]
        if not isinstance(root, str):
            raise ValueError(f"Unknown optimizable parameter: {param_name!r}")
        if root not in type(self).model_fields:
            raise ValueError(f"Unknown optimizable parameter: {param_name!r}")
        if len(parts) == 1:
            return self.model_copy(update={root: value})

        container = getattr(self, root, None)
        if container is None:
            raise ValueError(f"Optimizable parameter root {root!r} is None")
        updated_container = self._apply_nested_param(container, parts[1:], value)
        return self.model_copy(update={root: updated_container})

    def optimizable_param_description(self, param_name: str) -> str:
        parts = self._split_optimizable_param(param_name)
        if not parts:
            return ""

        root_field = type(self).model_fields.get(parts[0])
        description = root_field.description if root_field is not None and root_field.description else ""
        if len(parts) == 1:
            return description

        current: Any = getattr(self, parts[0], None)
        for part in parts[1:]:
            if not isinstance(current, BaseModel):
                break
            if not isinstance(part, str):
                break
            field = type(current).model_fields.get(part)
            if field is None:
                break
            if field.description:
                description = field.description
            current = getattr(current, part)
        return description

    # -- Display hooks --------------------------------------------------

    def signature(self) -> str:
        """One-line human-readable summary used by ``LazyFrame.show()``."""
        return f"{type(self).__name__}(...)"

    def child_lfs(self) -> list[tuple[str, Any]]:
        """Return ``(label, LazyFrame)`` pairs for nested LazyFrame display."""
        return []


# ------------------------------------------------------------------
# Source Node
# ------------------------------------------------------------------


class SourceNode(BaseNode):
    """Source node representing input data."""

    lazyframe_ref: Any = None
    df: pd.DataFrame | None = None
    expected_schema: dict[str, str] | None = None

    def __call__(
        self, df: pd.DataFrame | None = None, resolver: Resolver = _no_resolver, **context: Any
    ) -> pd.DataFrame:
        """Return the source DataFrame."""
        lotus.logger.debug("SourceNode: loading source")
        if df is not None:
            lotus.logger.debug(f"SourceNode: loaded {len(df)} rows from provided df")
            if self.expected_schema is not None:
                self._validate_schema(df)
            return df
        if self.df is not None:
            lotus.logger.debug(f"SourceNode: loaded {len(self.df)} rows from bound df")
            if self.expected_schema is not None:
                self._validate_schema(self.df)
            return self.df
        raise ValueError("No DataFrame provided for source")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        if not self.expected_schema:
            return
        for col_name, expected_dtype in self.expected_schema.items():
            if col_name not in df.columns:
                raise ValueError(f"Schema validation failed: column '{col_name}' not found in DataFrame")
            actual_dtype = str(df[col_name].dtype)
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"Schema validation failed: column '{col_name}' has dtype '{actual_dtype}', "
                    f"expected '{expected_dtype}'"
                )

    def signature(self) -> str:
        schema_str = f", schema={len(self.expected_schema)} cols" if self.expected_schema else ""
        return f"Source(bound={self.df is not None}{schema_str})"


# ------------------------------------------------------------------
# Semantic Operator Nodes
# ------------------------------------------------------------------


class SemFilterNode(BaseNode):
    """Filters rows where a natural language predicate evaluates to true."""

    _HELPER_FILTER_INSTRUCTION_PARAM: str = "cascade_args.helper_filter_instruction"

    user_instruction: str = Field(
        description="Natural language predicate evaluated per row. Use {ColumnName} to reference columns."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prepended to every LLM call for this filter operation."
    )
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
    output_tokens: tuple[str, str] = ("True", "False")

    def supports_optimizable_param(self, param_name: str) -> bool:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            return self.cascade_args is not None and self.cascade_args.proxy_model == ProxyModel.HELPER_LM
        return super().supports_optimizable_param(param_name)

    def resolve_optimizable_param_value(self, param_name: str) -> Any:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            if self.cascade_args is None:
                raise ValueError("cascade_args is required to resolve helper filter instruction")
            return self.cascade_args.helper_filter_instruction or self.user_instruction
        return super().resolve_optimizable_param_value(param_name)

    def apply_optimizable_param_value(self, param_name: str, value: Any) -> BaseNode:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            if self.cascade_args is None:
                raise ValueError("cascade_args is required to set helper filter instruction")
            updated_cascade = self.cascade_args.model_copy(update={"helper_filter_instruction": value})
            return self.model_copy(update={"cascade_args": updated_cascade})
        return super().apply_optimizable_param_value(param_name, value)

    def optimizable_param_description(self, param_name: str) -> str:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            return (
                "Instruction used by the helper model in sem_filter cascades. "
                "Defaults to user_instruction when unset."
            )
        return super().optimizable_param_description(param_name)

    def __call__(  # type: ignore
        self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        lotus.logger.debug(f"SemFilterNode: filtering {len(df)} rows")
        needs_learning = (
            self.cascade_args is not None
            and self.cascade_args.filter_pos_cascade_threshold is None
            and context.get("update_cascade_args", False)
        )

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
            return_stats=self.return_stats or needs_learning,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            additional_cot_instructions=self.additional_cot_instructions,
            system_prompt=self.system_prompt,
            output_tokens=self.output_tokens,
        )

        if needs_learning and isinstance(result, tuple):
            df_out, stats = result
            learned = stats.get("cascade_args")
            if learned is not None:
                self.cascade_args = learned
            return (df_out, stats) if self.return_stats else df_out

        return result

    def signature(self) -> str:
        return f"sem_filter({_truncate(self.user_instruction)!r})"


class SemMapNode(BaseNode):
    """Transforms each row using a natural language instruction, producing a new column."""

    user_instruction: str = Field(
        description="Natural language transformation instruction applied per row. Use {ColumnName} to reference columns."
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prepended to every LLM call for this map operation."
    )
    postprocessor: Callable[[list[str], Any, bool], SemanticMapPostprocessOutput] | None = None
    return_explanations: bool = False
    return_raw_outputs: bool = False
    suffix: str = "_map"
    examples: pd.DataFrame | None = None
    strategy: ReasoningStrategy | None = None
    safe_mode: bool = False
    progress_bar_desc: str = "Mapping"
    model_kwargs: dict[str, Any] | None = None

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemMapNode: mapping {len(df)} rows")
        kwargs: dict[str, Any] = {}
        if self.postprocessor is not None:
            kwargs["postprocessor"] = self.postprocessor
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)

        return df.sem_map(
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

    def signature(self) -> str:
        return f"sem_map({_truncate(self.user_instruction)!r})"


class SemExtractNode(BaseNode):
    """Extracts structured attributes from text columns into new columns."""

    input_cols: list[str]
    output_cols: dict[str, str | None] = Field(
        description="Mapping of output column names to natural language descriptions of what to extract."
    )
    extract_quotes: bool = False
    postprocessor: Callable[[list[str], Any, bool], SemanticExtractPostprocessOutput] | None = None
    return_raw_outputs: bool = False
    safe_mode: bool = False
    progress_bar_desc: str = "Extracting"
    return_explanations: bool = False
    strategy: ReasoningStrategy | None = None

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemExtractNode: extracting from {len(df)} rows")
        kwargs: dict[str, Any] = {}
        if self.postprocessor is not None:
            kwargs["postprocessor"] = self.postprocessor

        return df.sem_extract(
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

    def signature(self) -> str:
        return f"sem_extract({self.input_cols!r}, {self.output_cols!r})"


class SemAggNode(BaseNode):
    """Aggregates/summarizes rows using a natural language instruction."""

    user_instruction: str = Field(
        description="Natural language aggregation instruction describing how to summarize the rows. Use {ColumnName} to reference columns."
    )
    all_cols: bool = False
    suffix: str = "_output"
    group_by: list[str] | None = None
    safe_mode: bool = False
    progress_bar_desc: str = "Aggregating"
    long_context_strategy: LongContextStrategy | None = LongContextStrategy.CHUNK
    response_format: type[BaseModel] | dict | None = None
    split_fields_into_cols: bool = True

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemAggNode: aggregating {len(df)} rows")
        return df.sem_agg(
            self.user_instruction,
            all_cols=self.all_cols,
            suffix=self.suffix,
            group_by=self.group_by,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            long_context_strategy=self.long_context_strategy,
            response_format=self.response_format,
            split_fields_into_cols=self.split_fields_into_cols,
        )

    def signature(self) -> str:
        return f"sem_agg({_truncate(self.user_instruction)!r})"


class SemTopKNode(BaseNode):
    """Ranks rows by a natural language criterion and returns the top K."""

    user_instruction: str = Field(
        description="Natural language ranking criterion. Use {ColumnName} to reference columns."
    )
    K: int
    method: str = "quick"
    strategy: ReasoningStrategy | None = None
    group_by: list[str] | None = None
    cascade_threshold: float | None = None
    return_stats: bool = False
    safe_mode: bool = False
    return_explanations: bool = False

    def __call__(  # type: ignore
        self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        lotus.logger.debug(f"SemTopKNode: selecting top {self.K} from {len(df)} rows")
        return df.sem_topk(
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

    def signature(self) -> str:
        return f"sem_topk({_truncate(self.user_instruction)!r}, {self.K})"


class _JoinMixin(BaseModel):
    """Shared fields and logic for join nodes that reference a right-side input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    right_source_node: SourceNode | None = None
    right_lf: Any = None
    right_df: pd.DataFrame | None = None

    def _resolve_right(self, resolver: Callable[[Any], Any]) -> pd.DataFrame:
        """Resolve the right-side input to a concrete DataFrame."""
        if self.right_df is not None:
            return self.right_df
        if self.right_lf is not None:
            return resolver(self.right_lf)
        if self.right_source_node is not None:
            return resolver(self.right_source_node)
        raise ValueError("Join node has no right DataFrame specified.")

    def _right_child_lfs(self) -> list[tuple[str, Any]]:
        from .lazyframe import LazyFrame

        if self.right_lf is not None and isinstance(self.right_lf, LazyFrame):
            return [("right LazyFrame", self.right_lf)]
        return []


class SemJoinNode(_JoinMixin, BaseNode):
    """Joins two DataFrames on a natural language predicate."""

    join_instruction: str = Field(
        description="Natural language join predicate between left and right DataFrames. Use {ColumnName} to reference columns from either side."
    )
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

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        right = self._resolve_right(resolver)
        lotus.logger.debug(f"SemJoinNode: joining {len(df)} left rows with {len(right)} right rows")
        needs_learning = (
            self.cascade_args is not None
            and self.cascade_args.join_cascade_pos_threshold is None
            and context.get("update_cascade_args", False)
        )

        result = df.sem_join(
            right,
            self.join_instruction,
            return_explanations=self.return_explanations,
            how=self.how,
            suffix=self.suffix,
            examples=self.examples,
            strategy=self.strategy,
            default=self.default,
            cascade_args=self.cascade_args,
            return_stats=self.return_stats or needs_learning,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
        )

        if needs_learning and isinstance(result, tuple):
            df_out, stats = result
            learned = stats.get("cascade_args")
            if learned is not None:
                self.cascade_args = learned
            return (df_out, stats) if self.return_stats else df_out

        return result

    def signature(self) -> str:
        return (
            f"sem_join(join_instruction={_truncate(self.join_instruction)!r}, "
            f"how={self.how!r}, suffix={self.suffix!r})"
        )

    def child_lfs(self) -> list[tuple[str, Any]]:
        return self._right_child_lfs()


class SemSimJoinNode(_JoinMixin, BaseNode):
    """Semantic similarity join node."""

    left_on: str
    right_on: str
    K: int
    lsuffix: str = ""
    rsuffix: str = ""
    score_suffix: str = ""
    keep_index: bool = False

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        right = self._resolve_right(resolver)
        lotus.logger.debug(f"SemSimJoinNode: joining {len(df)} left rows with {len(right)} right rows, K={self.K}")
        return df.sem_sim_join(
            right,
            left_on=self.left_on,
            right_on=self.right_on,
            K=self.K,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            score_suffix=self.score_suffix,
            keep_index=self.keep_index,
        )

    def signature(self) -> str:
        return f"sem_sim_join(left_on={self.left_on!r}, right_on={self.right_on!r}, K={self.K})"

    def child_lfs(self) -> list[tuple[str, Any]]:
        return self._right_child_lfs()


class SemSearchNode(BaseNode):
    """Returns rows most semantically similar to a natural language query."""

    col_name: str
    query: str = Field(description="Natural language query for semantic similarity search.")
    K: int | None = None
    n_rerank: int | None = None
    return_scores: bool = False
    suffix: str = "_sim_score"

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemSearchNode: searching {len(df)} rows")
        return df.sem_search(
            self.col_name,
            self.query,
            K=self.K,
            n_rerank=self.n_rerank,
            return_scores=self.return_scores,
            suffix=self.suffix,
        )

    def signature(self) -> str:
        return f"sem_search({self.col_name!r}, {self.query!r})"


class SemIndexNode(BaseNode):
    """Semantic index node."""

    col_name: str
    index_dir: str

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemIndexNode: indexing column '{self.col_name}' to {self.index_dir}")
        return df.sem_index(self.col_name, self.index_dir)

    def signature(self) -> str:
        return f"sem_index({self.col_name!r}, {self.index_dir!r})"


class LoadSemIndexNode(BaseNode):
    """Load semantic index metadata node."""

    col_name: str
    index_dir: str

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"LoadSemIndexNode: loading index for column '{self.col_name}' from {self.index_dir}")
        return df.load_sem_index(self.col_name, self.index_dir)

    def signature(self) -> str:
        return f"load_sem_index({self.col_name!r}, {self.index_dir!r})"


class SemClusterByNode(BaseNode):
    """Semantic cluster node."""

    col_name: str
    ncentroids: int
    return_scores: bool = False
    return_centroids: bool = False
    niter: int = 20
    verbose: bool = False

    def __call__(  # type: ignore
        self,
        df: pd.DataFrame,
        resolver: Resolver = _no_resolver,
        **context: Any,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        lotus.logger.debug(f"SemClusterByNode: clustering {len(df)} rows into {self.ncentroids} clusters")
        return df.sem_cluster_by(
            self.col_name,
            ncentroids=self.ncentroids,
            return_scores=self.return_scores,
            return_centroids=self.return_centroids,
            niter=self.niter,
            verbose=self.verbose,
        )

    def signature(self) -> str:
        return f"sem_cluster_by({self.col_name!r}, {self.ncentroids})"


class SemDedupNode(BaseNode):
    """Semantic deduplication node."""

    col_name: str
    threshold: float

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemDedupNode: deduplicating {len(df)} rows with threshold={self.threshold}")
        return df.sem_dedup(self.col_name, self.threshold)

    def signature(self) -> str:
        return f"sem_dedup({self.col_name!r}, {self.threshold})"


class SemPartitionByNode(BaseNode):
    """Semantic partition node."""

    partition_fn: Callable[[pd.DataFrame], list[int]]

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"SemPartitionByNode: partitioning {len(df)} rows")
        return df.sem_partition_by(self.partition_fn)

    def signature(self) -> str:
        return "sem_partition_by(...)"


# ------------------------------------------------------------------
# Pandas Operator Nodes
# ------------------------------------------------------------------


class PandasFilterNode(BaseNode):
    """Pandas boolean filter node."""

    predicate: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"PandasFilterNode: filtering {len(df)} rows")
        return df[self.predicate(df)]

    def signature(self) -> str:
        return "filter(...)"


class PandasOpNode(BaseNode):
    """Generic pandas operation node.

    Handles method calls (``df.sort_values(...)``), attribute access
    (``df.columns``), subscript (``df[['col']]``), and column assignment
    (``df.assign(col=value)``).  LazyFrame references in *args*/*kwargs* are
    stored in ``lf_args``/``lf_kwargs`` and resolved via ``resolver`` at
    call time.
    """

    op_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    is_attr: bool = False
    lf_args: dict[str, Any] | None = None
    lf_kwargs: dict[str, Any] | None = None

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame | Any:  # type: ignore
        if self.is_attr:
            lotus.logger.debug(f"PandasOpNode: accessing attribute '{self.op_name}'")
            return getattr(df, self.op_name)

        args = list(self.args)
        kwargs = dict(self.kwargs or {})

        if self.lf_args:
            for key, lf in self.lf_args.items():
                args[int(key.split("_")[-1])] = resolver(lf)
        if self.lf_kwargs:
            for key, lf in self.lf_kwargs.items():
                kwargs[key.replace("_lf_kwarg_", "")] = resolver(lf)

        lotus.logger.debug(f"PandasOpNode: calling '{self.op_name}' on {len(df)} rows")
        result = getattr(df, self.op_name)(*args, **kwargs)
        return result if result is not None else df

    def signature(self) -> str:
        if self.is_attr:
            return self.op_name
        if self.op_name == "__getitem__":
            return f"[{self.args[0]!r}]"
        if self.op_name == "assign":
            cols = list((self.kwargs or {}).keys())
            if self.lf_kwargs:
                for key in self.lf_kwargs:
                    name = key.replace("_lf_kwarg_", "")
                    if name not in cols:
                        cols.append(name)
            return f"assign({', '.join(cols)}=...)"
        args_str = ", ".join(repr(a) for a in self.args) if self.args else ""
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in (self.kwargs or {}).items()) if self.kwargs else ""
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{self.op_name}({all_args})"

    def child_lfs(self) -> list[tuple[str, Any]]:
        from .lazyframe import LazyFrame

        refs: list[tuple[str, Any]] = []
        if self.lf_args:
            for key, lf in self.lf_args.items():
                if isinstance(lf, LazyFrame):
                    refs.append((f"arg {key}", lf))
        if self.lf_kwargs:
            for key, lf in self.lf_kwargs.items():
                if isinstance(lf, LazyFrame):
                    name = key.replace("_lf_kwarg_", "")
                    refs.append((f"kwarg {name}", lf))
        return refs


# ------------------------------------------------------------------
# Eval Nodes
# ------------------------------------------------------------------


class LLMAsJudgeNode(BaseNode):
    """Evaluates rows using an LLM-as-judge approach (multi-trial sem_map)."""

    judge_instruction: str = Field(
        description="Natural language instruction guiding the judging process. Use {ColumnName} to reference columns."
    )
    response_format: Any = None
    n_trials: int = 1
    system_prompt: str | None = None
    postprocessor: Callable[[list[str], Any, bool], SemanticMapPostprocessOutput] | None = None
    return_raw_outputs: bool = False
    return_explanations: bool = False
    suffix: str = "_judge"
    examples: pd.DataFrame | None = None
    cot_reasoning: list[str] | None = None
    strategy: ReasoningStrategy | None = None
    extra_cols_to_include: list[str] | None = None
    safe_mode: bool = False
    progress_bar_desc: str = "Evaluating"
    model_kwargs: dict[str, Any] | None = None

    def __call__(self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any) -> pd.DataFrame:  # type: ignore
        lotus.logger.debug(f"LLMAsJudgeNode: judging {len(df)} rows")
        kwargs: dict[str, Any] = {}
        if self.postprocessor is not None:
            kwargs["postprocessor"] = self.postprocessor
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)

        return df.llm_as_judge(
            self.judge_instruction,
            response_format=self.response_format,
            n_trials=self.n_trials,
            system_prompt=self.system_prompt,
            return_raw_outputs=self.return_raw_outputs,
            return_explanations=self.return_explanations,
            suffix=self.suffix,
            examples=self.examples,
            cot_reasoning=self.cot_reasoning,
            strategy=self.strategy,
            extra_cols_to_include=self.extra_cols_to_include,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            **kwargs,
        )

    def signature(self) -> str:
        return f"llm_as_judge({_truncate(self.judge_instruction)!r}, n_trials={self.n_trials})"


class PairwiseJudgeNode(BaseNode):
    """Judge the given df's col1 and col2, based on the judging criteria, context and grading scale."""

    _HELPER_FILTER_INSTRUCTION_PARAM: str = "cascade_args.helper_filter_instruction"

    col1: str
    col2: str
    judge_instruction: str = Field(
        description="Natural language instruction guiding the pairwise judging. Use {ColumnName} to reference columns."
    )
    n_trials: int = 1
    permute_cols: bool = False
    system_prompt: str | None = None
    return_raw_outputs: bool = False
    return_explanations: bool = False
    default_to_col1: bool = True
    suffix: str = "_judge"
    examples: pd.DataFrame | None = None
    helper_examples: pd.DataFrame | None = None
    strategy: ReasoningStrategy | None = None
    cascade_args: CascadeArgs | None = None
    return_stats: bool = False
    safe_mode: bool = False
    progress_bar_desc: str = "Evaluating"
    additional_cot_instructions: str = ""
    model_kwargs: dict[str, Any] | None = None

    def _effective_sem_filter_user_instruction(self) -> str:
        renamed_instr = self.judge_instruction.replace(f"{{{self.col1}}}", "{col_A}").replace(
            f"{{{self.col2}}}", "{col_B}"
        )
        return f"{{col_A}} is better than {{col_B}} given the criteria: {renamed_instr}"

    def supports_optimizable_param(self, param_name: str) -> bool:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            return self.cascade_args is not None and self.cascade_args.proxy_model == ProxyModel.HELPER_LM
        return super().supports_optimizable_param(param_name)

    def resolve_optimizable_param_value(self, param_name: str) -> Any:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            if self.cascade_args is None:
                raise ValueError("cascade_args is required to resolve helper filter instruction")
            return self.cascade_args.helper_filter_instruction or self._effective_sem_filter_user_instruction()
        return super().resolve_optimizable_param_value(param_name)

    def apply_optimizable_param_value(self, param_name: str, value: Any) -> BaseNode:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            if self.cascade_args is None:
                raise ValueError("cascade_args is required to set helper filter instruction")
            updated_cascade = self.cascade_args.model_copy(update={"helper_filter_instruction": value})
            return self.model_copy(update={"cascade_args": updated_cascade})
        return super().apply_optimizable_param_value(param_name, value)

    def optimizable_param_description(self, param_name: str) -> str:
        if param_name == self._HELPER_FILTER_INSTRUCTION_PARAM:
            return (
                "Instruction used by the helper model in pairwise_judge sem_filter cascades. "
                "Defaults to the generated pairwise sem_filter instruction when unset."
            )
        return super().optimizable_param_description(param_name)

    def __call__(  # type: ignore
        self, df: pd.DataFrame, resolver: Resolver = _no_resolver, **context: Any
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        lotus.logger.debug(f"PairwiseJudgeNode: judging {len(df)} rows ({self.col1} vs {self.col2})")
        needs_learning = (
            self.cascade_args is not None
            and self.cascade_args.filter_pos_cascade_threshold is None
            and context.get("update_cascade_args", False)
        )

        kwargs: dict[str, Any] = {}
        if self.model_kwargs:
            kwargs.update(self.model_kwargs)

        result = df.pairwise_judge(
            col1=self.col1,
            col2=self.col2,
            judge_instruction=self.judge_instruction,
            n_trials=self.n_trials,
            permute_cols=self.permute_cols,
            system_prompt=self.system_prompt,
            return_raw_outputs=self.return_raw_outputs,
            return_explanations=self.return_explanations,
            suffix=self.suffix,
            examples=self.examples,
            strategy=self.strategy,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
            default_to_col1=self.default_to_col1,
            helper_examples=self.helper_examples,
            cascade_args=self.cascade_args,
            return_stats=self.return_stats or needs_learning,
            additional_cot_instructions=self.additional_cot_instructions,
            **kwargs,
        )

        if needs_learning and isinstance(result, tuple):
            df_out, all_stats = result
            learned = next(
                (stats.get("cascade_args") for stats in all_stats if stats.get("cascade_args") is not None), None
            )
            if learned is not None:
                self.cascade_args = learned
            return (df_out, all_stats) if self.return_stats else df_out

        return result

    def signature(self) -> str:
        return f"pairwise_judge({self.col1!r}, {self.col2!r}, " f"{_truncate(self.judge_instruction)!r})"


# ------------------------------------------------------------------
# Function Nodes
# ------------------------------------------------------------------


class ApplyFnNode(BaseNode):
    """Node for callables that consume resolved LazyFrame outputs.

    Used by ``LazyFrame.from_fn`` and ``LazyFrame.concat``.  Arguments may
    contain LazyFrame references nested in lists, tuples, or dicts; all are
    resolved recursively before the function is called.
    """

    fn: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None

    def __call__(self, df: pd.DataFrame | None = None, resolver: Resolver = _no_resolver, **context: Any) -> Any:  # type: ignore
        args = tuple(resolver(a) for a in self.args)
        kwargs = {k: resolver(v) for k, v in (self.kwargs or {}).items()}
        lotus.logger.debug(f"ApplyFnNode: calling {getattr(self.fn, '__name__', self.fn)}")
        return self.fn(*args, **kwargs)

    def signature(self) -> str:
        fn_name = getattr(self.fn, "__name__", repr(self.fn))
        args_repr = [repr(arg) for arg in self.args] if self.args else []
        kwargs_repr = [f"{k}={v!r}" for k, v in (self.kwargs or {}).items()]
        all_args = ", ".join(args_repr + kwargs_repr)
        return f"{fn_name}({all_args})"

    def child_lfs(self) -> list[tuple[str, Any]]:
        from .lazyframe import LazyFrame

        refs: list[tuple[str, Any]] = []

        def _scan(value: Any, label: str) -> None:
            if isinstance(value, LazyFrame):
                refs.append((label, value))
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    _scan(item, f"{label}[{idx}]")
            elif isinstance(value, dict):
                for key, item in value.items():
                    _scan(item, f"{label}.{key}")

        for i, arg in enumerate(self.args):
            _scan(arg, f"arg {i}")
        if self.kwargs:
            for key, val in self.kwargs.items():
                _scan(val, f"kwarg {key}")

        return refs
