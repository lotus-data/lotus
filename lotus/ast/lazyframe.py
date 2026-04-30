"""LazyFrame builder for LOTUS AST operations.

``LazyFrame`` is an immutable builder that records a sequence of semantic and
pandas operations as AST nodes.  Nothing is executed until ``.execute()`` (or
``.run().execute()``) is called.

Example::

    lf = LazyFrame().sem_filter("{text} is about sports").sem_map("Summarize {text}")
    result = lf.execute(df)
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from pydantic import BaseModel

import lotus
from lotus.cache import Cache, CacheFactory
from lotus.types import (
    CascadeArgs,
    LongContextStrategy,
    ReasoningStrategy,
    SemanticExtractPostprocessOutput,
    SemanticMapPostprocessOutput,
)

from .nodes import (
    ApplyFnNode,
    BaseNode,
    LLMAsJudgeNode,
    LoadSemIndexNode,
    PairwiseJudgeNode,
    PandasFilterNode,
    PandasOpNode,
    SemAggNode,
    SemClusterByNode,
    SemDedupNode,
    SemExtractNode,
    SemFilterNode,
    SemIndexNode,
    SemJoinNode,
    SemMapNode,
    SemPartitionByNode,
    SemSearchNode,
    SemSimJoinNode,
    SemTopKNode,
    SourceNode,
)

if TYPE_CHECKING:
    from .optimizer.base import BaseOptimizer
    from .run import LazyFrameRun


class _LazyMethodProxy:
    """Capture pandas method calls and append a corresponding AST node."""

    def __init__(self, lazyframe: "LazyFrame", method_name: str) -> None:
        self._lazyframe = lazyframe
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> "LazyFrame":
        processed_args, lf_args = LazyFrame._split_args(args)
        processed_kwargs, lf_kwargs = LazyFrame._split_kwargs(kwargs)

        node = PandasOpNode(
            op_name=self._method_name,
            args=processed_args,
            kwargs=processed_kwargs if processed_kwargs else None,
            is_attr=False,
            lf_args=lf_args if lf_args else None,
            lf_kwargs=lf_kwargs if lf_kwargs else None,
        )
        return self._lazyframe._append_node(node)


class LazyFrame:
    """Immutable lazy version of DataFrame and semantic operators.

    Operations are recorded as AST nodes and only materialised when
    ``.execute()`` is called.

    Args:
        df: Optional bound DataFrame.  When provided the source data is
            stored directly on the LazyFrame so no external input is
            required at execution time.
        schema: Optional ``{col_name: dtype}`` dict validated at execution
            time against the source DataFrame.
        _nodes: Internal — pre-built node list (used by copy/optimise).
        _source: Internal — explicit source node reference.

    Example::

        >>> lf = LazyFrame().sem_filter("{text} is about sports").sem_map("Summarize {text}")
        >>> result = lf.execute(df)
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        *,
        schema: dict[str, str] | None = None,
        _nodes: list[BaseNode] | None = None,
        _source: SourceNode | None = None,
        _default_cache: Cache | None = None,
    ) -> None:
        self._default_cache: Cache = _default_cache or CacheFactory.create_default_cache(max_size=10_000)
        if _nodes is not None:
            self._nodes = list(_nodes)
            self._source: SourceNode | None = (
                _source
                if _source is not None
                else (self._nodes[0] if self._nodes and isinstance(self._nodes[0], SourceNode) else None)
            )
        else:
            source_node = SourceNode(lazyframe_ref=self, df=df, expected_schema=schema)
            self._nodes = [source_node]
            self._source = source_node

    def _append_node(self, node: BaseNode) -> "LazyFrame":
        """Return a new LazyFrame with the node appended (immutable)."""
        lotus.logger.debug(f"LazyFrame: appending {type(node).__name__}")
        new_nodes = list(self._nodes)
        new_nodes.append(node)
        return LazyFrame(_nodes=new_nodes, _source=self._source, _default_cache=self._default_cache)

    # -- Helper: split LazyFrame refs out of args / kwargs ---------------

    @staticmethod
    def _split_args(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, "LazyFrame"]]:
        processed_args: list[Any] = []
        lf_args: dict[str, LazyFrame] = {}
        for idx, arg in enumerate(args):
            if isinstance(arg, LazyFrame):
                lf_args[f"_lf_arg_{idx}"] = arg
                processed_args.append(None)
            else:
                processed_args.append(arg)
        return tuple(processed_args), lf_args

    @staticmethod
    def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, "LazyFrame"]]:
        processed_kwargs: dict[str, Any] = {}
        lf_kwargs: dict[str, LazyFrame] = {}
        for key, value in kwargs.items():
            if isinstance(value, LazyFrame):
                lf_kwargs[f"_lf_kwarg_{key}"] = value
                processed_kwargs[key] = None
            else:
                processed_kwargs[key] = value
        return processed_kwargs, lf_kwargs

    @staticmethod
    def _split_right_input(right: "LazyFrame" | pd.DataFrame) -> tuple["LazyFrame" | None, pd.DataFrame | None]:
        if isinstance(right, LazyFrame):
            return right, None
        if isinstance(right, pd.DataFrame):
            return None, right
        raise TypeError(f"right must be LazyFrame or DataFrame, got {type(right)}")

    def copy(self) -> "LazyFrame":
        """Return a deep copy of this LazyFrame.

        ``SourceNode.lazyframe_ref`` values are restored to match the original
        graph so copied pipelines still resolve ``dict[LazyFrame, DataFrame]``
        inputs correctly (including nested child LazyFrames).
        """
        copied = LazyFrame(_nodes=deepcopy(self._nodes), _default_cache=self._default_cache)
        self._restore_source_refs(self._nodes, copied._nodes, set())
        return copied

    @staticmethod
    def _restore_source_refs(original: Any, copied: Any, seen: set[tuple[int, int]]) -> None:
        """Recursively restore SourceNode lazyframe refs after deep copy."""
        pair = (id(original), id(copied))
        if pair in seen:
            return
        seen.add(pair)

        if isinstance(original, SourceNode) and isinstance(copied, SourceNode):
            copied.lazyframe_ref = original.lazyframe_ref
            return

        if isinstance(original, LazyFrame) and isinstance(copied, LazyFrame):
            LazyFrame._restore_source_refs(original._nodes, copied._nodes, seen)
            return

        if isinstance(original, BaseNode) and isinstance(copied, BaseNode):
            for field_name in type(original).model_fields:
                LazyFrame._restore_source_refs(
                    getattr(original, field_name, None),
                    getattr(copied, field_name, None),
                    seen,
                )
            return

        if isinstance(original, list) and isinstance(copied, list):
            for original_item, copied_item in zip(original, copied):
                LazyFrame._restore_source_refs(original_item, copied_item, seen)
            return

        if isinstance(original, tuple) and isinstance(copied, tuple):
            for original_item, copied_item in zip(original, copied):
                LazyFrame._restore_source_refs(original_item, copied_item, seen)
            return

        if isinstance(original, dict) and isinstance(copied, dict):
            for key in original.keys() & copied.keys():
                LazyFrame._restore_source_refs(original[key], copied[key], seen)

    # ------------------------------------------------------------------
    # Source Management
    # ------------------------------------------------------------------

    def add_source(self, df: pd.DataFrame | None = None, schema: dict[str, str] | None = None) -> "LazyFrame":
        """Set the LazyFrame source (optional bound DataFrame and schema).

        Replaces the single source node. Use this to bind a df or add schema validation.
        """
        source_node = SourceNode(lazyframe_ref=self, df=df, expected_schema=schema)
        new_nodes: list[BaseNode] = [source_node] + list(self._nodes[1:]) if len(self._nodes) > 1 else [source_node]
        return LazyFrame(_nodes=new_nodes, _source=source_node, _default_cache=self._default_cache)

    # ------------------------------------------------------------------
    # Optimization Annotations
    # ------------------------------------------------------------------

    def mark_optimizable(self, node_idx: int, params: list[str]) -> "LazyFrame":
        """Mark specific parameters on a node for GEPA optimization.

        Args:
            node_idx: Index of the node in the LazyFrame's node list.
            params: List of parameter names to optimize, e.g. ["user_instruction"].
                    Pass an empty list to explicitly exclude the node from optimization.

        Returns:
            New LazyFrame with the targeted node annotated.
        """
        if node_idx < 0 or node_idx >= len(self._nodes):
            raise IndexError(f"node_idx {node_idx} out of range for LazyFrame with {len(self._nodes)} nodes")
        node = self._nodes[node_idx]
        self._validate_optimizable_paths(node, params)
        updated = node.model_copy(update={"optimizable_params": frozenset(params)})
        new_nodes = list(self._nodes)
        new_nodes[node_idx] = updated
        return LazyFrame(_nodes=new_nodes, _source=self._source, _default_cache=self._default_cache)

    @staticmethod
    def _validate_optimizable_paths(node: BaseNode, params: list[str]) -> None:
        unsupported = [param for param in params if not node.supports_optimizable_param(param)]
        if unsupported:
            raise ValueError(
                f"Node {type(node).__name__} does not support optimizable parameter path(s): {unsupported}"
            )

    def _append_node_with_optimizable(self, node: BaseNode, mark_optimizable: list[str] | None) -> "LazyFrame":
        """Append a node, optionally setting optimizable_params."""
        if mark_optimizable is not None:
            self._validate_optimizable_paths(node, mark_optimizable)
            node = node.model_copy(update={"optimizable_params": frozenset(mark_optimizable)})
        return self._append_node(node)

    # ------------------------------------------------------------------
    # Semantic Operators
    # ------------------------------------------------------------------

    def sem_filter(
        self,
        user_instruction: str,
        *,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        return_all: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: pd.DataFrame | None = None,
        helper_examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Filtering",
        additional_cot_instructions: str = "",
        system_prompt: str | None = None,
        output_tokens: tuple[str, str] = ("True", "False"),
        mark_optimizable: list[str] | None = None,
    ) -> "LazyFrame":
        """Add a semantic filter operation."""
        node = SemFilterNode(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            return_raw_outputs=return_raw_outputs,
            return_explanations=return_explanations,
            return_all=return_all,
            default=default,
            suffix=suffix,
            examples=examples,
            helper_examples=helper_examples,
            strategy=strategy,
            cascade_args=cascade_args,
            return_stats=return_stats,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            additional_cot_instructions=additional_cot_instructions,
            output_tokens=output_tokens,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_map(
        self,
        user_instruction: str,
        *,
        system_prompt: str | None = None,
        postprocessor: Callable[[list[str], Any, bool], SemanticMapPostprocessOutput] | None = None,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        suffix: str = "_map",
        examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Mapping",
        mark_optimizable: list[str] | None = None,
        **model_kwargs: Any,
    ) -> "LazyFrame":
        """Add a semantic map operation."""
        node = SemMapNode(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            postprocessor=postprocessor,
            return_explanations=return_explanations,
            return_raw_outputs=return_raw_outputs,
            suffix=suffix,
            examples=examples,
            strategy=strategy,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_extract(
        self,
        input_cols: list[str],
        output_cols: dict[str, str | None],
        *,
        extract_quotes: bool = False,
        postprocessor: Callable[[list[str], Any, bool], SemanticExtractPostprocessOutput] | None = None,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        strategy: ReasoningStrategy | None = None,
        mark_optimizable: list[str] | None = None,
    ) -> "LazyFrame":
        """Add a semantic extract operation."""
        node = SemExtractNode(
            input_cols=input_cols,
            output_cols=output_cols,
            extract_quotes=extract_quotes,
            postprocessor=postprocessor,
            return_raw_outputs=return_raw_outputs,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            return_explanations=return_explanations,
            strategy=strategy,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_agg(
        self,
        user_instruction: str,
        *,
        all_cols: bool = False,
        suffix: str = "_output",
        group_by: list[str] | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Aggregating",
        long_context_strategy: LongContextStrategy | None = LongContextStrategy.CHUNK,
        response_format: type[BaseModel] | dict | None = None,
        split_fields_into_cols: bool = True,
        mark_optimizable: list[str] | None = None,
    ) -> "LazyFrame":
        """Add a semantic aggregation operation."""
        node = SemAggNode(
            user_instruction=user_instruction,
            all_cols=all_cols,
            suffix=suffix,
            group_by=group_by,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            long_context_strategy=long_context_strategy,
            response_format=response_format,
            split_fields_into_cols=split_fields_into_cols,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_topk(
        self,
        user_instruction: str,
        K: int,
        *,
        method: str = "quick",
        strategy: ReasoningStrategy | None = None,
        group_by: list[str] | None = None,
        cascade_threshold: float | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        return_explanations: bool = False,
        mark_optimizable: list[str] | None = None,
    ) -> "LazyFrame":
        """Add a semantic top-k operation."""
        node = SemTopKNode(
            user_instruction=user_instruction,
            K=K,
            method=method,
            strategy=strategy,
            group_by=group_by,
            cascade_threshold=cascade_threshold,
            return_stats=return_stats,
            safe_mode=safe_mode,
            return_explanations=return_explanations,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_join(
        self,
        right: "LazyFrame" | pd.DataFrame,
        join_instruction: str,
        *,
        return_explanations: bool = False,
        how: str = "inner",
        suffix: str = "_join",
        examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        default: bool = True,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Join comparisons",
        mark_optimizable: list[str] | None = None,
    ) -> "LazyFrame":
        """Add a semantic join operation."""
        right_lf, right_df = self._split_right_input(right)

        node = SemJoinNode(
            right_source_node=None,
            right_lf=right_lf,
            right_df=right_df,
            join_instruction=join_instruction,
            return_explanations=return_explanations,
            how=how,
            suffix=suffix,
            examples=examples,
            strategy=strategy,
            default=default,
            cascade_args=cascade_args,
            return_stats=return_stats,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_sim_join(
        self,
        right: "LazyFrame" | pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        *,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
    ) -> "LazyFrame":
        """Add a semantic similarity join operation."""
        right_lf, right_df = self._split_right_input(right)

        node = SemSimJoinNode(
            right_source_node=None,
            right_lf=right_lf,
            right_df=right_df,
            left_on=left_on,
            right_on=right_on,
            K=K,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            score_suffix=score_suffix,
            keep_index=keep_index,
        )
        return self._append_node(node)

    def sem_search(
        self,
        col_name: str,
        query: str,
        *,
        K: int | None = None,
        n_rerank: int | None = None,
        return_scores: bool = False,
        suffix: str = "_sim_score",
        mark_optimizable: list[str] | None = None,
    ) -> "LazyFrame":
        """Add a semantic search operation."""
        node = SemSearchNode(
            col_name=col_name,
            query=query,
            K=K,
            n_rerank=n_rerank,
            return_scores=return_scores,
            suffix=suffix,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def sem_index(self, col_name: str, index_dir: str) -> "LazyFrame":
        """Add a semantic index operation."""
        node = SemIndexNode(col_name=col_name, index_dir=index_dir)
        return self._append_node(node)

    def load_sem_index(self, col_name: str, index_dir: str) -> "LazyFrame":
        """Add a semantic index load operation."""
        node = LoadSemIndexNode(col_name=col_name, index_dir=index_dir)
        return self._append_node(node)

    def sem_cluster_by(
        self,
        col_name: str,
        ncentroids: int,
        *,
        return_scores: bool = False,
        return_centroids: bool = False,
        niter: int = 20,
        verbose: bool = False,
    ) -> "LazyFrame":
        """Add a semantic clustering operation."""
        node = SemClusterByNode(
            col_name=col_name,
            ncentroids=ncentroids,
            return_scores=return_scores,
            return_centroids=return_centroids,
            niter=niter,
            verbose=verbose,
        )
        return self._append_node(node)

    def sem_dedup(self, col_name: str, threshold: float) -> "LazyFrame":
        """Add a semantic deduplication operation."""
        node = SemDedupNode(col_name=col_name, threshold=threshold)
        return self._append_node(node)

    def sem_partition_by(self, partition_fn: Callable[[pd.DataFrame], list[int]]) -> "LazyFrame":
        """Add a semantic partition operation."""
        node = SemPartitionByNode(partition_fn=partition_fn)
        return self._append_node(node)

    # ------------------------------------------------------------------
    # Eval Operators
    # ------------------------------------------------------------------

    def llm_as_judge(
        self,
        judge_instruction: str,
        *,
        response_format: Any = None,
        n_trials: int = 1,
        system_prompt: str | None = None,
        postprocessor: Callable[[list[str], Any, bool], SemanticMapPostprocessOutput] | None = None,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        suffix: str = "_judge",
        examples: pd.DataFrame | None = None,
        cot_reasoning: list[str] | None = None,
        strategy: ReasoningStrategy | None = None,
        extra_cols_to_include: list[str] | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Evaluating",
        mark_optimizable: list[str] | None = None,
        **model_kwargs: Any,
    ) -> "LazyFrame":
        """Add an LLM-as-judge evaluation operation."""
        node = LLMAsJudgeNode(
            judge_instruction=judge_instruction,
            response_format=response_format,
            n_trials=n_trials,
            system_prompt=system_prompt,
            postprocessor=postprocessor,
            return_raw_outputs=return_raw_outputs,
            return_explanations=return_explanations,
            suffix=suffix,
            examples=examples,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            extra_cols_to_include=extra_cols_to_include,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    def pairwise_judge(
        self,
        col1: str,
        col2: str,
        judge_instruction: str,
        *,
        n_trials: int = 1,
        permute_cols: bool = False,
        system_prompt: str | None = None,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        suffix: str = "_judge",
        examples: pd.DataFrame | None = None,
        strategy: ReasoningStrategy | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Evaluating",
        default_to_col1: bool = True,
        helper_examples: pd.DataFrame | None = None,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        additional_cot_instructions: str = "",
        mark_optimizable: list[str] | None = None,
        **model_kwargs: Any,
    ) -> "LazyFrame":
        """Add a pairwise judge evaluation operation."""
        node = PairwiseJudgeNode(
            col1=col1,
            col2=col2,
            judge_instruction=judge_instruction,
            n_trials=n_trials,
            permute_cols=permute_cols,
            system_prompt=system_prompt,
            return_raw_outputs=return_raw_outputs,
            return_explanations=return_explanations,
            suffix=suffix,
            examples=examples,
            strategy=strategy,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            default_to_col1=default_to_col1,
            helper_examples=helper_examples,
            cascade_args=cascade_args,
            return_stats=return_stats,
            additional_cot_instructions=additional_cot_instructions,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        return self._append_node_with_optimizable(node, mark_optimizable)

    # ------------------------------------------------------------------
    # Pandas Operations
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> "LazyFrame":
        """Add a pandas boolean filter operation."""
        node = PandasFilterNode(predicate=predicate)
        return self._append_node(node)

    def __getattr__(self, name: str) -> Any:
        """Intercept pandas DataFrame method/attribute access."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if hasattr(pd.DataFrame, name):
            df_attr = getattr(pd.DataFrame, name)
            if callable(df_attr):
                return _LazyMethodProxy(self, name)
            node = PandasOpNode(op_name=name, is_attr=True)
            return self._append_node(node)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Any) -> "LazyFrame":
        """Support column selection via ``lf['col']`` or ``lf[['col1', 'col2']]``."""
        if callable(key):
            return self.filter(key)

        node = PandasOpNode(op_name="__getitem__", args=(key,))
        return self._append_node(node)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support column assignment via ``lf['col'] = value``.

        This mutates the LazyFrame in place.  For immutable behaviour use
        ``.assign(col=value)``.
        """
        if isinstance(value, LazyFrame):
            lotus.logger.debug(f"LazyFrame.__setitem__: storing LazyFrame reference for column '{key}'")
            node = PandasOpNode(
                op_name="assign",
                kwargs={key: None},
                lf_kwargs={f"_lf_kwarg_{key}": value},
            )
        else:
            node = PandasOpNode(op_name="assign", kwargs={key: value})
        self._nodes.append(node)

    def assign(self, **kwargs: Any) -> "LazyFrame":
        """Add column assignments and return a new LazyFrame.

        Values may be scalars, callables ``(df -> Series)``, or other
        LazyFrame instances (resolved lazily at execution time).
        """
        regular: dict[str, Any] = {}
        lf_kwargs: dict[str, LazyFrame] = {}

        for col, val in kwargs.items():
            if isinstance(val, LazyFrame):
                lotus.logger.debug(f"LazyFrame.assign: storing LazyFrame reference for column '{col}'")
                lf_kwargs[f"_lf_kwarg_{col}"] = val
                regular[col] = None
            else:
                regular[col] = val

        node = PandasOpNode(
            op_name="assign",
            kwargs=regular if regular else None,
            lf_kwargs=lf_kwargs if lf_kwargs else None,
        )
        return self._append_node(node)

    # ------------------------------------------------------------------
    # LazyFrame Functions (concat / from_fn)
    # ------------------------------------------------------------------

    @classmethod
    def from_fn(
        cls,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> "LazyFrame":
        """Create a LazyFrame node that applies a callable to resolved inputs."""
        node = ApplyFnNode(
            fn=fn,
            args=args,
            kwargs=kwargs if kwargs else None,
        )
        return cls(_nodes=[node], _source=None)

    @classmethod
    def concat(
        cls,
        objs: list["LazyFrame"] | "LazyFrame",
        **kwargs: Any,
    ) -> "LazyFrame":
        """Concatenate one or more LazyFrame results via ``pd.concat``."""
        if isinstance(objs, LazyFrame):
            objs = [objs]
        else:
            objs = list(objs)

        return cls.from_fn(pd.concat, objs, **kwargs)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        inputs: pd.DataFrame | dict["LazyFrame", pd.DataFrame],
        *,
        cache: Cache | None = None,
    ) -> "LazyFrameRun":
        """Create a ``LazyFrameRun`` for this LazyFrame.

        Args:
            inputs: Single DataFrame (for this LazyFrame) or
                   dict mapping LazyFrame objects to DataFrames.
        """
        from .run import LazyFrameRun

        if not isinstance(inputs, dict):
            inputs = {self: inputs}
        return LazyFrameRun(self, inputs, cache=cache or self._default_cache)

    def execute(
        self,
        inputs: pd.DataFrame | dict["LazyFrame", pd.DataFrame],
        *,
        cache: Cache | None = None,
    ) -> pd.DataFrame | Any:
        """Execute the LazyFrame and return the result.

        Args:
            inputs: Single DataFrame (for this LazyFrame) or
                   dict of ``LazyFrame -> DataFrame``.
        """
        lotus.logger.debug(f"LazyFrame.execute: starting with {len(self._nodes)} nodes")
        result = self.run(inputs, cache=cache).execute()
        lotus.logger.debug(f"LazyFrame.execute: completed, result type={type(result).__name__}")
        return result

    # ------------------------------------------------------------------
    # Persistence

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize this LazyFrame pipeline to a file.

        The pipeline structure (all AST nodes) is persisted using pickle.
        Bound DataFrames, callables, and nested LazyFrame references are
        included — the file is *not* portable across different Python
        environments if custom callables are used.

        Args:
            path: Destination file path (e.g. ``"pipeline.pkl"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"nodes": self._nodes, "source": self._source}, f)
        lotus.logger.debug(f"LazyFrame.save: wrote {len(self._nodes)} nodes to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LazyFrame":
        """Load a LazyFrame pipeline from a file saved with :meth:`save`.

        Args:
            path: File path previously written by ``save()``.

        Returns:
            A reconstructed LazyFrame with the same pipeline structure.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        lf = cls(_nodes=data["nodes"], _source=data["source"])
        lotus.logger.debug(f"LazyFrame.load: read {len(lf._nodes)} nodes from {path}")
        return lf

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        optimizers: list["BaseOptimizer"] = [],
        *,
        inplace: bool = False,
        train_data: pd.DataFrame | dict["LazyFrame", pd.DataFrame] | None = None,
        auto_include_default_optimizers: bool = True,
    ) -> "LazyFrame":
        """Apply optimizations to this LazyFrame.

        Args:
            optimizers: List of optimizers to apply.
            inplace: If True, modify this LazyFrame in place.
            train_data: Optional training data for optimizers that require it.
            auto_include_default_optimizer: If True (default), include the following optimizers:
                - PredicatePushdownOptimizer
        Returns:
            The optimized LazyFrame (same object if *inplace*, new otherwise).
        """
        from .optimizer import DEFAULT_OPTIMIZERS

        all_optimizers = (DEFAULT_OPTIMIZERS + optimizers) if auto_include_default_optimizers else optimizers

        if not all_optimizers:
            lotus.logger.warning("LazyFrame.optimize: no optimizers provided, returning original LazyFrame")
            return self if inplace else self.copy()

        lotus.logger.debug(
            f"LazyFrame.optimize: {len(self._nodes)} nodes, " f"{len(all_optimizers)} optimizer(s), inplace={inplace}"
        )

        optimized_nodes = self._nodes[:]
        for optimizer in all_optimizers:
            lotus.logger.debug(f"LazyFrame.optimize: applying {optimizer.__class__.__name__}")
            optimized_nodes = optimizer.optimize(optimized_nodes, train_data=train_data)

        if inplace:
            self._nodes = optimized_nodes
            return self
        else:
            return LazyFrame(_nodes=optimized_nodes, _source=self._source, _default_cache=self._default_cache)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LazyFrame(nodes={len(self._nodes)})"

    def show(self) -> str:
        """Return the LazyFrame structure as a tree-like string."""
        if not self._nodes:
            return "LazyFrame()"

        INDENT = "    "

        def _build(node_idx: int, indent: int = 0) -> list[str]:
            if node_idx < 0 or node_idx >= len(self._nodes):
                return []

            node = self._nodes[node_idx]
            prefix_str = INDENT * indent
            arrow = "-- " if indent > 0 else ""

            lines: list[str] = [f"{prefix_str}{arrow}{node.signature()}"]

            children = node.child_lfs()
            is_join = isinstance(node, (SemJoinNode, SemSimJoinNode))

            if is_join and node_idx > 0:
                lines.append(f"{prefix_str}{INDENT}-- current LazyFrame")
                lines.extend(_build(node_idx - 1, indent + 2))

            for label, child_lf in children:
                lines.append(f"{prefix_str}{INDENT}-- {label}")
                if hasattr(child_lf, "show"):
                    for cl in child_lf.show().split("\n"):
                        lines.append(f"{prefix_str}{INDENT}{INDENT}{cl}")
                elif isinstance(child_lf, SourceNode):
                    lines.append(f"{prefix_str}{INDENT}{INDENT}{child_lf.signature()}")

            if not is_join and node_idx > 0:
                lines.extend(_build(node_idx - 1, indent + 1))

            return lines

        return "\n".join(_build(len(self._nodes) - 1))

    def print_tree(self) -> None:
        """Print the LazyFrame structure as a tree."""
        print(self.show())

    def __len__(self) -> int:
        return len(self._nodes)
