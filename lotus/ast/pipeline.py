"""LazyFrame builder for LOTUS AST operations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

import lotus
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
    PandasAssignNode,
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
    """Proxy that captures pandas method calls and returns a new LazyFrame.

    Detects LazyFrame arguments and stores them separately for resolution during execution.
    """

    def __init__(self, pipeline: "LazyFrame", method_name: str) -> None:
        self._pipeline = pipeline
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> "LazyFrame":
        # Process positional args - detect LazyFrame references
        processed_args: list[Any] = []
        pipeline_args: dict[str, Any] = {}

        for i, arg in enumerate(args):
            if isinstance(arg, LazyFrame):
                # Store LazyFrame reference separately, use None as placeholder
                pipeline_args[f"_pipeline_arg_{i}"] = arg
                processed_args.append(None)
            else:
                processed_args.append(arg)

        # Process kwargs - detect LazyFrame references
        processed_kwargs: dict[str, Any] = {}
        pipeline_kwargs: dict[str, Any] = {}

        for key, value in kwargs.items():
            if isinstance(value, LazyFrame):
                pipeline_kwargs[f"_pipeline_kwarg_{key}"] = value
                processed_kwargs[key] = None
            else:
                processed_kwargs[key] = value

        node = PandasOpNode(
            op_name=self._method_name,
            args=tuple(processed_args),
            kwargs=processed_kwargs if processed_kwargs else None,
            is_attr=False,
            pipeline_args=pipeline_args if pipeline_args else None,
            pipeline_kwargs=pipeline_kwargs if pipeline_kwargs else None,
        )
        return self._pipeline._append_node(node)


class LazyFrame:
    """Lazy DataFrame.
    A LazyFrame is a pipeline of operations that can be executed to produce a DataFrame. It is a wrapper around a pandas DataFrame that allows for lazy execution of operations.

    Usage:
        lazy_df = LazyFrame("queries")  # lazy_df name = "queries"
        lazy_df = lazy_df.sem_filter("...").sem_map("...")
        result = lazy_df.execute(queries_df)  # or execute({"queries": queries_df})

        # With optional bound DataFrame at construction:
        lazy_df = LazyFrame("queries", df=queries_df)

    For joins, the right side is another LazyFrame (with its own source key):
        left_lazy_df = LazyFrame("left").sem_join(right_lazy_df, ...)
        result = left_lazy_df.execute({"left": left_df, "right": right_df})
    """

    def __init__(
        self,
        key: str = "default",
        df: pd.DataFrame | None = None,
        *,
        _nodes: list[BaseNode] | None = None,
        _source: SourceNode | None = None,
    ) -> None:
        if _nodes is not None:
            self._nodes = list(_nodes)
            self._source: SourceNode | None = (
                _source
                if _source is not None
                else (self._nodes[0] if self._nodes and isinstance(self._nodes[0], SourceNode) else None)
            )
        else:
            source_node = SourceNode(key=key, df=df)
            self._nodes = [source_node]
            self._source = source_node

    @property
    def key(self) -> str:
        """LazyFrame name (source key), used to look up input in execute(inputs)."""
        return self._source.key if self._source is not None else "default"

    def _append_node(self, node: BaseNode) -> "LazyFrame":
        """Return a new LazyFrame with the node appended (immutable)."""
        lotus.logger.debug(f"LazyFrame: appending {type(node).__name__}")
        new_nodes = list(self._nodes)
        new_nodes.append(node)
        return LazyFrame(_nodes=new_nodes, _source=self._source)

    def copy(self) -> "LazyFrame":
        """Return a deep copy of this pipeline."""
        return LazyFrame(
            _nodes=deepcopy(self._nodes),
            _source=deepcopy(self._source),
        )

    # ------------------------------------------------------------------
    # Source Management
    # ------------------------------------------------------------------

    def add_source(self, key: str = "default", df: pd.DataFrame | None = None) -> "LazyFrame":
        """Set the pipeline source (key = pipeline name, optional bound DataFrame).

        Replaces the single source node. Use this to rename the pipeline or bind a df.
        """
        source_node = SourceNode(key=key, df=df)
        new_nodes = [source_node] + list(self._nodes[1:]) if len(self._nodes) > 1 else [source_node]
        return LazyFrame(_nodes=new_nodes, _source=source_node)

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
    ) -> "LazyFrame":
        """Add a semantic filter operation."""
        node = SemFilterNode(
            user_instruction=user_instruction,
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
        )
        return self._append_node(node)

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
        return self._append_node(node)

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
        return self._append_node(node)

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
        )
        return self._append_node(node)

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
        return self._append_node(node)

    def sem_join(
        self,
        right: str | SourceNode | "LazyFrame" | pd.DataFrame,
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
    ) -> "LazyFrame":
        """Add a semantic join operation.

        Args:
            right: The right side of the join. Can be:
                - str: Key of the right source (must be passed in inputs)
                - SourceNode: Source node to join with
                - LazyFrame: Another pipeline to execute first
                - pd.DataFrame: Direct DataFrame to join with
            join_instruction: Natural language join instruction
        """
        # Determine which mode to use based on the type of right
        right_source_node: SourceNode | None = None
        right_pipeline: LazyFrame | None = None
        right_df: pd.DataFrame | None = None

        if isinstance(right, str):
            right_source_node = SourceNode(key=right)
        elif isinstance(right, SourceNode):
            right_source_node = right
        elif isinstance(right, LazyFrame):
            right_pipeline = right
        elif isinstance(right, pd.DataFrame):
            right_df = right
        else:
            raise TypeError(f"right must be str, LazyFrame, or DataFrame, got {type(right)}")

        node = SemJoinNode(
            right_source_node=right_source_node,
            right_pipeline=right_pipeline,
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
        return self._append_node(node)

    def sem_sim_join(
        self,
        right: str | SourceNode | "LazyFrame" | pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        *,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
    ) -> "LazyFrame":
        """Add a semantic similarity join operation.

        Args:
            right: The right side of the join. Can be:
                - str: Key of the right source (must be passed in inputs)
                - SourceNode: Source node to join with
                - LazyFrame: Another pipeline to execute first
                - pd.DataFrame: Direct DataFrame to join with
            left_on: Column name in left DataFrame for join
            right_on: Column name in right DataFrame for join
            K: Number of top matches to return
        """
        # Determine which mode to use based on the type of right
        right_source_node: SourceNode | None = None
        right_pipeline: LazyFrame | None = None
        right_df: pd.DataFrame | None = None

        if isinstance(right, str):
            right_source_node = SourceNode(key=right)
        elif isinstance(right, SourceNode):
            right_source_node = right
        elif isinstance(right, LazyFrame):
            right_pipeline = right
        elif isinstance(right, pd.DataFrame):
            right_df = right
        else:
            raise TypeError(f"right must be str, LazyFrame, or DataFrame, got {type(right)}")

        node = SemSimJoinNode(
            right_source_node=right_source_node,
            right_pipeline=right_pipeline,
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
        return self._append_node(node)

    def sem_index(self, col_name: str, index_dir: str) -> "LazyFrame":
        """Add a semantic index operation."""
        node = SemIndexNode(col_name=col_name, index_dir=index_dir)
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
    # Pandas Operations
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> "LazyFrame":
        """Add a pandas boolean filter operation."""
        node = PandasFilterNode(predicate=predicate)
        return self._append_node(node)

    def __getattr__(self, name: str) -> Any:
        """Intercept pandas DataFrame method/attribute access."""
        # Avoid recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if it's a DataFrame method/attribute
        if hasattr(pd.DataFrame, name):
            df_attr = getattr(pd.DataFrame, name)
            if callable(df_attr):
                return _LazyMethodProxy(self, name)
            else:
                # Property access - record as node
                node = PandasOpNode(op_name=name, is_attr=True)
                return self._append_node(node)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Any) -> "LazyFrame":
        """Support column selection via pipeline['col'] or pipeline[['col1', 'col2']]."""
        if callable(key):
            return self.filter(key)

        node = PandasOpNode(op_name="__getitem__", args=(key,))
        return self._append_node(node)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support column assignment via pipeline['col'] = value.

        Note: This mutates the pipeline in place (unlike most LazyFrame operations).
        For immutable behavior, use .assign(col=value) instead.

        Args:
            key: Column name to assign
            value: Value to assign. Can be:
                - Scalar values
                - Callables that take the DataFrame and return a Series
                - LazyFrame objects (resolved lazily during execution)
        """
        from .nodes import PandasAssignNode

        if isinstance(value, LazyFrame):
            # Store LazyFrame reference for lazy resolution
            lotus.logger.debug(f"LazyFrame.__setitem__: storing LazyFrame reference for column '{key}'")
            node = PandasAssignNode(column=key, value_pipeline=value)
        else:
            node = PandasAssignNode(column=key, value=value)
        self._nodes.append(node)

    def assign(self, **kwargs: Any) -> "LazyFrame":
        """Add column assignments (immutable, returns new LazyFrame).

        Similar to pandas DataFrame.assign().

        Args:
            **kwargs: Column name -> value mappings. Values can be:
                - Scalar values
                - Callables that take the DataFrame and return a Series
                - LazyFrame objects (resolved lazily during execution)

        Returns:
            New LazyFrame with the assign operation added.
        """
        from .nodes import PandasAssignNode

        # Separate LazyFrame references from regular values
        regular_assignments: dict[str, Any] = {}
        pipeline_assignments: dict[str, Any] = {}

        for col, val in kwargs.items():
            if isinstance(val, LazyFrame):
                lotus.logger.debug(f"LazyFrame.assign: storing LazyFrame reference for column '{col}'")
                pipeline_assignments[col] = val
            else:
                regular_assignments[col] = val

        node = PandasAssignNode(
            assignments=regular_assignments if regular_assignments else None,
            assignment_pipelines=pipeline_assignments if pipeline_assignments else None,
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
        """Create a LazyFrame from a function that takes LazyFrame results.

        Stores args/kwargs as-is. During execution, _resolve_pipeline_structure will
        recursively detect and execute any LazyFrame instances found in args/kwargs.
        Supports nested structures (lists, tuples, dicts).

        Args:
            fn: Callable to execute (e.g., pd.concat)
            *args: Positional arguments (may contain LazyFrames or nested structures)
            **kwargs: Keyword arguments (may contain LazyFrames or nested structures)

        Returns:
            LazyFrame with a single ApplyFnNode and no source

        Examples:
            >>> p1 = LazyFrame("data1")
            >>> p2 = LazyFrame("data2")
            >>> combined = LazyFrame.from_fn(pd.concat, [p1, p2])
            >>> result = combined.execute({"data1": df1, "data2": df2})
        """
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
        """Concatenate multiple LazyFrame results using pd.concat.

        Args:
            objs: Single LazyFrame or list/iterable of LazyFrames
            **kwargs: Additional arguments passed to pd.concat (e.g., axis, ignore_index)

        Returns:
            LazyFrame that concatenates the results of all input pipelines

        Examples:
            >>> p1 = LazyFrame("data1")
            >>> p2 = LazyFrame("data2")
            >>> combined = LazyFrame.concat([p1, p2])
            >>> result = combined.execute({"data1": df1, "data2": df2})
        """
        # Normalize objs to a list if it's a single LazyFrame
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
        inputs: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> "LazyFrameRun":
        """Create a LazyFrameRun object for this pipeline.

        Args:
            inputs: Single DataFrame (uses this pipeline's key) or
                   dict mapping pipeline names (source keys) to DataFrames
        """
        from .run import LazyFrameRun

        if not isinstance(inputs, dict):
            inputs = {self.key: inputs}
        return LazyFrameRun(self, inputs)

    def execute(
        self,
        inputs: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> pd.DataFrame | Any:
        """Execute the pipeline and return the result.

        Args:
            inputs: Single DataFrame (for this pipeline's key) or
                   dict of pipeline name (key) -> DataFrame
        """
        lotus.logger.debug(f"LazyFrame.execute: starting with {len(self._nodes)} nodes")
        result = self.run(inputs).execute()
        if hasattr(result, "__len__") and not isinstance(result, str):
            lotus.logger.debug(f"LazyFrame.execute: completed with {len(result)} rows/items")
        else:
            lotus.logger.debug(f"LazyFrame.execute: completed with result type={type(result).__name__}")
        return result

    def optimize(
        self,
        optimizers: list["BaseOptimizer"],
        *,
        inplace: bool = False,
    ) -> "LazyFrame":
        """Apply optimizations to this pipeline.

        Args:
            optimizers: List of optimizers to apply.
            inplace: If True, modify the pipeline in place. If False, return a new pipeline.

        Returns:
            The optimized pipeline (same object if inplace=True, new object otherwise)

        Examples:
            >>> pipeline = LazyFrame("data").sem_filter("test").filter(lambda d: d["a"] > 1)
            >>> from lotus.ast.optimizer import PredicatePushdownOptimizer
            >>> optimized = pipeline.optimize([PredicatePushdownOptimizer()])
            >>> # Or in-place:
            >>> pipeline.optimize([PredicatePushdownOptimizer()], inplace=True)  # Modifies pipeline directly
        """
        lotus.logger.debug(
            f"LazyFrame.optimize: optimizing pipeline with {len(self._nodes)} nodes, "
            f"{len(optimizers)} optimizer(s), inplace={inplace}"
        )

        if inplace:
            self._nodes = self._optimize_nodes(self._nodes, optimizers)
            lotus.logger.debug("LazyFrame.optimize: in-place optimization complete")
            return self
        else:
            optimized_nodes = self._optimize_nodes(list(self._nodes), optimizers)
            lotus.logger.debug("LazyFrame.optimize: created new optimized pipeline")
            return LazyFrame(_nodes=optimized_nodes, _source=self._source)

    def _optimize_nodes(self, nodes: list[BaseNode], optimizers: list["BaseOptimizer"]) -> list[BaseNode]:
        """Apply all optimizations to a node list.

        Args:
            nodes: List of nodes to optimize
            optimizers: List of optimizers to apply

        Returns:
            Optimized list of nodes
        """
        for optimizer in optimizers:
            lotus.logger.debug(f"LazyFrame._optimize_nodes: applying {optimizer.get_name()} optimizer")
            nodes = optimizer.optimize_nodes(nodes)
        return nodes

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a simple representation of the pipeline."""
        return f"LazyFrame({self.key!r})"

    def show(self) -> str:
        """Display the full pipeline structure as a tree.

        Nodes are shown in reverse order (last node first, then dependencies).
        Format:
            node3(args, kwargs)
            -- node2(args, kwargs)
                -- node1(args, kwargs)

        For sem_join:
            sem_join(other kwargs)
            -- current pipeline
            ...
            -- right df

        For source node: Source(key)
        For from_fn: fn_name(args, kwargs)
        """
        if not self._nodes:
            return "LazyFrame()"

        def format_node_signature(node: BaseNode) -> str:
            """Format the node signature (name and args/kwargs)."""
            if isinstance(node, SourceNode):
                return f"Source({node.key!r})"

            elif isinstance(node, SemFilterNode):
                instruction = (
                    node.user_instruction[:50] + "..." if len(node.user_instruction) > 50 else node.user_instruction
                )
                return f"sem_filter({instruction!r})"

            elif isinstance(node, SemMapNode):
                instruction = (
                    node.user_instruction[:50] + "..." if len(node.user_instruction) > 50 else node.user_instruction
                )
                return f"sem_map({instruction!r})"

            elif isinstance(node, SemExtractNode):
                return f"sem_extract({node.input_cols!r}, {node.output_cols!r})"

            elif isinstance(node, SemAggNode):
                instruction = (
                    node.user_instruction[:50] + "..." if len(node.user_instruction) > 50 else node.user_instruction
                )
                return f"sem_agg({instruction!r})"

            elif isinstance(node, SemTopKNode):
                instruction = (
                    node.user_instruction[:50] + "..." if len(node.user_instruction) > 50 else node.user_instruction
                )
                return f"sem_topk({instruction!r}, {node.K})"

            elif isinstance(node, SemJoinNode):
                # Build kwargs string (excluding right_pipeline, right_source_node, right_df)
                join_kwargs: dict[str, Any] = {}
                if hasattr(node, "join_instruction"):
                    join_kwargs["join_instruction"] = node.join_instruction
                if hasattr(node, "how"):
                    join_kwargs["how"] = node.how
                if hasattr(node, "suffix"):
                    join_kwargs["suffix"] = node.suffix
                # Add other kwargs if they exist
                for attr in ["return_explanations", "default", "safe_mode"]:
                    if hasattr(node, attr):
                        join_kwargs[attr] = getattr(node, attr)

                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in join_kwargs.items())
                return f"sem_join({kwargs_str})"

            elif isinstance(node, SemSimJoinNode):
                return f"sem_sim_join(left_on={node.left_on!r}, right_on={node.right_on!r}, K={node.K})"

            elif isinstance(node, SemSearchNode):
                return f"sem_search({node.col_name!r}, {node.query!r})"

            elif isinstance(node, SemIndexNode):
                return f"sem_index({node.col_name!r}, {node.index_dir!r})"

            elif isinstance(node, SemClusterByNode):
                return f"sem_cluster_by({node.col_name!r}, {node.ncentroids})"

            elif isinstance(node, SemDedupNode):
                return f"sem_dedup({node.col_name!r}, {node.threshold})"

            elif isinstance(node, SemPartitionByNode):
                return "sem_partition_by(...)"

            elif isinstance(node, PandasFilterNode):
                return "filter(...)"

            elif isinstance(node, PandasAssignNode):
                if hasattr(node, "column") and node.column:
                    return f"[{node.column!r}] = ..."
                elif hasattr(node, "assignments") and node.assignments:
                    cols = list(node.assignments.keys())
                    return f"assign({', '.join(cols)}=...)"
                else:
                    return "assign(...)"

            elif isinstance(node, PandasOpNode):
                if getattr(node, "is_attr", False):
                    return node.op_name
                elif getattr(node, "op_name", None) == "__getitem__":
                    return f"[{node.args[0]!r}]"
                else:
                    args_str = ", ".join(repr(a) for a in node.args) if node.args else ""
                    kwargs_str = (
                        ", ".join(f"{k}={repr(v)}" for k, v in (node.kwargs or {}).items()) if node.kwargs else ""
                    )
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    return f"{node.op_name}({all_args})"

            elif isinstance(node, ApplyFnNode):
                # Get function name
                fn_name = getattr(node.fn, "__name__", repr(node.fn))

                # Format args and kwargs using __repr__
                args_repr = [repr(arg) for arg in node.args] if node.args else []
                kwargs_repr = [f"{k}={repr(v)}" for k, v in (node.kwargs or {}).items()]
                args_str = ", ".join(args_repr + kwargs_repr)

                return f"{fn_name}({args_str})"

            else:
                return f"{type(node).__name__}(...)"

        def build_tree_lines(node_idx: int, indent: int = 0) -> list[str]:
            """Recursively build tree lines for nodes starting from node_idx."""
            if node_idx < 0 or node_idx >= len(self._nodes):
                return []

            node = self._nodes[node_idx]
            INDENT = "    "
            indent_prefix = INDENT * indent
            prefix = "-- " if indent > 0 else ""

            lines: list[str] = []

            # Special handling for SemJoinNode
            if isinstance(node, SemJoinNode):
                sig = format_node_signature(node)
                lines.append(f"{indent_prefix}{prefix}{sig}")

                # Add current pipeline (all nodes before this join)
                if node_idx > 0:
                    lines.append(f"{indent_prefix}{INDENT}-- current pipeline")
                    # Show the entire pipeline up to this point as a subtree
                    prev_lines = build_tree_lines(node_idx - 1, indent + 2)
                    lines.extend(prev_lines)

                # Add right side
                if node.right_pipeline:
                    lines.append(f"{indent_prefix}{INDENT}-- right pipeline")
                    right_lines = node.right_pipeline.show().split("\n")
                    for right_line in right_lines:
                        lines.append(f"{indent_prefix}{INDENT}{INDENT}{right_line}")
                elif node.right_source_node:
                    lines.append(f"{indent_prefix}{INDENT}-- right source")
                    lines.append(f"{indent_prefix}{INDENT}{INDENT}Source({node.right_source_node.key!r})")
                elif node.right_df is not None:
                    lines.append(f"{indent_prefix}{INDENT}-- right df")

            else:
                # Regular node - just show its signature
                sig = format_node_signature(node)
                lines.append(f"{indent_prefix}{prefix}{sig}")

                # For ApplyFnNode, check if args/kwargs contain LazyFrames
                if isinstance(node, ApplyFnNode):
                    # Check args for LazyFrames
                    for arg in node.args:
                        if isinstance(arg, LazyFrame):
                            lines.append(f"{indent_prefix}{INDENT}-- arg pipeline")
                            arg_lines = arg.show().split("\n")
                            for arg_line in arg_lines:
                                lines.append(f"{indent_prefix}{INDENT}{INDENT}{arg_line}")
                        elif isinstance(arg, (list, tuple)):
                            for item in arg:
                                if isinstance(item, LazyFrame):
                                    lines.append(f"{indent_prefix}{INDENT}-- arg pipeline")
                                    item_lines = item.show().split("\n")
                                    for item_line in item_lines:
                                        lines.append(f"{indent_prefix}{INDENT}{INDENT}{item_line}")

                    # Check kwargs for LazyFrames
                    if node.kwargs:
                        for val in node.kwargs.values():
                            if isinstance(val, LazyFrame):
                                lines.append(f"{indent_prefix}{INDENT}-- kwarg pipeline")
                                val_lines = val.show().split("\n")
                                for val_line in val_lines:
                                    lines.append(f"{indent_prefix}{INDENT}{INDENT}{val_line}")

                # For regular nodes, show the previous node as a child
                if node_idx > 0:
                    prev_lines = build_tree_lines(node_idx - 1, indent + 1)
                    lines.extend(prev_lines)

            return lines

        # Start from the last node and build backwards
        if not self._nodes:
            return "LazyFrame()"

        all_lines = build_tree_lines(len(self._nodes) - 1, indent=0)
        return "\n".join(all_lines)

    def __len__(self) -> int:
        return len(self._nodes)
