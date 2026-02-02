"""Pipeline builder for LOTUS AST operations."""

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
    from .run import Run


class _LazyMethodProxy:
    """Proxy that captures pandas method calls and returns a new Pipeline.

    Detects Pipeline arguments and stores them separately for resolution during execution.
    """

    def __init__(self, pipeline: "Pipeline", method_name: str) -> None:
        self._pipeline = pipeline
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> "Pipeline":
        # Process positional args - detect Pipeline references
        processed_args: list[Any] = []
        pipeline_args: dict[str, Any] = {}

        for i, arg in enumerate(args):
            if isinstance(arg, Pipeline):
                # Store Pipeline reference separately, use None as placeholder
                pipeline_args[f"_pipeline_arg_{i}"] = arg
                processed_args.append(None)
            else:
                processed_args.append(arg)

        # Process kwargs - detect Pipeline references
        processed_kwargs: dict[str, Any] = {}
        pipeline_kwargs: dict[str, Any] = {}

        for key, value in kwargs.items():
            if isinstance(value, Pipeline):
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


class Pipeline:
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
        """Pipeline name (source key), used to look up input in execute(inputs)."""
        return self._source.key if self._source is not None else "default"

    def _append_node(self, node: BaseNode) -> "Pipeline":
        """Return a new Pipeline with the node appended (immutable)."""
        lotus.logger.debug(f"Pipeline: appending {type(node).__name__}")
        new_nodes = list(self._nodes)
        new_nodes.append(node)
        return Pipeline(_nodes=new_nodes, _source=self._source)

    def copy(self) -> "Pipeline":
        """Return a deep copy of this pipeline."""
        return Pipeline(
            _nodes=deepcopy(self._nodes),
            _source=deepcopy(self._source),
        )

    # ------------------------------------------------------------------
    # Source Management
    # ------------------------------------------------------------------

    def add_source(self, key: str = "default", df: pd.DataFrame | None = None) -> "Pipeline":
        """Set the pipeline source (key = pipeline name, optional bound DataFrame).

        Replaces the single source node. Use this to rename the pipeline or bind a df.
        """
        source_node = SourceNode(key=key, df=df)
        new_nodes = [source_node] + list(self._nodes[1:]) if len(self._nodes) > 1 else [source_node]
        return Pipeline(_nodes=new_nodes, _source=source_node)

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
    ) -> "Pipeline":
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
    ) -> "Pipeline":
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
    ) -> "Pipeline":
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
    ) -> "Pipeline":
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
    ) -> "Pipeline":
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
        right: str | SourceNode | "Pipeline" | pd.DataFrame,
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
    ) -> "Pipeline":
        """Add a semantic join operation.

        Args:
            right: The right side of the join. Can be:
                - str: Key of the right source (must be passed in inputs)
                - SourceNode: Source node to join with
                - Pipeline: Another pipeline to execute first
                - pd.DataFrame: Direct DataFrame to join with
            join_instruction: Natural language join instruction
        """
        # Determine which mode to use based on the type of right
        right_source_node: SourceNode | None = None
        right_pipeline: Pipeline | None = None
        right_df: pd.DataFrame | None = None

        if isinstance(right, str):
            right_source_node = SourceNode(key=right)
        elif isinstance(right, SourceNode):
            right_source_node = right
        elif isinstance(right, Pipeline):
            right_pipeline = right
        elif isinstance(right, pd.DataFrame):
            right_df = right
        else:
            raise TypeError(f"right must be str, Pipeline, or DataFrame, got {type(right)}")

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
        right: str | SourceNode | "Pipeline" | pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        *,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
    ) -> "Pipeline":
        """Add a semantic similarity join operation.

        Args:
            right: The right side of the join. Can be:
                - str: Key of the right source (must be passed in inputs)
                - SourceNode: Source node to join with
                - Pipeline: Another pipeline to execute first
                - pd.DataFrame: Direct DataFrame to join with
            left_on: Column name in left DataFrame for join
            right_on: Column name in right DataFrame for join
            K: Number of top matches to return
        """
        # Determine which mode to use based on the type of right
        right_source_node: SourceNode | None = None
        right_pipeline: Pipeline | None = None
        right_df: pd.DataFrame | None = None

        if isinstance(right, str):
            right_source_node = SourceNode(key=right)
        elif isinstance(right, SourceNode):
            right_source_node = right
        elif isinstance(right, Pipeline):
            right_pipeline = right
        elif isinstance(right, pd.DataFrame):
            right_df = right
        else:
            raise TypeError(f"right must be str, Pipeline, or DataFrame, got {type(right)}")

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
    ) -> "Pipeline":
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

    def sem_index(self, col_name: str, index_dir: str) -> "Pipeline":
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
    ) -> "Pipeline":
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

    def sem_dedup(self, col_name: str, threshold: float) -> "Pipeline":
        """Add a semantic deduplication operation."""
        node = SemDedupNode(col_name=col_name, threshold=threshold)
        return self._append_node(node)

    def sem_partition_by(self, partition_fn: Callable[[pd.DataFrame], list[int]]) -> "Pipeline":
        """Add a semantic partition operation."""
        node = SemPartitionByNode(partition_fn=partition_fn)
        return self._append_node(node)

    # ------------------------------------------------------------------
    # Pandas Operations
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> "Pipeline":
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

    def __getitem__(self, key: Any) -> "Pipeline":
        """Support column selection via pipeline['col'] or pipeline[['col1', 'col2']]."""
        if callable(key):
            return self.filter(key)

        node = PandasOpNode(op_name="__getitem__", args=(key,))
        return self._append_node(node)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support column assignment via pipeline['col'] = value.

        Note: This mutates the pipeline in place (unlike most Pipeline operations).
        For immutable behavior, use .assign(col=value) instead.

        Args:
            key: Column name to assign
            value: Value to assign. Can be:
                - Scalar values
                - Callables that take the DataFrame and return a Series
                - Pipeline objects (resolved lazily during execution)
        """
        from .nodes import PandasAssignNode

        if isinstance(value, Pipeline):
            # Store Pipeline reference for lazy resolution
            lotus.logger.debug(f"Pipeline.__setitem__: storing Pipeline reference for column '{key}'")
            node = PandasAssignNode(column=key, value_pipeline=value)
        else:
            node = PandasAssignNode(column=key, value=value)
        self._nodes.append(node)

    def assign(self, **kwargs: Any) -> "Pipeline":
        """Add column assignments (immutable, returns new Pipeline).

        Similar to pandas DataFrame.assign().

        Args:
            **kwargs: Column name -> value mappings. Values can be:
                - Scalar values
                - Callables that take the DataFrame and return a Series
                - Pipeline objects (resolved lazily during execution)

        Returns:
            New Pipeline with the assign operation added.
        """
        from .nodes import PandasAssignNode

        # Separate Pipeline references from regular values
        regular_assignments: dict[str, Any] = {}
        pipeline_assignments: dict[str, Any] = {}

        for col, val in kwargs.items():
            if isinstance(val, Pipeline):
                lotus.logger.debug(f"Pipeline.assign: storing Pipeline reference for column '{col}'")
                pipeline_assignments[col] = val
            else:
                regular_assignments[col] = val

        node = PandasAssignNode(
            assignments=regular_assignments if regular_assignments else None,
            assignment_pipelines=pipeline_assignments if pipeline_assignments else None,
        )
        return self._append_node(node)

    # ------------------------------------------------------------------
    # Pipeline Functions (concat / from_fn)
    # ------------------------------------------------------------------

    @classmethod
    def from_fn(
        cls,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> "Pipeline":
        """Create a Pipeline from a function that takes Pipeline results.

        Stores args/kwargs as-is. During execution, _resolve_pipeline_structure will
        recursively detect and execute any Pipeline instances found in args/kwargs.
        Supports nested structures (lists, tuples, dicts).

        Args:
            fn: Callable to execute (e.g., pd.concat)
            *args: Positional arguments (may contain Pipelines or nested structures)
            **kwargs: Keyword arguments (may contain Pipelines or nested structures)

        Returns:
            Pipeline with a single ApplyFnNode and no source

        Examples:
            >>> p1 = Pipeline("data1")
            >>> p2 = Pipeline("data2")
            >>> combined = Pipeline.from_fn(pd.concat, [p1, p2])
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
        objs: list["Pipeline"] | "Pipeline",
        **kwargs: Any,
    ) -> "Pipeline":
        """Concatenate multiple Pipeline results using pd.concat.

        Args:
            objs: Single Pipeline or list/iterable of Pipelines
            **kwargs: Additional arguments passed to pd.concat (e.g., axis, ignore_index)

        Returns:
            Pipeline that concatenates the results of all input pipelines

        Examples:
            >>> p1 = Pipeline("data1")
            >>> p2 = Pipeline("data2")
            >>> combined = Pipeline.concat([p1, p2])
            >>> result = combined.execute({"data1": df1, "data2": df2})
        """
        # Normalize objs to a list if it's a single Pipeline
        if isinstance(objs, Pipeline):
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
    ) -> "Run":
        """Create a Run object for this pipeline.

        Args:
            inputs: Single DataFrame (uses this pipeline's key) or
                   dict mapping pipeline names (source keys) to DataFrames
        """
        from .run import Run

        if not isinstance(inputs, dict):
            inputs = {self.key: inputs}
        return Run(self, inputs)

    def execute(
        self,
        inputs: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> pd.DataFrame | Any:
        """Execute the pipeline and return the result.

        Args:
            inputs: Single DataFrame (for this pipeline's key) or
                   dict of pipeline name (key) -> DataFrame
        """
        lotus.logger.debug(f"Pipeline.execute: starting with {len(self._nodes)} nodes")
        result = self.run(inputs).execute()
        if hasattr(result, "__len__") and not isinstance(result, str):
            lotus.logger.debug(f"Pipeline.execute: completed with {len(result)} rows/items")
        else:
            lotus.logger.debug(f"Pipeline.execute: completed with result type={type(result).__name__}")
        return result

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        def node_repr(node, indent=0):
            INDENT = "    "
            prefix = INDENT * indent

            # Child pipeline detection (e.g. for SemJoinNode, SemSimJoinNode)
            def show_node(node, indent):
                if isinstance(node, SourceNode):
                    return f"{prefix}.add_source({node.key!r})"
                elif isinstance(node, SemFilterNode):
                    return f'{prefix}.sem_filter("{node.user_instruction}")'
                elif isinstance(node, SemMapNode):
                    return f'{prefix}.sem_map("{node.user_instruction}")'
                elif isinstance(node, SemExtractNode):
                    return f"{prefix}.sem_extract({node.input_cols!r}, {node.output_cols!r})"
                elif isinstance(node, SemAggNode):
                    return f'{prefix}.sem_agg("{node.user_instruction}")'
                elif isinstance(node, SemTopKNode):
                    return f'{prefix}.sem_topk("{node.user_instruction}", {node.K})'
                elif isinstance(node, SemJoinNode):
                    # Tree depiction for right_pipeline, right_source_node, or direct DataFrame
                    if node.right_source_node is not None:
                        right_repr = f'"{node.right_source_node.key}"'
                        children = []
                    elif node.right_pipeline:
                        right_repr = "<Pipeline>"
                        children = getattr(node.right_pipeline, "_nodes", [])
                    else:
                        right_repr = "<DataFrame>"
                        children = []
                    s = f'{prefix}.sem_join({right_repr}, "{node.join_instruction}")'
                    if children:
                        # Show right pipeline subtree indented
                        s += ":\n"
                        for child in children:
                            s += node_repr(child, indent + 1) + "\n"
                        s = s.rstrip()
                    return s
                elif isinstance(node, SemSimJoinNode):
                    if node.right_source_node is not None:
                        right_repr = f'"{node.right_source_node.key}"'
                        children = []
                    elif node.right_pipeline:
                        right_repr = "<Pipeline>"
                        children = getattr(node.right_pipeline, "_nodes", [])
                    else:
                        right_repr = "<DataFrame>"
                        children = []
                    s = f"{prefix}.sem_sim_join({right_repr}, ...)"
                    if children:
                        s += ":\n"
                        for child in children:
                            s += node_repr(child, indent + 1) + "\n"
                        s = s.rstrip()
                    return s
                elif isinstance(node, SemSearchNode):
                    return f'{prefix}.sem_search("{node.col_name}", "{node.query}")'
                elif isinstance(node, SemIndexNode):
                    return f'{prefix}.sem_index("{node.col_name}", "{node.index_dir}")'
                elif isinstance(node, SemClusterByNode):
                    return f'{prefix}.sem_cluster_by("{node.col_name}", {node.ncentroids})'
                elif isinstance(node, SemDedupNode):
                    return f'{prefix}.sem_dedup("{node.col_name}", {node.threshold})'
                elif isinstance(node, SemPartitionByNode):
                    return f"{prefix}.sem_partition_by(...)"
                elif isinstance(node, PandasFilterNode):
                    return f"{prefix}.filter(...)"
                elif isinstance(node, PandasAssignNode):
                    if getattr(node, "column", None):
                        return f"{prefix}[{node.column!r}] = ..."
                    elif getattr(node, "assignments", None):
                        cols = list(node.assignments.keys())
                        return f"{prefix}.assign({', '.join(cols)}=...)"
                    else:
                        return f"{prefix}.assign(...)"
                elif isinstance(node, PandasOpNode):
                    if getattr(node, "is_attr", False):
                        return f"{prefix}.{node.op_name}"
                    elif getattr(node, "op_name", None) == "__getitem__":
                        return f"{prefix}[{node.args[0]!r}]"
                    else:
                        args_str = ", ".join(repr(a) for a in node.args)
                        if node.kwargs:
                            kw_str = ", ".join(f"{k}={v!r}" for k, v in node.kwargs.items())
                            if args_str:
                                args_str = f"{args_str}, {kw_str}"
                            else:
                                args_str = kw_str
                        return f"{prefix}.{node.op_name}({args_str})"
                else:
                    return f"{prefix}.{type(node).__name__}(...)"

            return show_node(node, indent)

        if not self._nodes:
            return "Pipeline()"

        lines = [f"Pipeline({self.key!r})"]
        for node in self._nodes[1:]:  # skip source (shown in Pipeline(key))
            node_str = node_repr(node, indent=0)
            lines.append(node_str)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._nodes)
