"""Execution context for pipelines with cached intermediates."""

from __future__ import annotations

from typing import Any

import pandas as pd

import lotus

from .cache import compute_cache_key, hash_result
from .nodes import ApplyFnNode, PandasAssignNode, PandasOpNode, SemJoinNode, SemSimJoinNode, SourceNode
from .pipeline import LazyFrame


class LazyFrameRun:
    """LazyFrameRun a pipeline against DataFrames with content-addressable caching.

    Cache keys are computed from (node definition hash + input state hash).
    Sub-pipelines (joins, assigns) share the same cache to avoid re-execution.
    """

    def __init__(
        self,
        pipeline: LazyFrame,
        inputs: pd.DataFrame | dict[LazyFrame, pd.DataFrame],
        *,
        _shared_cache: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._inputs: dict[LazyFrame, pd.DataFrame] = inputs if isinstance(inputs, dict) else {pipeline: inputs}
        # Content-addressable cache: cache_key -> result (shared with sub-pipelines)
        self._content_cache: dict[str, Any] = _shared_cache if _shared_cache is not None else {}
        self._cache_stats: dict[str, int] = {"hits": 0, "misses": 0}

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache hit/miss statistics for this run."""
        return dict(self._cache_stats)

    def execute(self) -> pd.DataFrame | Any:
        """Execute the pipeline and return the final result."""
        lotus.logger.debug("LazyFrameRun.execute: starting execution")
        pipeline = self._pipeline

        nodes = pipeline._nodes
        lotus.logger.debug(f"LazyFrameRun.execute: {len(nodes)} nodes to execute")

        current: Any = None
        current_hash = ""

        for i in range(len(nodes)):
            node = nodes[i]
            lotus.logger.debug(f"LazyFrameRun.execute: step {i+1}/{len(nodes)} - {type(node).__name__}")

            cache_key = compute_cache_key(node, current_hash)
            if cache_key in self._content_cache:
                lotus.logger.debug(f"LazyFrameRun.execute: cache HIT for {type(node).__name__}")
                self._cache_stats["hits"] += 1
                current = self._content_cache[cache_key]
                current_hash = hash_result(current)
                continue

            self._cache_stats["misses"] += 1

            if isinstance(node, SourceNode):
                current = self._execute_source_node(node)

            elif isinstance(node, (SemJoinNode, SemSimJoinNode)):
                current = self._execute_join_node(node, current)

            elif isinstance(node, PandasOpNode):
                current = self._execute_pandas_op_node(node, current)

            elif isinstance(node, PandasAssignNode):
                current = self._execute_assign_node(node, current)

            elif isinstance(node, ApplyFnNode):
                # ApplyFnNode has no "self" DataFrame; it takes only LazyFrame results
                current = self._execute_apply_fn_node(node)

            else:
                # Unary operations (sem_filter, sem_map, etc.)
                if current is None:
                    raise ValueError(
                        "LazyFrame has no source node. Use LazyFrame(key) so the pipeline "
                        "has a source by default, or ensure the first node is a source."
                    )
                current = node(current)

            self._content_cache[cache_key] = current
            current_hash = hash_result(current)
            lotus.logger.debug(f"LazyFrameRun.execute: step {i+1} completed, cached result")

        return current

    def _execute_source_node(self, node: SourceNode) -> pd.DataFrame:
        """Execute a source node."""
        lotus.logger.debug("LazyFrameRun._execute_source_node: loading source")
        # Look up DataFrame by LazyFrame reference
        df = self._inputs.get(node.pipeline_ref) if node.pipeline_ref else None

        # If not found by reference, and there's only one input, use it
        if df is None and len(self._inputs) == 1:
            df = next(iter(self._inputs.values()))
            lotus.logger.debug("LazyFrameRun._execute_source_node: using single input DataFrame")

        # Call the node to execute it (this triggers schema validation)
        if df is not None:
            lotus.logger.debug(f"LazyFrameRun._execute_source_node: executing source node with {len(df)} rows")
            return node(df)
        elif node.df is not None:
            lotus.logger.debug("LazyFrameRun._execute_source_node: executing source node with bound df")
            return node()
        else:
            raise ValueError("No DataFrame provided for source")

    def _resolve_right_df(self, node: SemJoinNode | SemSimJoinNode) -> pd.DataFrame:
        """Resolve the right DataFrame for a join node.

        Supports two modes:
        1. right_pipeline: LazyFrame to execute first
        2. right_df: direct DataFrame reference
        """
        # Mode 1: Direct DataFrame
        if node.right_df is not None:
            lotus.logger.debug(f"LazyFrameRun._resolve_right_df: using direct DataFrame ({len(node.right_df)} rows)")
            return node.right_df

        # Mode 2: LazyFrame reference (share cache with main run)
        if node.right_pipeline is not None:
            lotus.logger.debug("LazyFrameRun._resolve_right_df: executing right pipeline")
            sub_run = LazyFrameRun(node.right_pipeline, self._inputs, _shared_cache=self._content_cache)
            result = sub_run.execute()
            lotus.logger.debug(f"LazyFrameRun._resolve_right_df: right pipeline returned {len(result)} rows")
            return result

        if node.right_source_node is not None:
            lotus.logger.debug(
                f"LazyFrameRun._resolve_right_df: using right source node '{node.right_source_node.pipeline_ref}'"
            )
            return self._execute_source_node(node.right_source_node)

        raise ValueError("Join node has no right DataFrame specified. ")

    def _execute_join_node(
        self,
        node: SemJoinNode | SemSimJoinNode,
        current: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Execute a join node."""
        if current is None:
            raise ValueError("No left DataFrame for join operation.")

        lotus.logger.debug(f"LazyFrameRun._execute_join_node: executing {type(node).__name__}")
        right_df = self._resolve_right_df(node)
        result = node(current, right_df=right_df)
        lotus.logger.debug(f"LazyFrameRun._execute_join_node: join produced {len(result)} rows")
        return result

    def _resolve_pandas_args(self, node: PandasOpNode) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Resolve LazyFrame references in pandas op args and kwargs.

        Returns:
            Tuple of (resolved_args, resolved_kwargs)
        """
        lotus.logger.debug(f"LazyFrameRun._resolve_pandas_args: resolving args for '{node.op_name}'")
        # Start with the stored args/kwargs
        resolved_args = list(node.args)
        resolved_kwargs = dict(node.kwargs) if node.kwargs else {}

        # Resolve pipeline args (positional)
        if node.pipeline_args:
            lotus.logger.debug(f"LazyFrameRun._resolve_pandas_args: resolving {len(node.pipeline_args)} pipeline args")
            for key, pipeline in node.pipeline_args.items():
                # Key format: "_pipeline_arg_N" where N is the arg index
                idx = int(key.split("_")[-1])
                lotus.logger.debug(f"LazyFrameRun._resolve_pandas_args: executing pipeline for arg {idx}")
                sub_run = LazyFrameRun(pipeline, self._inputs, _shared_cache=self._content_cache)
                resolved_args[idx] = sub_run.execute()

        # Resolve pipeline kwargs
        if node.pipeline_kwargs:
            lotus.logger.debug(
                f"LazyFrameRun._resolve_pandas_args: resolving {len(node.pipeline_kwargs)} pipeline kwargs"
            )
            for key, pipeline in node.pipeline_kwargs.items():
                # Key format: "_pipeline_kwarg_NAME" where NAME is the kwarg name
                kwarg_name = key.replace("_pipeline_kwarg_", "")
                lotus.logger.debug(f"LazyFrameRun._resolve_pandas_args: executing pipeline for kwarg '{kwarg_name}'")
                sub_run = LazyFrameRun(pipeline, self._inputs, _shared_cache=self._content_cache)
                resolved_kwargs[kwarg_name] = sub_run.execute()

        return tuple(resolved_args), resolved_kwargs

    def _execute_pandas_op_node(self, node: PandasOpNode, current: pd.DataFrame | None) -> pd.DataFrame | Any:
        """Execute a pandas operation node."""
        if current is None:
            raise ValueError("No DataFrame for pandas operation.")

        lotus.logger.debug(f"LazyFrameRun._execute_pandas_op_node: executing '{node.op_name}'")
        # Check if we need to resolve any pipeline arguments
        if node.pipeline_args or node.pipeline_kwargs:
            resolved_args, resolved_kwargs = self._resolve_pandas_args(node)
            result = node(current, resolved_args=resolved_args, resolved_kwargs=resolved_kwargs)
        else:
            # No pipeline args, use simple execution path
            result = node(current)

        if hasattr(result, "__len__") and not isinstance(result, str):
            lotus.logger.debug(f"LazyFrameRun._execute_pandas_op_node: output {len(result)} rows/items")
        return result

    def _resolve_pipeline_value(self, value_pipeline: LazyFrame, current: pd.DataFrame) -> Any:
        """Resolve a LazyFrame value, optimizing for self-referencing pipelines.

        If the value_pipeline shares a common prefix with the main pipeline,
        we only execute the suffix operations on the current DataFrame instead
        of re-executing the entire pipeline (which would duplicate work).
        """
        # Check if value_pipeline shares nodes with the main pipeline
        main_nodes = self._pipeline._nodes
        value_nodes = value_pipeline._nodes

        # Find the common prefix length
        common_prefix_len = 0
        for i, (main_node, value_node) in enumerate(zip(main_nodes, value_nodes)):
            if main_node is value_node:  # Same object reference
                common_prefix_len = i + 1
            else:
                break

        if common_prefix_len > 0 and common_prefix_len < len(value_nodes):
            # The value_pipeline is derived from the main pipeline
            # Execute only the suffix operations on the current DataFrame
            suffix_nodes = value_nodes[common_prefix_len:]
            lotus.logger.debug(
                f"LazyFrameRun._resolve_pipeline_value: detected self-reference, "
                f"executing {len(suffix_nodes)} suffix nodes instead of full pipeline"
            )

            # Execute suffix nodes on current DataFrame
            result: Any = current
            for suffix_node in suffix_nodes:
                if isinstance(suffix_node, PandasOpNode):
                    result = self._execute_pandas_op_node(suffix_node, result)
                elif isinstance(suffix_node, PandasAssignNode):
                    result = self._execute_assign_node(suffix_node, result)
                else:
                    result = suffix_node(result)
            return result
        else:
            # No common prefix, execute the full pipeline (share cache)
            lotus.logger.debug("LazyFrameRun._resolve_pipeline_value: executing full pipeline")
            sub_run = LazyFrameRun(value_pipeline, self._inputs, _shared_cache=self._content_cache)
            return sub_run.execute()

    def _execute_assign_node(self, node: PandasAssignNode, current: pd.DataFrame | None) -> pd.DataFrame:
        """Execute an assign node, resolving any LazyFrame references."""
        if current is None:
            raise ValueError("No DataFrame for assign operation.")

        lotus.logger.debug("LazyFrameRun._execute_assign_node: executing assignment")

        context: dict[str, Any] = {}

        # Resolve value_pipeline for single column mode
        if node.value_pipeline is not None:
            lotus.logger.debug(f"LazyFrameRun._execute_assign_node: resolving LazyFrame for column '{node.column}'")
            resolved_value = self._resolve_pipeline_value(node.value_pipeline, current)
            context["resolved_value"] = resolved_value
            lotus.logger.debug(
                f"LazyFrameRun._execute_assign_node: resolved LazyFrame produced {type(resolved_value).__name__}"
            )

        # Resolve assignment_pipelines for multi-column mode
        if node.assignment_pipelines:
            resolved_assignments: dict[str, Any] = {}
            for col, pipeline in node.assignment_pipelines.items():
                lotus.logger.debug(f"LazyFrameRun._execute_assign_node: resolving LazyFrame for column '{col}'")
                resolved_assignments[col] = self._resolve_pipeline_value(pipeline, current)
            context["resolved_assignments"] = resolved_assignments
            lotus.logger.debug(
                f"LazyFrameRun._execute_assign_node: resolved {len(resolved_assignments)} pipeline assignments"
            )

        result = node(current, **context)
        lotus.logger.debug(f"LazyFrameRun._execute_assign_node: output {len(result)} rows")
        return result

    def _resolve_pipeline_structure(self, value: Any) -> Any:
        """Recursively resolve LazyFrame references in nested structures.

        Args:
            value: Can be a LazyFrame, list, tuple, dict, or any other value

        Returns:
            Resolved value with LazyFrames executed to DataFrames, preserving structure
        """
        from .pipeline import LazyFrame

        # Check type name to avoid issues, but LazyFrame should be imported
        if isinstance(value, LazyFrame):
            # Execute the pipeline and return the DataFrame
            lotus.logger.debug("LazyFrameRun._resolve_pipeline_structure: executing LazyFrame")
            sub_run = LazyFrameRun(value, self._inputs, _shared_cache=self._content_cache)
            result = sub_run.execute()
            lotus.logger.debug(
                f"LazyFrameRun._resolve_pipeline_structure: LazyFrame returned {len(result) if hasattr(result, '__len__') else 'non-DataFrame'} result"
            )
            return result
        elif isinstance(value, list):
            # Recursively resolve each element, preserving list type
            return [self._resolve_pipeline_structure(item) for item in value]
        elif isinstance(value, tuple):
            # Recursively resolve each element, preserving tuple type
            return tuple(self._resolve_pipeline_structure(item) for item in value)
        elif isinstance(value, dict):
            # Recursively resolve each value, preserving keys
            return {k: self._resolve_pipeline_structure(v) for k, v in value.items()}
        else:
            # Static value, return as-is
            return value

    def _execute_apply_fn_node(self, node: ApplyFnNode) -> Any:
        """Execute an ApplyFnNode by resolving LazyFrame refs and calling the function.

        Args:
            node: The ApplyFnNode to execute

        Returns:
            Result of calling node.fn with resolved arguments
        """
        lotus.logger.debug("LazyFrameRun._execute_apply_fn_node: executing ApplyFnNode")

        # Resolve all args recursively (detects and executes LazyFrames wherever they appear)
        resolved_args = tuple(self._resolve_pipeline_structure(arg) for arg in node.args)

        # Resolve all kwargs recursively
        resolved_kwargs = {}
        if node.kwargs:
            for key, value in node.kwargs.items():
                resolved_kwargs[key] = self._resolve_pipeline_structure(value)

        # Call the node with resolved args/kwargs
        result = node(resolved_fn_args=resolved_args, resolved_fn_kwargs=resolved_kwargs)
        return result

    # ------------------------------------------------------------------
    # DataFrame-like access (execute on demand)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Execute and access result attribute."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        result = self.execute()
        return getattr(result, name)

    def __getitem__(self, key: Any) -> Any:
        """Execute and index into result."""
        return self.execute()[key]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LazyFrameRun({self._pipeline!r})"
