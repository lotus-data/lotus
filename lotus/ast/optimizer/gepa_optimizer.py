"""GEPA-based optimizer for LOTUS AST pipelines.

Uses GEPA's optimize_anything to evolve natural language instructions
(langex) in semantic operator nodes via LLM-guided evolutionary search.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..nodes import (
    ApplyFnNode,
    BaseNode,
    PandasOpNode,
    SemAggNode,
    SemFilterNode,
    SemJoinNode,
    SemMapNode,
    SemSearchNode,
    SemSimJoinNode,
    SemTopKNode,
    SourceNode,
)
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..pipeline import LazyFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default optimizable parameters per node type
# ---------------------------------------------------------------------------

# Maps node types to the set of field names optimized by default.
# When node.optimizable_params is None (unset), these defaults apply.
# Set optimizable_params to frozenset() to explicitly exclude a node.
DEFAULT_OPTIMIZABLE_PARAMS: dict[type, frozenset[str]] = {
    SemFilterNode: frozenset({"user_instruction"}),
    SemMapNode: frozenset({"user_instruction"}),
    SemAggNode: frozenset({"user_instruction"}),
    SemTopKNode: frozenset({"user_instruction"}),
    SemJoinNode: frozenset({"join_instruction"}),
    SemSearchNode: frozenset({"query"}),
}

# Type alias for user evaluation function
UserEvalFn = Callable[..., "float | tuple[float, dict[str, Any]]"]


# ---------------------------------------------------------------------------
# Pipeline path types for nested LazyFrame addressing
# ---------------------------------------------------------------------------

# A path entry identifies a nested pipeline location:
#   (node_idx, field_name) for direct fields like right_pipeline
#   (node_idx, dict_field, dict_key) for dict entries like pipeline_args
#   (node_idx, "args"/"kwargs", *nested_path) for ApplyFn nested structures
PipelinePathEntry = tuple[Any, ...]
PipelinePath = tuple[PipelinePathEntry, ...]


# ---------------------------------------------------------------------------
# Optimization target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OptTarget:
    """An optimizable parameter in the pipeline.

    Encapsulates all information about a single parameter target: which node
    it belongs to, the parameter name, current value, and the path to the
    sub-pipeline (for nested LazyFrames in joins/PandasOps).
    """

    node_idx: int
    param_name: str
    value: str  # current value as string (JSON-encoded for non-str fields)
    step_idx: int  # global sequential counter across all targets
    is_json_encoded: bool = False  # True if value was json.dumps'd (non-str field)
    pipeline_path: PipelinePath = ()

    @property
    def candidate_key(self) -> str:
        """GEPA candidate dict key for this target."""
        return f"step{self.step_idx}_{self.param_name}"

    @classmethod
    def parse_key(cls, key: str) -> tuple[int, str]:
        """Parse a candidate key back into (step_idx, param_name)."""
        prefix, _, param_name = key.partition("_")
        return int(prefix.removeprefix("step")), param_name


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_optimizable_params(node: BaseNode) -> frozenset[str]:
    """Return the set of params to optimize for a node.

    - If ``node.optimizable_params`` is explicitly set, use it (empty = exclude).
    - Otherwise fall back to ``DEFAULT_OPTIMIZABLE_PARAMS`` for the node type.
    """
    if node.optimizable_params is not None:
        return node.optimizable_params
    return DEFAULT_OPTIMIZABLE_PARAMS.get(type(node), frozenset())


# ---------------------------------------------------------------------------
# GEPAOptimizer
# ---------------------------------------------------------------------------


class GEPAOptimizer(BaseOptimizer):
    """GEPA-based prompt/instruction optimizer for LOTUS pipelines.

    Automatically optimizes natural language instructions in semantic operator
    nodes using LLM-guided evolutionary search (GEPA).

    By default, ``user_instruction`` on sem_filter/sem_map/sem_agg/sem_topk,
    ``join_instruction`` on sem_join, and ``query`` on sem_search are optimized.
    Use ``mark_optimizable`` on the ``LazyFrame`` to customize which parameters
    to optimize or to exclude specific nodes entirely.

    Args:
        eval_fn: Scoring function called once per (candidate, example) pair.
            Signature: ``(output_df, example) -> float`` or
            ``(output_df, example) -> (float, side_info_dict)``.
            Higher scores are better. Return a ``side_info`` dict alongside the
            score to give the GEPA reflection LLM diagnostic context (e.g.
            expected vs. actual output, precision/recall breakdown).
        valset: Optional held-out validation set (list of examples) for GEPA
            generalization mode. When provided, GEPA selects the best candidate
            based on valset performance rather than training performance.
        gepa_config: ``GEPAConfig`` object controlling max LLM calls, model,
            temperature, etc. Defaults to GEPA's built-in defaults when ``None``.
        objective: Natural language goal string passed to the reflection LLM.
            Auto-generated from the pipeline structure when ``None``.
        background: Domain context / constraints string for the reflection LLM.
            Auto-generated with LOTUS operator reference when ``None``.

    Training data format (``train_data`` arg to ``lf.optimize``)::

        # Single DataFrame — one training example
        train_data = df

        # Multiple DataFrames — list of example dicts
        train_data = [
            {"input": df_train1, "expected": labels1},
            {"input": df_train2, "expected": labels2},
        ]

        # Multi-source pipeline — preserves LazyFrame → DataFrame mapping
        train_data = {left_lf: left_df, right_lf: right_df}

    **Basic example** — optimize a filter + map pipeline::

        import pandas as pd
        import lotus
        from lotus.ast import LazyFrame
        from lotus.ast.optimizer import GEPAOptimizer
        from lotus.models import LM

        lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

        df = pd.DataFrame({"review": ["great product!", "terrible, broke in a day", "ok"]})

        def eval_fn(output_df, example):
            # Score: fraction of reviews kept that are actually positive
            positive_kept = sum("great" in r or "ok" in r for r in output_df["review"])
            return positive_kept / max(len(output_df), 1)

        optimizer = GEPAOptimizer(eval_fn=eval_fn)
        lf = LazyFrame(df=df).sem_filter("{review} is a positive product review")
        optimized_lf = lf.optimize([optimizer], train_data=df)
        result = optimized_lf.execute({})

    **With side info** — return diagnostics for better GEPA reflection::

        def eval_fn(output_df, example):
            expected_ids = set(example["expected_ids"])
            got_ids = set(output_df.index)
            precision = len(expected_ids & got_ids) / max(len(got_ids), 1)
            recall = len(expected_ids & got_ids) / max(len(expected_ids), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            return f1, {"precision": precision, "recall": recall, "got": list(got_ids)}

        optimizer = GEPAOptimizer(eval_fn=eval_fn)

    **Selective optimization** — opt specific nodes in or out::

        lf = (
            LazyFrame(df=df)
            # Optimize only this filter's instruction (explicit override)
            .sem_filter("{text} is relevant", mark_optimizable=["user_instruction"])
            # Exclude this map entirely from optimization
            .sem_map("Clean up {text}", mark_optimizable=[])
        )
        # Or annotate post-hoc (node index 1 = first sem_filter):
        lf = lf.mark_optimizable(node_idx=1, params=["user_instruction"])

    **With GEPA config** — control budget and model::

        from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig

        optimizer = GEPAOptimizer(
            eval_fn=eval_fn,
            gepa_config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=200),
                reflection=ReflectionConfig(model="gpt-4o"),
            ),
        )

    **Generalization mode** — separate train and val sets::

        optimizer = GEPAOptimizer(
            eval_fn=eval_fn,
            valset=[{"input": df_val1}, {"input": df_val2}],
        )
        optimized_lf = lf.optimize([optimizer], train_data=[
            {"input": df_train1}, {"input": df_train2}
        ])

    **Multi-source join** — pass a dict to preserve source mappings::

        left_lf = LazyFrame("courses").sem_filter("{Name} is an advanced course")
        right_lf = LazyFrame("skills")
        joined_lf = left_lf.sem_join(right_lf, "{Name} will help learn {Skill}")

        def eval_fn(output_df, example):
            return len(output_df) / 10.0  # higher join volume = better

        optimizer = GEPAOptimizer(eval_fn=eval_fn)
        optimized = joined_lf.optimize(
            [optimizer],
            train_data={left_lf: courses_df, right_lf: skills_df},
        )
    """

    requires_train_data: bool = True

    def __init__(
        self,
        eval_fn: UserEvalFn,
        *,
        valset: list[Any] | None = None,
        gepa_config: Any = None,  # GEPAConfig, kept as Any to allow lazy import
        objective: str | None = None,
        background: str | None = None,
    ) -> None:
        self._eval_fn = eval_fn
        self._valset = valset
        self._gepa_config = gepa_config
        self._objective = objective
        self._background = background

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: "dict[LazyFrame, pd.DataFrame] | pd.DataFrame | list[Any] | None" = None,
    ) -> list[BaseNode]:
        """Optimize pipeline node parameters using GEPA.

        Args:
            nodes: List of AST nodes from the pipeline.
            train_data: Training examples. Can be:
                - ``list[dict]``: Each dict is an example (must have ``"input"`` key).
                - ``pd.DataFrame``: Single example.
                - ``dict[LazyFrame, pd.DataFrame]``: Standard optimizer format.

        Returns:
            New list of nodes with optimized parameter values.
        """
        try:
            from gepa.optimize_anything import GEPAConfig, optimize_anything
        except ImportError as exc:
            raise ImportError("GEPA package not found. Install it with: pip install gepa") from exc

        # 1. Collect optimizable targets
        targets = self._collect_targets(nodes)

        if not targets:
            logger.info("GEPAOptimizer: no optimizable parameters found, returning nodes unchanged.")
            return nodes

        seed_candidate = {t.candidate_key: t.value for t in targets}
        logger.info(f"GEPAOptimizer: optimizing {len(targets)} parameter(s): {list(seed_candidate.keys())}")

        # 2. Normalize training data
        dataset = self._normalize_train_data(train_data)

        # 3. Build objective / background
        objective = self._objective or self._build_default_objective(nodes, targets)
        background = self._background or self._build_default_background(nodes)

        # 4. Shared cache for cross-evaluation reuse
        shared_cache: dict[str, Any] = {}

        # 5. Source node
        source = nodes[0] if nodes and isinstance(nodes[0], SourceNode) else None

        # 6. Build GEPA evaluator wrapper
        evaluator = self._make_evaluator(nodes, targets, shared_cache, source)

        # 7. Run GEPA
        config = self._gepa_config if self._gepa_config is not None else GEPAConfig()

        result = optimize_anything(
            seed_candidate=seed_candidate,
            evaluator=evaluator,
            dataset=dataset,
            valset=self._valset,
            objective=objective,
            background=background,
            config=config,
        )

        logger.info(
            f"GEPAOptimizer: optimization complete. " f"Best score: {result.val_aggregate_scores[result.best_idx]:.4f}"
        )

        # 8. Apply best candidate back to nodes
        return self._apply_candidate(nodes, result.best_candidate, targets)

    # ------------------------------------------------------------------
    # Target collection
    # ------------------------------------------------------------------

    def _collect_targets(self, nodes: list[BaseNode]) -> list[_OptTarget]:
        """Walk all nodes (including nested pipelines) and collect optimizable targets."""
        targets: list[_OptTarget] = []
        counter = [0]  # mutable int for closure
        self._walk_pipeline(nodes, (), targets, counter)
        return targets

    def _walk_pipeline(
        self,
        nodes: list[BaseNode],
        path: PipelinePath,
        targets: list[_OptTarget],
        counter: list[int],
    ) -> None:
        """Walk a pipeline's nodes, collecting targets and recursing into nested LazyFrames."""
        for node_idx, node in enumerate(nodes):
            params = _get_optimizable_params(node)
            if params:
                for param_name in sorted(params):
                    if param_name not in type(node).model_fields:
                        logger.warning(
                            f"Parameter '{param_name}' does not exist on "
                            f"{type(node).__name__} at index {node_idx}. Skipping."
                        )
                        continue

                    # Use pydantic model_dump for JSON-safe serialization
                    dump = node.model_dump(mode="json", include={param_name})
                    value = dump[param_name]
                    is_json = not isinstance(value, str)
                    if is_json:
                        value = json.dumps(value, ensure_ascii=False)

                    targets.append(
                        _OptTarget(
                            node_idx=node_idx,
                            param_name=param_name,
                            value=value,
                            step_idx=counter[0],
                            is_json_encoded=is_json,
                            pipeline_path=path,
                        )
                    )
                counter[0] += 1

            # Recurse into nested LazyFrames
            self._walk_nested(node, node_idx, path, targets, counter)

    def _walk_nested(
        self,
        node: BaseNode,
        node_idx: int,
        path: PipelinePath,
        targets: list[_OptTarget],
        counter: list[int],
    ) -> None:
        """Recurse into nested LazyFrames on join/PandasOp/ApplyFn nodes."""
        from ..pipeline import LazyFrame

        if isinstance(node, (SemJoinNode, SemSimJoinNode)):
            if node.right_pipeline is not None and isinstance(node.right_pipeline, LazyFrame):
                entry: PipelinePathEntry = (node_idx, "right_pipeline")  # type: ignore[assignment]
                self._walk_pipeline(node.right_pipeline._nodes, path + (entry,), targets, counter)

        elif isinstance(node, PandasOpNode):
            if node.pipeline_args:
                for key, pipeline in node.pipeline_args.items():
                    if isinstance(pipeline, LazyFrame):
                        entry = (node_idx, "pipeline_args", key)  # type: ignore[assignment]
                        self._walk_pipeline(pipeline._nodes, path + (entry,), targets, counter)
            if node.pipeline_kwargs:
                for key, pipeline in node.pipeline_kwargs.items():
                    if isinstance(pipeline, LazyFrame):
                        entry = (node_idx, "pipeline_kwargs", key)  # type: ignore[assignment]
                        self._walk_pipeline(pipeline._nodes, path + (entry,), targets, counter)

        elif isinstance(node, ApplyFnNode):
            for arg_idx, arg_value in enumerate(node.args):
                self._walk_applyfn_value(
                    value=arg_value,
                    node_idx=node_idx,
                    path=path,
                    field_name="args",
                    value_path=(arg_idx,),
                    targets=targets,
                    counter=counter,
                )
            if node.kwargs:
                for kwarg_key, kwarg_value in node.kwargs.items():
                    self._walk_applyfn_value(
                        value=kwarg_value,
                        node_idx=node_idx,
                        path=path,
                        field_name="kwargs",
                        value_path=(kwarg_key,),
                        targets=targets,
                        counter=counter,
                    )

    def _walk_applyfn_value(
        self,
        value: Any,
        node_idx: int,
        path: PipelinePath,
        field_name: str,
        value_path: tuple[Any, ...],
        targets: list[_OptTarget],
        counter: list[int],
    ) -> None:
        """Recursively walk a value from ApplyFnNode args/kwargs for nested LazyFrames."""
        from ..pipeline import LazyFrame

        if isinstance(value, LazyFrame):
            entry = (node_idx, field_name, *value_path)
            self._walk_pipeline(value._nodes, path + (entry,), targets, counter)
            return

        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self._walk_applyfn_value(
                    value=item,
                    node_idx=node_idx,
                    path=path,
                    field_name=field_name,
                    value_path=value_path + (idx,),
                    targets=targets,
                    counter=counter,
                )
            return

        if isinstance(value, dict):
            for key, item in value.items():
                self._walk_applyfn_value(
                    value=item,
                    node_idx=node_idx,
                    path=path,
                    field_name=field_name,
                    value_path=value_path + (key,),
                    targets=targets,
                    counter=counter,
                )

    def _get_nested_value(self, value: Any, nested_path: tuple[Any, ...]) -> Any | None:
        """Navigate a nested list/tuple/dict structure and return the value at the given path."""
        current = value
        for key in nested_path:
            if isinstance(current, (list, tuple)):
                if not isinstance(key, int) or key < 0 or key >= len(current):
                    return None
                current = current[key]
            elif isinstance(current, dict):
                if key not in current:
                    return None
                current = current[key]
            else:
                return None
        return current

    def _set_nested_value(self, value: Any, nested_path: tuple[Any, ...], replacement: Any) -> Any:
        """Return a shallow copy of a nested structure with the value at nested_path replaced."""
        if not nested_path:
            return replacement

        key = nested_path[0]
        rest = nested_path[1:]

        if isinstance(value, tuple):
            if not isinstance(key, int):
                return value
            items = list(value)
            if key < 0 or key >= len(items):
                return value
            items[key] = self._set_nested_value(items[key], rest, replacement)
            return tuple(items)

        if isinstance(value, list):
            if not isinstance(key, int):
                return value
            items = list(value)
            if key < 0 or key >= len(items):
                return value
            items[key] = self._set_nested_value(items[key], rest, replacement)
            return items

        if isinstance(value, dict):
            if key not in value:
                return value
            updated = dict(value)
            updated[key] = self._set_nested_value(updated[key], rest, replacement)
            return updated

        return value

    # ------------------------------------------------------------------
    # Candidate application
    # ------------------------------------------------------------------

    def _apply_candidate(
        self,
        nodes: list[BaseNode],
        candidate: dict[str, str],
        targets: list[_OptTarget],
    ) -> list[BaseNode]:
        """Create a new node list with candidate parameter values applied.

        Handles nested LazyFrames by grouping targets by pipeline_path and
        applying updates recursively from deepest to shallowest.
        """
        by_path: dict[PipelinePath, list[_OptTarget]] = defaultdict(list)
        for t in targets:
            by_path[t.pipeline_path].append(t)

        return self._apply_at_path(nodes, candidate, by_path, ())

    def _apply_at_path(
        self,
        nodes: list[BaseNode],
        candidate: dict[str, str],
        by_path: dict[PipelinePath, list[_OptTarget]],
        path: PipelinePath,
    ) -> list[BaseNode]:
        """Recursively apply candidate values at a given pipeline path."""
        from ..pipeline import LazyFrame

        patched = list(nodes)

        # 1. Apply direct parameter updates for targets at this path
        updates_by_node: dict[int, dict[str, Any]] = defaultdict(dict)
        for t in by_path.get(path, []):
            value = candidate.get(t.candidate_key)
            if value is None:
                continue
            resolved: Any = json.loads(value) if t.is_json_encoded else value
            updates_by_node[t.node_idx][t.param_name] = resolved

        for idx, updates in updates_by_node.items():
            patched[idx] = patched[idx].model_copy(update=updates)

        # 2. Recursively handle child pipeline paths
        for child_path in by_path:
            if len(child_path) != len(path) + 1 or child_path[: len(path)] != path:
                continue

            entry = child_path[-1]
            node_idx = int(entry[0])

            if len(entry) == 2:
                # Direct field: (node_idx, field_name)
                field_name = str(entry[1])
                nested_lf = getattr(patched[node_idx], field_name)
                if nested_lf is not None and isinstance(nested_lf, LazyFrame):
                    new_nodes = self._apply_at_path(nested_lf._nodes, candidate, by_path, child_path)
                    new_lf = LazyFrame(_nodes=new_nodes, _source=nested_lf._source)
                    patched[node_idx] = patched[node_idx].model_copy(update={field_name: new_lf})

            elif len(entry) >= 3 and str(entry[1]) in {"args", "kwargs"}:
                # ApplyFn nested structure: (node_idx, "args"/"kwargs", *nested_path)
                field_name = str(entry[1])
                nested_path = tuple(entry[2:])
                root_value = getattr(patched[node_idx], field_name)
                nested_lf = self._get_nested_value(root_value, nested_path)
                if nested_lf is not None and isinstance(nested_lf, LazyFrame):
                    new_nodes = self._apply_at_path(nested_lf._nodes, candidate, by_path, child_path)
                    new_lf = LazyFrame(_nodes=new_nodes, _source=nested_lf._source)
                    updated_root = self._set_nested_value(root_value, nested_path, new_lf)
                    patched[node_idx] = patched[node_idx].model_copy(update={field_name: updated_root})

            elif len(entry) == 3:
                # Dict field: (node_idx, dict_field_name, dict_key)
                dict_field, dict_key = str(entry[1]), str(entry[2])
                nested_dict = dict(getattr(patched[node_idx], dict_field) or {})
                nested_lf = nested_dict.get(dict_key)
                if nested_lf is not None and isinstance(nested_lf, LazyFrame):
                    new_nodes = self._apply_at_path(nested_lf._nodes, candidate, by_path, child_path)
                    new_lf = LazyFrame(_nodes=new_nodes, _source=nested_lf._source)
                    nested_dict[dict_key] = new_lf
                    patched[node_idx] = patched[node_idx].model_copy(update={dict_field: nested_dict})

        return patched

    # ------------------------------------------------------------------
    # Pipeline description
    # ------------------------------------------------------------------

    def _describe_pipeline(self, nodes: list[BaseNode], indent: int = 0) -> str:
        """Generate a human-readable pipeline description with param details.

        Uses each node's docstring for the operator description and includes
        field-level descriptions from pydantic Field metadata for optimizable params.
        Also recurses into nested LazyFrames (joins, PandasOps).
        """
        from ..pipeline import LazyFrame

        prefix = "  " * indent
        lines: list[str] = []

        for i, node in enumerate(nodes):
            node_type = type(node).__name__
            # Use node class docstring (first line) as the operator description
            doc = (type(node).__doc__ or "").strip().split("\n")[0]
            line = f"{prefix}  Step {i} ({node_type}): {doc}"

            # Add optimizable param values and their Field descriptions
            params = _get_optimizable_params(node)
            if params:
                param_details: list[str] = []
                for pname in sorted(params):
                    if pname not in type(node).model_fields:
                        continue
                    val = getattr(node, pname)
                    preview = str(val)
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    field_desc = type(node).model_fields[pname].description or ""
                    if field_desc:
                        param_details.append(f"{prefix}    - {pname} ({field_desc}): {preview!r}")
                    else:
                        param_details.append(f"{prefix}    - {pname}: {preview!r}")
                if param_details:
                    line += "\n" + "\n".join(param_details)

            lines.append(line)

            # Describe nested pipelines
            if isinstance(node, (SemJoinNode, SemSimJoinNode)):
                if node.right_pipeline is not None and isinstance(node.right_pipeline, LazyFrame):
                    lines.append(f"{prefix}    [right pipeline]:")
                    lines.append(self._describe_pipeline(node.right_pipeline._nodes, indent + 2))

            elif isinstance(node, PandasOpNode):
                if node.pipeline_args:
                    for key, pipeline in node.pipeline_args.items():
                        if isinstance(pipeline, LazyFrame):
                            lines.append(f"{prefix}    [pipeline arg {key}]:")
                            lines.append(self._describe_pipeline(pipeline._nodes, indent + 2))
                if node.pipeline_kwargs:
                    for key, pipeline in node.pipeline_kwargs.items():
                        if isinstance(pipeline, LazyFrame):
                            lines.append(f"{prefix}    [pipeline kwarg {key}]:")
                            lines.append(self._describe_pipeline(pipeline._nodes, indent + 2))

            elif isinstance(node, ApplyFnNode):
                self._describe_applyfn_nested(node, lines, prefix, indent)

        return "\n".join(lines)

    def _describe_applyfn_nested(
        self,
        node: ApplyFnNode,
        lines: list[str],
        prefix: str,
        indent: int,
    ) -> None:
        """Append pipeline descriptions for any LazyFrames nested in ApplyFnNode args/kwargs."""
        from ..pipeline import LazyFrame

        def _scan(value: Any, label: str) -> None:
            if isinstance(value, LazyFrame):
                lines.append(f"{prefix}    [{label}]:")
                lines.append(self._describe_pipeline(value._nodes, indent + 2))
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    _scan(item, f"{label}[{idx}]")
            elif isinstance(value, dict):
                for key, item in value.items():
                    _scan(item, f"{label}.{key}")

        for arg_idx, arg_value in enumerate(node.args):
            _scan(arg_value, f"arg {arg_idx}")
        if node.kwargs:
            for kwarg_key, kwarg_value in node.kwargs.items():
                _scan(kwarg_value, f"kwarg {kwarg_key}")

    # ------------------------------------------------------------------
    # Default OBJECTIVE and BACKGROUND
    # ------------------------------------------------------------------

    def _build_default_objective(self, nodes: list[BaseNode], targets: list[_OptTarget]) -> str:
        """Generate the default OBJECTIVE for GEPA optimization."""
        pipeline_desc = self._describe_pipeline(nodes)
        param_lines: list[str] = []
        for t in targets:
            preview = t.value[:60] + "..." if len(t.value) > 60 else t.value
            # Include field description if available
            node = nodes[t.node_idx] if not t.pipeline_path else None
            field_desc = ""
            if node is not None and t.param_name in type(node).model_fields:
                field_desc = type(node).model_fields[t.param_name].description or ""
            desc_suffix = f" ({field_desc})" if field_desc else ""
            param_lines.append(f"  - {t.candidate_key}{desc_suffix}: {preview!r}")
        params_desc = "\n".join(param_lines)

        return (
            "Optimize the natural language instructions in a LOTUS semantic data processing pipeline "
            "to maximize the evaluation metric.\n\n"
            f"Pipeline structure:\n{pipeline_desc}\n\n"
            f"Parameters being optimized:\n{params_desc}"
        )

    def _build_default_background(self, nodes: list[BaseNode]) -> str:
        """Generate the default BACKGROUND for GEPA optimization."""
        pipeline_desc = self._describe_pipeline(nodes)

        return (
            'LOTUS semantic operators use natural language instructions ("langex") parameterized by '
            'column names in curly braces. E.g., "{Course Name} requires math prerequisites".\n\n'
            "Operator reference:\n"
            "- sem_filter(instruction): Keep rows where instruction evaluates to true\n"
            "- sem_map(instruction): Transform each row, adding a result column\n"
            "- sem_agg(instruction): Summarize all rows into a single result\n"
            "- sem_topk(instruction, K): Rank and return top-K rows by criterion\n"
            "- sem_join(instruction): Join two tables on a natural language predicate\n"
            "- sem_search(query, K): Return K rows most semantically similar to query\n"
            "- sem_extract(output_cols): Extract structured attributes from text\n\n"
            "Instructions should:\n"
            "- Be precise and unambiguous\n"
            "- Reference DataFrame columns using {ColumnName} syntax\n"
            "- Be grounded in the actual data content and schema\n\n"
            f"Full pipeline (for context):\n{pipeline_desc}"
        )

    # ------------------------------------------------------------------
    # Evaluator wrapper
    # ------------------------------------------------------------------

    def _make_evaluator(
        self,
        nodes: list[BaseNode],
        targets: list[_OptTarget],
        shared_cache: dict[str, Any],
        source: SourceNode | None,
    ) -> Callable[[dict[str, str], Any], tuple[float, dict[str, Any]]]:
        """Build the evaluator closure that GEPA calls for each (candidate, example) pair."""
        from ..pipeline import LazyFrame
        from ..run import LazyFrameRun

        def evaluator(candidate: dict[str, str], example: Any) -> tuple[float, dict[str, Any]]:
            side_info: dict[str, Any] = {}

            # 1. Patch nodes with candidate values
            patched_nodes = self._apply_candidate(nodes, candidate, targets)

            # 2. Build a temporary LazyFrame and execute
            temp_lf = LazyFrame(_nodes=patched_nodes, _source=source)
            input_df = example["input"] if isinstance(example, dict) else example

            try:
                run = LazyFrameRun(temp_lf, input_df, _shared_cache=shared_cache)
                output = run.execute()
            except Exception as e:
                side_info["execution_error"] = f"{type(e).__name__}: {e}"
                logger.warning(f"Pipeline execution failed: {e}")
                return 0.0, side_info

            # 3. Capture output summary as side info
            if isinstance(output, pd.DataFrame):
                side_info["output_rows"] = len(output)
                side_info["output_columns"] = list(output.columns)
                if len(output) > 0:
                    side_info["output_sample"] = output.head(3).to_dict("records")
            else:
                side_info["output_type"] = type(output).__name__
                side_info["output_preview"] = str(output)[:500]

            # 4. Call user's eval function
            try:
                result = self._eval_fn(output, example)
            except Exception as e:
                side_info["eval_error"] = f"{type(e).__name__}: {e}"
                logger.warning(f"User eval function failed: {e}")
                return 0.0, side_info

            if isinstance(result, tuple):
                score, user_side_info = result
                side_info.update(user_side_info)
            else:
                score = result

            return float(score), side_info

        return evaluator

    # ------------------------------------------------------------------
    # Train data normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_train_data(
        train_data: "dict[LazyFrame, pd.DataFrame] | pd.DataFrame | list[Any] | None",
    ) -> list[Any]:
        """Normalize train_data into a list of examples for GEPA."""
        if train_data is None:
            raise ValueError("GEPAOptimizer requires train_data. Pass it via lf.optimize([optimizer], train_data=...).")

        if isinstance(train_data, list):
            return train_data

        if isinstance(train_data, pd.DataFrame):
            return [{"input": train_data}]

        if isinstance(train_data, dict):
            return [{"input": train_data}]

        raise TypeError(f"Unsupported train_data type: {type(train_data)}")
