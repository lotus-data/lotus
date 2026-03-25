"""GEPA-based optimizer for LOTUS AST LazyFrames.

Uses GEPA's ``optimize_anything`` to evolve natural language instructions
(langex) in semantic operator nodes via LLM-guided evolutionary search.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from lotus.cache import Cache, CacheFactory

from ..nodes import (
    BaseNode,
    PairwiseJudgeNode,
    SemAggNode,
    SemFilterNode,
    SemJoinNode,
    SemMapNode,
    SemSearchNode,
    SemTopKNode,
    SourceNode,
)
from .base import BaseOptimizer

if TYPE_CHECKING:
    from gepa.optimize_anything import GEPAConfig

    from ..lazyframe import LazyFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default optimizable parameters per node type
# ---------------------------------------------------------------------------

DEFAULT_OPTIMIZABLE_PARAMS: dict[type, frozenset[str]] = {
    SemFilterNode: frozenset({"user_instruction", "cascade_args.helper_filter_instruction"}),
    PairwiseJudgeNode: frozenset({"judge_instruction", "cascade_args.helper_filter_instruction"}),
    SemMapNode: frozenset({"user_instruction"}),
    SemAggNode: frozenset({"user_instruction"}),
    SemTopKNode: frozenset({"user_instruction"}),
    SemJoinNode: frozenset({"join_instruction"}),
    SemSearchNode: frozenset({"query"}),
}

UserEvalFn = Callable[..., "float | tuple[float, dict[str, Any]]"]


# ---------------------------------------------------------------------------
# PathEntry — navigation to nested LazyFrames
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PathEntry:
    """One step in the path from a parent lazyframe to a nested LazyFrame.

    Addresses a LazyFrame at ``getattr(node, field_name)`` navigated
    further by ``sub_path``.  When ``sub_path`` is empty the field itself
    is the LazyFrame (e.g. a join's ``right_lf``).  When non-empty the
    sub-path indexes into a nested list/tuple/dict structure (e.g. a
    ``PandasOpNode.lf_args["key"]`` or ``ApplyFnNode.args[0][1]``).
    """

    node_idx: int
    field_name: str = field(default="")
    sub_path: tuple[Any, ...] = field(default=())

    # -- navigation --------------------------------------------------------

    def get_lf(self, node: BaseNode) -> "LazyFrame | None":
        """Extract the nested LazyFrame from *node*."""
        from ..lazyframe import LazyFrame

        root = getattr(node, self.field_name, None)
        if root is None:
            return None
        current = root
        for key in self.sub_path:
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
        return current if isinstance(current, LazyFrame) else None

    def set_lf(self, node: BaseNode, new_lf: "LazyFrame") -> BaseNode:
        """Return a copy of *node* with the nested LazyFrame replaced."""
        if not self.sub_path:
            return node.model_copy(update={self.field_name: new_lf})
        root = getattr(node, self.field_name)
        updated = self._set_nested(root, self.sub_path, new_lf)
        return node.model_copy(update={self.field_name: updated})

    # -- nested-structure utilities ----------------------------------------

    @staticmethod
    def _get_nested(value: Any, path: tuple[Any, ...]) -> Any | None:
        """Navigate a nested list/tuple/dict and return the leaf value."""
        current = value
        for key in path:
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

    @staticmethod
    def _set_nested(value: Any, path: tuple[Any, ...], replacement: Any) -> Any:
        """Return a shallow copy of *value* with the leaf at *path* replaced."""
        if not path:
            return replacement

        key, rest = path[0], path[1:]

        if isinstance(value, (list, tuple)):
            if not isinstance(key, int) or key < 0 or key >= len(value):
                return value
            items = list(value)
            items[key] = PathEntry._set_nested(items[key], rest, replacement)
            return type(value)(items) if isinstance(value, tuple) else items

        if isinstance(value, dict):
            if key not in value:
                return value
            return {k: (PathEntry._set_nested(v, rest, replacement) if k == key else v) for k, v in value.items()}

        return value

    # -- collection --------------------------------------------------------

    @staticmethod
    def collect(node: BaseNode, node_idx: int) -> "list[tuple[PathEntry, LazyFrame]]":
        """Collect all nested LazyFrame refs from a single node."""
        from ..lazyframe import LazyFrame

        if isinstance(node, SourceNode):
            return []

        results: list[tuple[PathEntry, LazyFrame]] = []

        def _scan(value: Any, fname: str, sp: tuple[Any, ...]) -> None:
            if isinstance(value, LazyFrame):
                results.append((PathEntry(node_idx, fname, sp), value))
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    _scan(item, fname, sp + (idx,))
            elif isinstance(value, dict):
                for k, item in value.items():
                    _scan(item, fname, sp + (k,))

        for fname in type(node).model_fields:
            root = getattr(node, fname, None)
            if root is not None:
                _scan(root, fname, ())

        return results


# Convenience alias
PathToLF = tuple[PathEntry, ...]


# ---------------------------------------------------------------------------
# Optimization target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OptTarget:
    """A single optimizable parameter somewhere in the lazyframe tree.

    ``path`` is a sequence of ``PathEntry`` steps from the root lazyframe
    to the sub-lazyframe containing this parameter.  Root-level targets
    have ``path = ()``.
    """

    node_idx: int
    param_name: str
    value: str
    step_idx: int
    is_json_encoded: bool = False
    path: PathToLF = ()

    @property
    def candidate_key(self) -> str:
        """GEPA candidate dict key for this target."""
        return f"step{self.step_idx}_{self.param_name}"

    @classmethod
    def parse_key(cls, key: str) -> tuple[int, str]:
        """Parse a candidate key back into ``(step_idx, param_name)``."""
        prefix, _, param_name = key.partition("_")
        return int(prefix.removeprefix("step")), param_name


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_optimizable_params(node: BaseNode) -> frozenset[str]:
    """Return the set of params to optimize for *node*.

    Uses ``node.optimizable_params`` if set, else ``DEFAULT_OPTIMIZABLE_PARAMS``.
    """
    optimizable_params = (
        node.optimizable_params
        if node.optimizable_params is not None
        else DEFAULT_OPTIMIZABLE_PARAMS.get(type(node), frozenset())
    )
    supported = frozenset(param for param in optimizable_params if node.supports_optimizable_param(param))
    unsupported = sorted(set(optimizable_params) - set(supported))
    if unsupported and node.optimizable_params is not None:
        logger.warning(
            "Node type %s does not support optimizable params: %s, skipping them",
            type(node).__name__,
            unsupported,
        )

    return supported


# ---------------------------------------------------------------------------
# GEPAOptimizer
# ---------------------------------------------------------------------------


class GEPAOptimizer(BaseOptimizer):
    """GEPA-based prompt/instruction optimizer for LOTUS LazyFrames.

    Automatically optimizes natural language instructions in semantic operator
    nodes using LLM-guided evolutionary search (GEPA).

    By default, ``user_instruction`` on sem_filter/sem_map/sem_agg/sem_topk,
    ``join_instruction`` on sem_join, and ``query`` on sem_search are optimized.
    For sem_filter cascades that use ``HELPER_LM``, the helper prompt target
    ``cascade_args.helper_filter_instruction`` is also optimized.
    The same helper prompt target is exposed for ``pairwise_judge`` nodes in
    ``mode="sem_filter"`` when using helper-LM cascades.
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
            Auto-generated from the LazyFrame structure when ``None``.
        background: Domain context / constraints string for the reflection LLM.
            Auto-generated with LOTUS operator reference when ``None``.

    Example::
        def eval_fn(output_df, example):
            # Score: fraction of reviews kept that are actually positive
            positive_kept = sum("great" in r or "ok" in r for r in output_df["review"])
            return positive_kept / max(len(output_df), 1)

        optimizer = GEPAOptimizer(eval_fn=eval_fn)
        lf = LazyFrame(df=df).sem_filter("{review} is a positive product review")
        optimized_lf = lf.optimize([optimizer], train_data=df)
        result = optimized_lf.execute({})
    """

    requires_train_data: bool = True

    def __init__(
        self,
        eval_fn: UserEvalFn,
        *,
        valset: "dict[LazyFrame, pd.DataFrame] | pd.DataFrame | list[Any] | None" = None,
        gepa_config: GEPAConfig | None = None,
        objective: str | None = None,
        background: str | None = None,
        cache: Cache | None = None,
        include_output_in_side_info: bool = True,
    ) -> None:
        self._eval_fn = eval_fn
        self._valset = self._normalize_input(valset) if valset is not None else None
        self._gepa_config = gepa_config
        self._objective = objective
        self._background = background
        self._cache = cache or CacheFactory.create_default_cache(max_size=10_000)
        self._include_output_in_side_info = include_output_in_side_info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: "dict[LazyFrame, pd.DataFrame] | pd.DataFrame | list[Any] | None" = None,
    ) -> list[BaseNode]:
        """Optimize LazyFrame node parameters using GEPA.

        Returns a new list of nodes with optimized parameter values.
        """
        try:
            from gepa.optimize_anything import GEPAConfig, optimize_anything
        except ImportError as exc:
            raise ImportError("GEPA package not found. Install it with: pip install gepa") from exc

        targets = self._collect_targets(nodes)
        if not targets:
            logger.info("GEPAOptimizer: no optimizable parameters found.")
            return nodes

        seed_candidate = {t.candidate_key: t.value for t in targets}
        logger.info("GEPAOptimizer: optimizing %d parameter(s): %s", len(targets), list(seed_candidate.keys()))

        dataset = self._normalize_input(train_data)
        objective = self._objective or self._build_default_objective(nodes, targets)
        background = self._background or self._build_default_background(nodes)

        source = nodes[0] if nodes and isinstance(nodes[0], SourceNode) else None
        evaluator = self._make_evaluator(nodes, targets, self._cache, source)

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
            "GEPAOptimizer: complete. Best score: %.4f",
            result.val_aggregate_scores[result.best_idx],
        )
        print(f"Result: {result.best_candidate}")
        return self._apply_candidate(nodes, result.best_candidate, targets)

    # ------------------------------------------------------------------
    # Target collection
    # ------------------------------------------------------------------

    def _collect_targets(self, nodes: list[BaseNode]) -> list[_OptTarget]:
        """Walk all nodes (including nested pipelines) and collect optimizable targets."""
        targets: list[_OptTarget] = []
        counter = [0]
        seen_step_by_node_id: dict[int, int] = {}
        seen_target_keys: set[tuple[int, str]] = set()
        self._walk(
            nodes,
            (),
            targets,
            counter,
            seen_step_by_node_id,
            dedupe_alias_paths=True,
            seen_target_keys=seen_target_keys,
        )
        print(f"Targets: {targets}")
        return targets

    def _collect_target_occurrences(self, nodes: list[BaseNode]) -> list[_OptTarget]:
        """Collect per-path target occurrences while reusing shared-node step ids."""
        occurrences: list[_OptTarget] = []
        counter = [0]
        seen_step_by_node_id: dict[int, int] = {}
        self._walk(
            nodes,
            (),
            occurrences,
            counter,
            seen_step_by_node_id,
            dedupe_alias_paths=False,
            seen_target_keys=None,
        )
        return occurrences

    def _walk(
        self,
        nodes: list[BaseNode],
        path: PathToLF,
        targets: list[_OptTarget],
        counter: list[int],
        seen_step_by_node_id: dict[int, int],
        *,
        dedupe_alias_paths: bool,
        seen_target_keys: set[tuple[int, str]] | None,
    ) -> None:
        """Collect targets from *nodes* and recurse into nested LazyFrame refs."""
        for node_idx, node in enumerate(nodes):
            params = _get_optimizable_params(node)
            if params:
                node_id = id(node)
                step_idx = seen_step_by_node_id.get(node_id)
                if step_idx is None:
                    step_idx = counter[0]
                    seen_step_by_node_id[node_id] = step_idx
                    counter[0] += 1

                for param_name in sorted(params):
                    value = node.resolve_optimizable_param_value(param_name)
                    is_json = not isinstance(value, str)
                    if is_json:
                        value = json.dumps(value, ensure_ascii=False)

                    target_key = (step_idx, param_name)
                    if dedupe_alias_paths and seen_target_keys is not None:
                        if target_key in seen_target_keys:
                            continue
                        seen_target_keys.add(target_key)

                    targets.append(
                        _OptTarget(
                            node_idx=node_idx,
                            param_name=param_name,
                            value=value,
                            step_idx=step_idx,
                            is_json_encoded=is_json,
                            path=path,
                        )
                    )

            # Check all the lazyframes in the member fields of current node
            for entry, nested_lf in PathEntry.collect(node, node_idx):
                self._walk(
                    nested_lf._nodes,
                    path + (entry,),
                    targets,
                    counter,
                    seen_step_by_node_id,
                    dedupe_alias_paths=dedupe_alias_paths,
                    seen_target_keys=seen_target_keys,
                )

    # ------------------------------------------------------------------
    # Candidate application
    # ------------------------------------------------------------------

    def _apply_candidate(
        self,
        nodes: list[BaseNode],
        candidate: dict[str, str],
        targets: list[_OptTarget],
    ) -> list[BaseNode]:
        """Create a new node list with candidate values applied.

        Groups targets by ``path`` and applies updates recursively
        from deepest to shallowest.
        """
        allowed_candidate_keys = {t.candidate_key for t in targets}
        occurrences = self._collect_target_occurrences(nodes)
        by_path: dict[PathToLF, list[_OptTarget]] = defaultdict(list)
        for t in occurrences:
            if t.candidate_key not in allowed_candidate_keys:
                continue
            by_path[t.path].append(t)

        # Work on a deep copy so evaluator runs cannot mutate the caller's
        # node tree through in-node state updates or nested LazyFrame paths.
        copied_nodes = deepcopy(nodes)
        self._restore_source_refs(nodes, copied_nodes)
        new_nodes = self._apply_at_path(copied_nodes, candidate, by_path, ())
        return new_nodes

    def _restore_source_refs(self, original_nodes: list[BaseNode], copied_nodes: list[BaseNode]) -> None:
        """Restore SourceNode.lazyframe_ref identity after deepcopy.

        ``deepcopy`` clones ``lazyframe_ref`` objects, which breaks input
        resolution in ``LazyFrameRun`` when examples are provided as
        ``dict[LazyFrame, DataFrame]`` with multiple keys. We keep copied
        nodes for immutability, but reattach original source references.
        """
        for idx, (original, copied) in enumerate(zip(original_nodes, copied_nodes)):
            if isinstance(original, SourceNode) and isinstance(copied, SourceNode):
                copied.lazyframe_ref = original.lazyframe_ref

            original_children = PathEntry.collect(original, idx)
            if not original_children:
                continue

            copied_children = {
                (entry.field_name, entry.sub_path): nested_lf for entry, nested_lf in PathEntry.collect(copied, idx)
            }
            for original_entry, original_lf in original_children:
                copied_lf = copied_children.get((original_entry.field_name, original_entry.sub_path))
                if copied_lf is None:
                    continue
                self._restore_source_refs(original_lf._nodes, copied_lf._nodes)

    def _apply_at_path(
        self,
        nodes: list[BaseNode],
        candidate: dict[str, str],
        by_path: dict[PathToLF, list[_OptTarget]],
        path: PathToLF,
        *,
        _applied_nodes: dict[int, BaseNode] | None = None,
    ) -> list[BaseNode]:
        """Recursively apply candidate values at a given lazyframe path.

        ``_applied_nodes`` maps ``id(original_node) → updated_node`` across
        all recursive calls so that nodes shared between the main pipeline
        and nested LazyFrames (e.g. after ``deepcopy``) remain the **same**
        Python object after updates.  This preserves identity for downstream
        optimizers like ``CascadeOptimizer`` which mutate
        nodes in-place.
        """
        from ..lazyframe import LazyFrame

        if _applied_nodes is None:
            _applied_nodes = {}

        patched = list(nodes)

        # Replace any nodes that were already updated at a different path
        for idx in range(len(patched)):
            orig_id = id(patched[idx])
            if orig_id in _applied_nodes:
                patched[idx] = _applied_nodes[orig_id]

        # Direct parameter updates for targets at this path
        updates_by_node: dict[int, list[tuple[str, Any]]] = defaultdict(list)
        for t in by_path.get(path, []):
            value = candidate.get(t.candidate_key)
            if value is None:
                continue
            parsed_value = json.loads(value) if t.is_json_encoded else value
            updates_by_node[t.node_idx].append((t.param_name, parsed_value))

        for idx, updates in updates_by_node.items():
            orig_id = id(nodes[idx])
            if orig_id in _applied_nodes:
                # Already updated via a shared reference at another path
                patched[idx] = _applied_nodes[orig_id]
            else:
                updated_node = patched[idx]
                for param_name, param_value in updates:
                    updated_node = updated_node.apply_optimizable_param_value(param_name, param_value)
                _applied_nodes[orig_id] = updated_node
                patched[idx] = updated_node

        # Recurse into child paths
        for child_path in by_path:
            if len(child_path) != len(path) + 1 or child_path[: len(path)] != path:
                continue

            entry = child_path[-1]
            nested_lf = entry.get_lf(patched[entry.node_idx])
            if nested_lf is not None:
                new_nodes = self._apply_at_path(
                    nested_lf._nodes, candidate, by_path, child_path, _applied_nodes=_applied_nodes
                )
                new_lf = LazyFrame(_nodes=new_nodes, _source=nested_lf._source)
                patched[entry.node_idx] = entry.set_lf(patched[entry.node_idx], new_lf)

        return patched

    # ------------------------------------------------------------------
    # LazyFrame description (for GEPA objective / background)
    # ------------------------------------------------------------------

    def _describe_lf(self, nodes: list[BaseNode], indent: int = 0) -> str:
        """Human-readable LazyFrame description with optimizable param details.

        Delegates to ``node.child_lfs()`` for nested LazyFrame discovery so
        this method doesn't need to know about individual node structures.
        """
        prefix = "  " * indent
        lines: list[str] = []

        for i, node in enumerate(nodes):
            node_type = type(node).__name__
            doc = (type(node).__doc__ or "").strip().split("\n")[0]
            line = f"{prefix}  Step {i} ({node_type}): {doc}"

            params = _get_optimizable_params(node)
            if params:
                param_details: list[str] = []
                for pname in sorted(params):
                    if not node.supports_optimizable_param(pname):
                        continue
                    val = node.resolve_optimizable_param_value(pname)
                    preview = str(val)
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    field_desc = node.optimizable_param_description(pname)
                    if field_desc:
                        param_details.append(f"{prefix}    - {pname} ({field_desc}): {preview!r}")
                    else:
                        param_details.append(f"{prefix}    - {pname}: {preview!r}")
                if param_details:
                    line += "\n" + "\n".join(param_details)

            lines.append(line)

            for label, child_lf in node.child_lfs():
                lines.append(f"{prefix}    [{label}]:")
                lines.append(self._describe_lf(child_lf._nodes, indent + 2))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Default OBJECTIVE and BACKGROUND
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_target_node(nodes: list[BaseNode], target: _OptTarget) -> BaseNode | None:
        current_nodes = nodes
        for entry in target.path:
            if entry.node_idx < 0 or entry.node_idx >= len(current_nodes):
                return None
            nested_lf = entry.get_lf(current_nodes[entry.node_idx])
            if nested_lf is None:
                return None
            current_nodes = nested_lf._nodes

        if target.node_idx < 0 or target.node_idx >= len(current_nodes):
            return None
        return current_nodes[target.node_idx]

    def _build_default_objective(self, nodes: list[BaseNode], targets: list[_OptTarget]) -> str:
        """Generate the default OBJECTIVE for GEPA optimisation."""
        param_lines: list[str] = []
        for t in targets:
            preview = t.value[:60] + "..." if len(t.value) > 60 else t.value
            node = self._resolve_target_node(nodes, t)
            field_desc = ""
            if node is not None:
                field_desc = node.optimizable_param_description(t.param_name)
            desc_suffix = f" ({field_desc})" if field_desc else ""
            param_lines.append(f"  - {t.candidate_key}{desc_suffix}: {preview!r}")

        return (
            "Optimize the parameters in a LOTUS semantic data processing LazyFrame "
            "to maximize the evaluation metric.\n\n"
            "Parameters being optimized:\n" + "\n".join(param_lines)
        )

    def _build_default_background(self, nodes: list[BaseNode]) -> str:
        """Generate the default BACKGROUND for GEPA optimisation."""
        lf_desc = self._describe_lf(nodes)

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
            f"Full LazyFrame (for context):\n{lf_desc}"
        )

    # ------------------------------------------------------------------
    # Evaluator wrapper
    # ------------------------------------------------------------------

    def _make_evaluator(
        self,
        nodes: list[BaseNode],
        targets: list[_OptTarget],
        cache: Cache,
        source: SourceNode | None,
    ) -> Callable[[dict[str, str], Any], tuple[float, dict[str, Any]]]:
        """Build the evaluator closure that GEPA calls per (candidate, example) pair."""
        from ..lazyframe import LazyFrame
        from ..run import LazyFrameRun

        def evaluator(candidate: dict[str, str], example: Any) -> tuple[float, dict[str, Any]]:
            side_info: dict[str, Any] = {}

            patched_nodes = self._apply_candidate(nodes, candidate, targets)
            temp_lf = LazyFrame(_nodes=patched_nodes)
            input_df = example["input"] if isinstance(example, dict) and "input" in example else example

            try:
                output = LazyFrameRun(temp_lf, input_df, cache=cache).execute()
            except Exception as e:
                side_info["execution_error"] = f"{type(e).__name__}: {e}"
                logger.warning("LazyFrame execution failed: %s", e)
                return 0.0, side_info

            if self._include_output_in_side_info:
                if isinstance(output, pd.DataFrame):
                    side_info["output_rows"] = len(output)
                    side_info["output_columns"] = list(output.columns)
                    if len(output) > 0:
                        side_info["output_sample"] = output.head(3).to_dict("records")
                else:
                    side_info["output_type"] = type(output).__name__
                    side_info["output_preview"] = str(output)[:500]

            try:
                result = self._eval_fn(output, example)
            except Exception as e:
                side_info["eval_error"] = f"{type(e).__name__}: {e}"
                logger.warning("User eval function failed: %s", e)
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
    def _normalize_input(
        train_data: "dict[LazyFrame, pd.DataFrame] | pd.DataFrame | list[Any] | None",
    ) -> list[Any] | None:
        """Normalize *train_data* into a list of examples for GEPA."""
        if train_data is None:
            return None

        if isinstance(train_data, list):
            return train_data

        if isinstance(train_data, pd.DataFrame):
            return [{"input": train_data}]

        if isinstance(train_data, dict):
            return [{"input": train_data}]

        raise TypeError(f"Unsupported train_data type: {type(train_data)}")
