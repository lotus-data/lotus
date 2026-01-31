"""AST (Abstract Syntax Tree) for LOTUS semantic operator pipelines.

Captures a user's program of chained semantic operators as a tree of nodes,
supporting lineage queries (ancestors and descendants) for each operator.

Also provides a ``LazyFrame`` wrapper that records a pipeline of semantic
operators as an AST without executing them, then replays via ``.execute()``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import pandas as pd


class ASTNode:
    """Base class for all AST nodes."""

    op_type: str = "unknown"

    def __init__(
        self,
        instruction: str = "",
        parents: list[ASTNode] | None = None,
        **kwargs: Any,
    ) -> None:
        self.instruction = instruction
        self.kwargs = kwargs
        self.parents: list[ASTNode] = parents or []
        self.children: list[ASTNode] = []

        # Register this node as a child of each parent
        for p in self.parents:
            p.children.append(self)

    # ------------------------------------------------------------------
    # Lineage helpers
    # ------------------------------------------------------------------

    def get_ancestors(self) -> list[ASTNode]:
        """Return all ancestors via BFS through parents (excludes self)."""
        visited: set[int] = set()
        result: list[ASTNode] = []
        queue: deque[ASTNode] = deque(self.parents)
        for p in self.parents:
            visited.add(id(p))
        while queue:
            node = queue.popleft()
            result.append(node)
            for p in node.parents:
                if id(p) not in visited:
                    visited.add(id(p))
                    queue.append(p)
        return result

    def get_descendants(self) -> list[ASTNode]:
        """Return all descendants via BFS through children (excludes self)."""
        visited: set[int] = set()
        result: list[ASTNode] = []
        queue: deque[ASTNode] = deque(self.children)
        for c in self.children:
            visited.add(id(c))
        while queue:
            node = queue.popleft()
            result.append(node)
            for c in node.children:
                if id(c) not in visited:
                    visited.add(id(c))
                    queue.append(c)
        return result

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _label(self) -> str:
        instr = f'("{self.instruction}")' if self.instruction else ""
        return f"{type(self).__name__}{instr}"

    def print_ancestors(self) -> None:
        ancestors = self.get_ancestors()
        names = ", ".join(a._label() for a in ancestors) if ancestors else "(none)"
        print(f"Ancestors of {self._label()}: {names}")

    def print_descendants(self) -> None:
        descendants = self.get_descendants()
        names = ", ".join(d._label() for d in descendants) if descendants else "(none)"
        print(f"Descendants of {self._label()}: {names}")

    def _get_roots(self) -> list[ASTNode]:
        """Find all root nodes reachable from this node."""
        visited: set[int] = {id(self)}
        roots: list[ASTNode] = []
        queue: deque[ASTNode] = deque([self])
        while queue:
            node = queue.popleft()
            if not node.parents:
                roots.append(node)
            for p in node.parents:
                if id(p) not in visited:
                    visited.add(id(p))
                    queue.append(p)
        return roots

    def print_tree(self) -> None:
        """Print the full tree starting from the root(s)."""
        roots = self._get_roots()
        visited: set[int] = set()
        for root in roots:
            self._print_subtree(root, "", True, True, visited)

    def _print_subtree(
        self, node: ASTNode, prefix: str, is_root: bool, is_last: bool, visited: set[int]
    ) -> None:
        if is_root:
            print(f"{prefix}{node._label()}")
        else:
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{node._label()}")

        if id(node) in visited:
            return
        visited.add(id(node))

        if is_root:
            child_prefix = prefix
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            self._print_subtree(
                child, child_prefix, False, i == len(node.children) - 1, visited
            )

    # ------------------------------------------------------------------
    # Chaining methods — each returns a new operator node
    # ------------------------------------------------------------------

    def sem_filter(self, instruction: str = "", **kwargs: Any) -> SemFilterNode:
        return SemFilterNode(instruction=instruction, parents=[self], **kwargs)

    def sem_map(self, instruction: str = "", **kwargs: Any) -> SemMapNode:
        return SemMapNode(instruction=instruction, parents=[self], **kwargs)

    def sem_extract(self, instruction: str = "", **kwargs: Any) -> SemExtractNode:
        return SemExtractNode(instruction=instruction, parents=[self], **kwargs)

    def sem_agg(self, instruction: str = "", **kwargs: Any) -> SemAggNode:
        return SemAggNode(instruction=instruction, parents=[self], **kwargs)

    def sem_topk(self, instruction: str = "", **kwargs: Any) -> SemTopKNode:
        return SemTopKNode(instruction=instruction, parents=[self], **kwargs)

    def sem_join(
        self, right: ASTNode, instruction: str = "", **kwargs: Any
    ) -> SemJoinNode:
        return SemJoinNode(instruction=instruction, parents=[self, right], **kwargs)

    def sem_search(self, instruction: str = "", **kwargs: Any) -> SemSearchNode:
        return SemSearchNode(instruction=instruction, parents=[self], **kwargs)

    def sem_sim_join(
        self, right: ASTNode, instruction: str = "", **kwargs: Any
    ) -> SemSimJoinNode:
        return SemSimJoinNode(instruction=instruction, parents=[self, right], **kwargs)

    def sem_index(self, instruction: str = "", **kwargs: Any) -> SemIndexNode:
        return SemIndexNode(instruction=instruction, parents=[self], **kwargs)

    def sem_cluster_by(
        self, instruction: str = "", **kwargs: Any
    ) -> SemClusterByNode:
        return SemClusterByNode(instruction=instruction, parents=[self], **kwargs)

    def sem_dedup(self, instruction: str = "", **kwargs: Any) -> SemDedupNode:
        return SemDedupNode(instruction=instruction, parents=[self], **kwargs)

    def sem_partition_by(
        self, instruction: str = "", **kwargs: Any
    ) -> SemPartitionByNode:
        return SemPartitionByNode(instruction=instruction, parents=[self], **kwargs)

    def __repr__(self) -> str:
        return self._label()


# ------------------------------------------------------------------
# Concrete node types
# ------------------------------------------------------------------


class SourceNode(ASTNode):
    op_type = "source"

    def __init__(self, name: str = "", **kwargs: Any) -> None:
        super().__init__(instruction=name, parents=[], **kwargs)


class SemFilterNode(ASTNode):
    op_type = "sem_filter"


class SemMapNode(ASTNode):
    op_type = "sem_map"


class SemExtractNode(ASTNode):
    op_type = "sem_extract"


class SemAggNode(ASTNode):
    op_type = "sem_agg"


class SemTopKNode(ASTNode):
    op_type = "sem_topk"


class SemJoinNode(ASTNode):
    op_type = "sem_join"


class SemSearchNode(ASTNode):
    op_type = "sem_search"


class SemSimJoinNode(ASTNode):
    op_type = "sem_sim_join"


class SemIndexNode(ASTNode):
    op_type = "sem_index"


class SemClusterByNode(ASTNode):
    op_type = "sem_cluster_by"


class SemDedupNode(ASTNode):
    op_type = "sem_dedup"


class SemPartitionByNode(ASTNode):
    op_type = "sem_partition_by"


class PandasFilterNode(ASTNode):
    op_type = "filter"

    def __init__(self, parents: list[ASTNode] | None = None, **kwargs: Any) -> None:
        super().__init__(instruction="filter(predicate)", parents=parents, **kwargs)


# ------------------------------------------------------------------
# LazyFrame — deferred execution wrapper
# ------------------------------------------------------------------


class LazyFrame:
    """Records a pipeline of LOTUS semantic operators as an AST and executes them lazily.

    Usage::

        lf = LazyFrame(df, name="courses_df")
        lf = lf.sem_filter("{Course Name} requires math")
        lf = lf.sem_map("Summarize {Course Name}")
        lf.print_tree()
        result = lf.execute()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = "source",
        *,
        _node: ASTNode | None = None,
        _ops: list[tuple[str, str | None, dict[str, Any]]] | None = None,
    ) -> None:
        self._source_df = df
        self._node = _node or SourceNode(name)
        self._ops: list[tuple[str, str | None, dict[str, Any]]] = list(_ops) if _ops else []

    # Helper to derive a new LazyFrame with an extra op
    def _chain(
        self,
        op_name: str,
        instruction: str | None,
        node: ASTNode,
        kwargs: dict[str, Any],
    ) -> LazyFrame:
        new_ops = list(self._ops)
        new_ops.append((op_name, instruction, kwargs))
        return LazyFrame(self._source_df, _node=node, _ops=new_ops)

    # ------------------------------------------------------------------
    # Sem_* methods
    # ------------------------------------------------------------------

    def sem_filter(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemFilterNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_filter", instruction, node, kwargs)

    def sem_map(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemMapNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_map", instruction, node, kwargs)

    def sem_agg(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemAggNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_agg", instruction, node, kwargs)

    def sem_extract(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemExtractNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_extract", instruction, node, kwargs)

    def sem_topk(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemTopKNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_topk", instruction, node, kwargs)

    def sem_join(self, right: LazyFrame | pd.DataFrame, instruction: str = "", **kwargs: Any) -> LazyFrame:
        if isinstance(right, LazyFrame):
            right_node = right._node
            kwargs["_right_lazy"] = right
        else:
            right_node = SourceNode("right_df")
            kwargs["_right_df"] = right
        node = SemJoinNode(instruction=instruction, parents=[self._node, right_node])
        return self._chain("sem_join", instruction, node, kwargs)

    def sem_search(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemSearchNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_search", instruction, node, kwargs)

    def sem_sim_join(self, right: LazyFrame | pd.DataFrame, instruction: str = "", **kwargs: Any) -> LazyFrame:
        if isinstance(right, LazyFrame):
            right_node = right._node
            kwargs["_right_lazy"] = right
        else:
            right_node = SourceNode("right_df")
            kwargs["_right_df"] = right
        node = SemSimJoinNode(instruction=instruction, parents=[self._node, right_node])
        return self._chain("sem_sim_join", instruction, node, kwargs)

    def sem_index(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemIndexNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_index", instruction, node, kwargs)

    def sem_cluster_by(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemClusterByNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_cluster_by", instruction, node, kwargs)

    def sem_dedup(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemDedupNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_dedup", instruction, node, kwargs)

    def sem_partition_by(self, instruction: str = "", **kwargs: Any) -> LazyFrame:
        node = SemPartitionByNode(instruction=instruction, parents=[self._node])
        return self._chain("sem_partition_by", instruction, node, kwargs)

    # ------------------------------------------------------------------
    # Pandas filter
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> LazyFrame:
        node = PandasFilterNode(parents=[self._node])
        new_ops = list(self._ops)
        new_ops.append(("filter", None, {"predicate": predicate}))
        return LazyFrame(self._source_df, _node=node, _ops=new_ops)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self) -> pd.DataFrame:
        """Replay all recorded operations and return the materialised DataFrame."""
        df = self._source_df.copy()
        for op_name, instruction, kwargs in self._ops:
            if op_name == "filter":
                df = df[kwargs["predicate"](df)]
            elif op_name in ("sem_join", "sem_sim_join"):
                kw = dict(kwargs)
                right_lazy: LazyFrame | None = kw.pop("_right_lazy", None)
                right_df: pd.DataFrame | None = kw.pop("_right_df", None)
                if right_lazy is not None:
                    right_df = right_lazy.execute()
                df = getattr(df, op_name)(right_df, instruction, **kw)
            else:
                df = getattr(df, op_name)(instruction, **kwargs)
        return df

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def print_tree(self) -> None:
        self._node.print_tree()

    def print_lineage(self) -> None:
        print_lineage(self._node)

    def __repr__(self) -> str:
        steps = []
        for op_name, instruction, _kwargs in self._ops:
            if instruction:
                steps.append(f'.{op_name}("{instruction}")')
            else:
                steps.append(f".{op_name}(...)")
        root = self._node._get_roots()[0]._label() if self._node._get_roots() else "LazyFrame"
        return f"LazyFrame({root})" + "".join(steps)


# ------------------------------------------------------------------
# Lineage report
# ------------------------------------------------------------------


def _collect_all_nodes(node: ASTNode) -> list[ASTNode]:
    """Collect all nodes in the tree reachable from *node* (via parents and children)."""
    visited: set[int] = set()
    result: list[ASTNode] = []
    queue: deque[ASTNode] = deque([node])
    visited.add(id(node))
    while queue:
        cur = queue.popleft()
        result.append(cur)
        for neighbour in cur.parents + cur.children:
            if id(neighbour) not in visited:
                visited.add(id(neighbour))
                queue.append(neighbour)
    # Sort: roots first, then by insertion order (topological-ish)
    roots = [n for n in result if not n.parents]
    non_roots = [n for n in result if n.parents]
    return roots + non_roots


def print_lineage(node: ASTNode) -> None:
    """Print ancestors and descendants for every node in the tree containing *node*."""
    all_nodes = _collect_all_nodes(node)
    print("=== Lineage Report ===")
    for n in all_nodes:
        print()
        print(f"Node: {n._label()}")
        ancestors = n.get_ancestors()
        anc_str = ", ".join(type(a).__name__ for a in ancestors) if ancestors else "(none)"
        print(f"  Ancestors: {anc_str}")
        descendants = n.get_descendants()
        desc_str = ", ".join(type(d).__name__ for d in descendants) if descendants else "(none)"
        print(f"  Descendants: {desc_str}")
