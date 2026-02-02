"""Optimizer for LOTUS pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lotus

from .nodes import BaseNode, PandasFilterNode, SemFilterNode, SourceNode

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Optimizer:
    """Optimizer that transforms pipelines for better performance.

    Currently implements:
    - Predicate pushdown: moves pandas filters before sem_filters where safe
    """

    def optimize(self, pipeline: "Pipeline", *, inplace: bool = False) -> "Pipeline":
        """Apply optimizations to a pipeline.

        Args:
            pipeline: The pipeline to optimize
            inplace: If True, modify the pipeline in place. If False, return a new pipeline.

        Returns:
            The optimized pipeline (same object if inplace=True, new object otherwise)
        """
        from .pipeline import Pipeline

        lotus.logger.debug(
            f"Optimizer.optimize: optimizing pipeline with {len(pipeline._nodes)} nodes, inplace={inplace}"
        )
        if inplace:
            pipeline._nodes = self._optimize_nodes(pipeline._nodes)
            lotus.logger.debug("Optimizer.optimize: in-place optimization complete")
            return pipeline
        else:
            optimized_nodes = self._optimize_nodes(list(pipeline._nodes))
            lotus.logger.debug("Optimizer.optimize: created new optimized pipeline")
            return Pipeline(_nodes=optimized_nodes, _source=pipeline._source)

    def _optimize_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Apply all optimizations to a node list."""
        lotus.logger.debug("Optimizer._optimize_nodes: applying predicate pushdown")
        nodes = self._predicate_pushdown(nodes)
        return nodes

    def _predicate_pushdown(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Move pandas filter nodes before sem_filter nodes where safe.

        This optimization reduces the number of rows processed by expensive
        semantic operations by filtering first with cheap pandas predicates.

        A pandas filter can be pushed past a sem_filter because sem_filter
        only removes rows - it doesn't add or rename columns that the filter
        might depend on.
        """
        nodes = list(nodes)  # Work on a copy
        pushes = 0

        for i in range(len(nodes)):
            if isinstance(nodes[i], PandasFilterNode):
                # Bubble this filter backwards past consecutive sem_filters
                j = i
                while j > 0 and isinstance(nodes[j - 1], SemFilterNode):
                    nodes[j], nodes[j - 1] = nodes[j - 1], nodes[j]
                    j -= 1
                    pushes += 1

        if pushes > 0:
            lotus.logger.debug(f"Optimizer._predicate_pushdown: pushed {pushes} filter(s) earlier in pipeline")
        else:
            lotus.logger.debug("Optimizer._predicate_pushdown: no optimizations applied")

        return nodes

    def _can_push_past(self, filter_node: PandasFilterNode, other_node: BaseNode) -> bool:
        """Check if a filter can be pushed past another node.

        Currently only allows pushing past SemFilterNode.
        Could be extended to handle more cases.
        """
        if isinstance(other_node, SemFilterNode):
            return True
        # Don't push past source nodes
        if isinstance(other_node, SourceNode):
            return False
        # Don't push past nodes that might add/rename columns
        return False
