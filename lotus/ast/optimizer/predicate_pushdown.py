"""Predicate pushdown optimizer for LOTUS LazyFrames.

Moves pandas filter nodes before sem_filter nodes where safe, reducing the
number of rows processed by expensive semantic operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

import lotus

from ..nodes import BaseNode, PandasFilterNode, SemFilterNode, SourceNode
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame


class PredicatePushdownOptimizer(BaseOptimizer):
    """Optimizer that moves pandas filters before sem_filters where safe.

    This optimization reduces the number of rows processed by expensive
    semantic operations by filtering first with cheap pandas predicates.

    A pandas filter can be pushed past a sem_filter because sem_filter
    only removes rows - it doesn't add or rename columns that the filter
    might depend on.
    """

    requires_train_data: bool = False

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None = None,
    ) -> list[BaseNode]:
        """Move pandas filter nodes before sem_filter nodes where safe.

        Args:
            nodes: List of nodes to optimize
            train_data: Optional training data (not used by this optimizer)

        Returns:
            Optimized list of nodes with filters pushed earlier
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
            lotus.logger.debug(f"PredicatePushdownOptimizer: pushed {pushes} filter(s) earlier in lazyframe")
        else:
            lotus.logger.debug("PredicatePushdownOptimizer: no optimizations applied")

        return nodes

    def _can_push_past(self, filter_node: PandasFilterNode, other_node: BaseNode) -> bool:
        """Check if a filter can be pushed past another node.

        Currently only allows pushing past SemFilterNode.
        Could be extended to handle more cases.

        Args:
            filter_node: The pandas filter node to push
            other_node: The node to push past

        Returns:
            True if the filter can be safely pushed past the other node
        """
        if isinstance(other_node, SemFilterNode):
            return True
        # Don't push past source nodes
        if isinstance(other_node, SourceNode):
            return False
        # Don't push past nodes that might add/rename columns
        return False
