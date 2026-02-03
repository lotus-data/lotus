"""Base optimizer interface for LOTUS pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..nodes import BaseNode


class BaseOptimizer(ABC):
    """Base class for pipeline optimizers.

    Each optimizer implements a specific optimization strategy that transforms
    a list of nodes to improve performance.
    """

    @abstractmethod
    def optimize_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Apply optimization to a list of nodes.

        Args:
            nodes: List of nodes to optimize

        Returns:
            Optimized list of nodes (may be the same list if no changes)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this optimizer.

        Returns:
            Human-readable name for logging/debugging
        """
        pass
