"""Base optimizer interface for LOTUS LazyFrames."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame
    from ..nodes import BaseNode


class BaseOptimizer(ABC):
    """Base class for LazyFrame optimizers.

    Each optimizer implements a specific optimization strategy that transforms
    a list of nodes to improve performance.
    """

    requires_train_data: bool = False

    @abstractmethod
    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None = None,
    ) -> list[BaseNode]:
        """Apply optimization to a list of nodes.

        Args:
            nodes: List of nodes to optimize
            train_data: Optional training data dict (LazyFrame -> DataFrame).
                       Only provided if requires_train_data is True.

        Returns:
            Optimized list of nodes (may be the same list if no changes)
        """
        pass
