"""Optimizer module for LOTUS LazyFrames."""

from .base import BaseOptimizer
from .cascade import CascadeOptimizer
from .gepa_optimizer import GEPAOptimizer
from .predicate_pushdown import PredicatePushdownOptimizer

DEFAULT_OPTIMIZERS: list[BaseOptimizer] = [PredicatePushdownOptimizer()]

__all__ = [
    "BaseOptimizer",
    "CascadeOptimizer",
    "DEFAULT_OPTIMIZERS",
    "GEPAOptimizer",
    "PredicatePushdownOptimizer",
]
