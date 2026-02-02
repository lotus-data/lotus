"""AST and pipeline utilities for LOTUS."""

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
from .optimizer import Optimizer
from .pipeline import Pipeline
from .run import Run

# Alias for convenience
LazyFrame = Pipeline

__all__ = [
    # Base
    "BaseNode",
    # Source
    "SourceNode",
    # Function-style operations
    "ApplyFnNode",
    # Semantic operators
    "SemFilterNode",
    "SemMapNode",
    "SemExtractNode",
    "SemAggNode",
    "SemTopKNode",
    "SemJoinNode",
    "SemSimJoinNode",
    "SemSearchNode",
    "SemIndexNode",
    "SemClusterByNode",
    "SemDedupNode",
    "SemPartitionByNode",
    # Pandas operators
    "PandasAssignNode",
    "PandasFilterNode",
    "PandasOpNode",
    # Pipeline
    "Pipeline",
    "LazyFrame",  # Alias for Pipeline
    # Execution
    "Run",
    # Optimization
    "Optimizer",
]
