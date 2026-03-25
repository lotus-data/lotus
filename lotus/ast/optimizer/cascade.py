"""Cache cascade thresholds optimizer for LOTUS LazyFrames.

Runs the LazyFrame on training data so that sem_filter/sem_join nodes with
cascade_args learn and cache their thresholds in-place.  Subsequent
executions skip the expensive threshold-learning phase automatically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ..nodes import BaseNode
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame

logger = logging.getLogger(__name__)


class CascadeOptimizer(BaseOptimizer):
    """Optimizer that pre-warms cascade thresholds on training data.

    Runs the LazyFrame pipeline once on training data.  ``SemFilterNode``
    and ``SemJoinNode`` nodes that have ``cascade_args`` but no cached
    thresholds automatically learn and store them in ``self.cascade_args``
    during ``__call__``.  Future executions then reuse the cached thresholds,
    skipping the threshold-learning sample.

    This works recursively — nested LazyFrames (e.g. the right side of a
    sem_join) are resolved by the standard runner and their nodes
    self-update in the same way.

    Requires ``train_data`` — a single DataFrame or a dict mapping
    LazyFrames to DataFrames.

    Example::

        from lotus.ast.optimizer import CascadeOptimizer

        optimizer = CascadeOptimizer()
        optimized_lf = lf.optimize([optimizer], train_data=df)
        # Subsequent executions reuse the cached thresholds
        result = optimized_lf.execute(df)
    """

    requires_train_data: bool = True

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None = None,
    ) -> list[BaseNode]:
        """Run the pipeline on train_data so cascade nodes learn and cache thresholds."""
        from ..lazyframe import LazyFrame
        from ..run import LazyFrameRun

        if train_data is None:
            raise ValueError(
                "CascadeOptimizer requires train_data. " "Pass it via lf.optimize([optimizer], train_data=...)."
            )

        tmp_lf = LazyFrame(_nodes=nodes)
        try:
            LazyFrameRun(tmp_lf, train_data, node_runtime_configs={"update_cascade_args": True}).execute()
        except Exception as e:
            logger.warning(
                "CascadeOptimizer: execution failed (%s), returning nodes unchanged.",
                e,
            )

        # Nodes have updated their own cascade_args in-place during execution.
        return nodes
