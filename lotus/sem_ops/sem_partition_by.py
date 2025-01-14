from typing import Any, Callable

import pandas as pd

from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_partition_by")
class SemPartitionByDataframe:
    """DataFrame accessor for semantic partitioning."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        partition_fn: Callable[[pd.DataFrame], list[int]],
    ) -> pd.DataFrame:
        """
        Perform semantic partitioning on the DataFrame.

        Args:
            partition_fn (Callable): The partitioning function.

        Returns:
            pd.DataFrame: The DataFrame with the partition assignments.
        """
        group_ids = partition_fn(self._obj)
        self._obj["_lotus_partition_id"] = pd.Series(group_ids, index=self._obj.index)
        return self._obj
