from typing import Any, Callable

import pandas as pd

from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_partition_by")
class SemPartitionByDataframe:
    """
    Perform semantic partitioning on the DataFrame.

    Args:
        partition_fn (Callable): The partitioning function.

    Returns:
        pd.DataFrame: The DataFrame with the partition assignments.
        
    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM, SentenceTransformersRM
        >>> from lotus.vector_store import FaissVS

        >>> lm = LM(model="gpt-4o-mini")
        >>> rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        >>> vs = FaissVS()

        >>> lotus.settings.configure(lm=lm, rm=rm, vs=vs)
        >>> df = pd.DataFrame({
                "Course Name": [
                    "Probability and Random Processes",
                    "Optimization Methods in Engineering",
                    "Digital Design and Integrated Circuits",
                    "Computer Security",
                    "Cooking",
                    "Food Sciences",
                ]
            })
        >>> df = df.sem_index("Course Name", "course_name_index")\
                .sem_partition_by(lotus.utils.cluster("Course Name", 2))
        >>> df.sem_agg("Summarize all {Course Name}")._output[0]
        Aggregating:   0%|                                                                         0/2 LM calls [00:00<?, ?it/s]
        Aggregating: 100%|████████████████████████████████████████████████████████████████ 2/2 LM calls [00:01<00:00,  1.90it/s]
        Aggregating: 100%|████████████████████████████████████████████████████████████████ 1/1 LM calls [00:01<00:00,  1.08s/it]
        'The courses offered include "Probability and Random Processes," "Optimization Methods in Engineering," "Digital Design 
        and Integrated Circuits," "Computer Security," "Cooking," and "Food Sciences."'

    """

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
        group_ids = partition_fn(self._obj)
        self._obj["_lotus_partition_id"] = pd.Series(group_ids, index=self._obj.index)
        return self._obj
