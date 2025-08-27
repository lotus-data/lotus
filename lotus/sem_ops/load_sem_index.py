from typing import Any

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("load_sem_index")
class LoadSemIndexDataframe:
    """
    Load a semantic index for a column in the DataFrame.

    Args:
        col_name (str): The column name to load the index for.
        index_dir (str): The directory to load the index from.

    Returns:
        pd.DataFrame: The DataFrame with the index loaded.

        Example:
            >>> import pandas as pd
            >>> import lotus
            >>> from lotus.models import LM, SentenceTransformersRM
            >>> from lotus.vector_store import FaissVS
            >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"), rm=SentenceTransformersRM(model="intfloat/e5-base-v2"), vs=FaissVS())

            >>> df = pd.DataFrame({
            ...     'title': ['Machine learning tutorial', 'Data science guide', 'Python basics'],
            ...     'category': ['ML', 'DS', 'Programming']
            ... })

            >>> df.sem_index('title', 'title_index') ## only needs to be run once; sem_index will save the index to the current directory in "title_index"
            >>> df.load_sem_index('title', 'title_index') ## load the index from disk
            >>> df.sem_search('title', 'AI', K=2)
            100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.88it/s]
                                    title
            0  Machine learning tutorial
            1         Data science guide
    """

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs["index_dirs"] = {}

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(self, col_name: str, index_dir: str) -> pd.DataFrame:
        self._obj.attrs["index_dirs"][col_name] = index_dir
        return self._obj
