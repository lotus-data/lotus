from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_index")
class SemIndexDataframe:
    """
    Create a vecgtor similarity index over a column in the DataFrame. Indexing is required for columns used in sem_search, sem_cluster_by, and sem_sim_join.
    When using retrieval-based cascades for sem_filter and sem_join, indexing is required for the columns used in the semantic operation.

    Args:
        col_name (str): The column name to index.
        index_dir (str): The directory to save the index.

    Returns:
        pd.DataFrame: The DataFrame with the index directory saved.

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

            # Example 1: create a new index using sem_index
            >>> df.sem_index('title', 'title_index') ## only needs to be run once; sem_index will save the index to the current directory in "title_index";
            >>> df.sem_search('title', 'AI', K=2)
            100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.88it/s]
                                    title
            0  Machine learning tutorial
            1         Data science guide

            # Example 2: load an existing index using load_sem_index
            >>> df.load_sem_index('title', 'title_index') ## index has already been created
            >>> df.sem_search('title', 'AI', K=2)
            100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.88it/s]
                                    title
            0  Machine learning tutorial
            1         Data science guide
    """

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs["index_dirs"] = {}

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(self, col_name: str, index_dir: str) -> pd.DataFrame:
        lotus.logger.warning(
            "Do not reset the dataframe index to ensure proper functionality of get_vectors_from_index"
        )

        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model using lotus.settings.configure()"
            )

        embeddings = rm(self._obj[col_name].tolist())
        vs.index(self._obj[col_name], embeddings, index_dir)
        self._obj.attrs["index_dirs"][col_name] = index_dir
        return self._obj
