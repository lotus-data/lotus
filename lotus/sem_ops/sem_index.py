import hashlib
from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_index")
class SemIndexDataframe:
    """
    Create a vector similarity index over a column in the DataFrame. Indexing is required for columns used in sem_search, sem_cluster_by, and sem_sim_join.
    When using retrieval-based cascades for sem_filter and sem_join, indexing is required for the columns used in the semantic operation.

    Args:
        col_name (str): The column name to index.
        index_dir (str): The directory to save the index. Required to prevent column name collisions.
        override (bool): If True, recreate index even if it exists and data is consistent. Defaults to False.

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

            # Example 3: force recreation of index with override=True
            >>> df.sem_index('title', 'title_index', override=True) ## recreates index even if it exists
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
    def __call__(self, col_name: str, index_dir: str, override: bool = False) -> pd.DataFrame:
        """
        Create or load a semantic index for the specified column.

        Args:
            col_name: Name of the column to index
            index_dir: Directory where the index should be stored/loaded from
            override: If True, recreate index even if it exists and data is consistent

        Returns:
            DataFrame with index directory stored in attrs
        """
        lotus.logger.warning(
            "Do not reset the dataframe index to ensure proper functionality of get_vectors_from_index"
        )

        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model using lotus.settings.configure()"
            )

        # Get data from column
        data = self._obj[col_name].tolist()
        model_name = getattr(rm, "model", None)

        # Check if index exists and data is consistent.
        index_exists = vs.index_exists(index_dir)
        data_consistent = False

        if index_exists:
            data_consistent = vs.is_data_consistent(index_dir, data, model_name)

        # Determine if we need to create a new index.
        should_create_index = not index_exists or not data_consistent or override

        if should_create_index:
            # Index does not exist, data is inconsistent, or override requested. Creating new index.
            if index_exists and not data_consistent and not override:
                raise ValueError(
                    f"Index exists at {index_dir} but data is inconsistent. "
                    f"Set override=True to recreate the index or use a different index_dir."
                )

            # Create data hash for consistency checking.
            content = str(sorted(data))
            if model_name:
                content += f"__{model_name}"
            data_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

            # Create new index.
            embeddings = rm(data)
            vs.index(self._obj[col_name], embeddings, index_dir)

            # Store metadata for data consistency checking (FAISS-specific)
            if hasattr(vs, "_store_metadata"):
                vs._store_metadata(index_dir, data_hash)
            lotus.logger.info(f"Created new index at {index_dir}")
        else:
            # Load existing index.
            vs.load_index(index_dir)
            lotus.logger.info(f"Loaded existing index from {index_dir}")

        self._obj.attrs["index_dirs"][col_name] = index_dir
        return self._obj
