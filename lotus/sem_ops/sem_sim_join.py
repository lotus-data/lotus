from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.models import RM
from lotus.types import RMOutput
from lotus.vector_store import VS


@pd.api.extensions.register_dataframe_accessor("sem_sim_join")
class SemSimJoinDataframe:
    """DataFrame accessor for semantic similarity join."""

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
        other: pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
    ) -> pd.DataFrame:
        """
        Perform semantic similarity join on the DataFrame.

        Args:
            other (pd.DataFrame): The other DataFrame to join with.
            left_on (str): The column name to join on in the left DataFrame.
            right_on (str): The column name to join on in the right DataFrame.
            K (int): The number of nearest neighbors to search for.
            lsuffix (str): The suffix to append to the left DataFrame.
            rsuffix (str): The suffix to append to the right DataFrame.
            score_suffix (str): The suffix to append to the similarity score column.
        """

        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if not isinstance(rm, RM) or not isinstance(vs, VS):
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model or vector store using lotus.settings.configure()"
            )

        # load query embeddings from index if they exist
        if left_on in self._obj.attrs.get("index_dirs", []):
            query_index_dir = self._obj.attrs["index_dirs"][left_on]
            if vs.index_dir != query_index_dir:
                vs.load_index(query_index_dir)
            assert vs.index_dir == query_index_dir
            try:
                queries = vs.get_vectors_from_index(query_index_dir, self._obj.index)
            except NotImplementedError:
                queries = self._obj[left_on]
        else:
            queries = self._obj[left_on]

        # load index to search over
        try:
            col_index_dir = other.attrs["index_dirs"][right_on]
        except KeyError:
            raise ValueError(f"Index directory for column {right_on} not found in DataFrame")
        if vs.index_dir != col_index_dir:
            vs.load_index(col_index_dir)
        assert vs.index_dir == col_index_dir

        query_vectors = rm.convert_query_to_query_vector(queries)

        right_ids = list(other.index)

        vs_output: RMOutput = vs(query_vectors, K, ids=right_ids)
        distances = vs_output.distances
        indices = vs_output.indices

        other_index_set = set(other.index)
        join_results = []

        # post filter
        for q_idx, res_ids in enumerate(indices):
            for i, res_id in enumerate(res_ids):
                if res_id != -1 and res_id in other_index_set:
                    join_results.append((self._obj.index[q_idx], res_id, distances[q_idx][i]))

        df1 = self._obj.copy()
        df2 = other.copy()
        df1["_left_id"] = df1.index
        df2["_right_id"] = df2.index
        temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id", "_scores" + score_suffix])
        joined_df = df1.join(
            temp_df.set_index("_left_id"),
            how="right",
            on="_left_id",
        ).join(
            df2.set_index("_right_id"),
            how="left",
            on="_right_id",
            lsuffix=lsuffix,
            rsuffix=rsuffix,
        )
        if not keep_index:
            joined_df.drop(columns=["_left_id", "_right_id"], inplace=True)

        return joined_df
