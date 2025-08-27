from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.models import RM
from lotus.types import RMOutput
from lotus.vector_store import VS


@pd.api.extensions.register_dataframe_accessor("sem_sim_join")
class SemSimJoinDataframe:
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

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import RM, VS
        >>> from lotus.vector_store import FaissVS

        >>> lm = LM(model="gpt-4o-mini")
        >>> rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        >>> vs = FaissVS()

        >>> lotus.settings.configure(lm=lm, rm=rm, vs=vs)

        >>> df1 = pd.DataFrame({
                'article': ['Machine learning tutorial', 'Data science guide', 'Python basics', 'AI in finance', 'Cooking healthy food', "Recipes for the holidays"],
            })
        >>> df2 = pd.DataFrame({
                'category': ['Computer Science', 'AI', 'Cooking'],
            })

        >>> df1.sem_index("article", "article_index")
        >>> df2.sem_index("category", "category_index")

        Example 1: sem_sim_join, K=1, join each article with the most similar category
        >>> df1.sem_sim_join(df2, "article", "category", K=1)
                            article   _scores           category
        0  Machine learning tutorial  0.834617  Computer Science
        1         Data science guide  0.820131  Computer Science
        2              Python basics  0.834945  Computer Science
        3              AI in finance  0.875249                AI
        4       Cooking healthy food  0.890393           Cooking
        5   Recipes for the holidays  0.786058           Cooking

        Example 2: sem_sim_join, K=2, join each article with the top 2 most similar categories
        >>> df1.sem_sim_join(df2, "article", "category", K=2)
                                article   _scores          category
        0  Machine learning tutorial  0.834617  Computer Science
        0  Machine learning tutorial  0.817893                AI
        1         Data science guide  0.820131  Computer Science
        1         Data science guide  0.785335                AI
        2              Python basics  0.834945  Computer Science
        2              Python basics  0.770674                AI
        3              AI in finance  0.875249                AI
        3              AI in finance  0.798493  Computer Science
        4       Cooking healthy food  0.890393           Cooking
        4       Cooking healthy food  0.755058  Computer Science
        5   Recipes for the holidays  0.786058           Cooking
        5   Recipes for the holidays  0.712726  Computer Science
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
        other: pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
    ) -> pd.DataFrame:
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
