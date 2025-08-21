from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.types import RerankerOutput, RMOutput


@pd.api.extensions.register_dataframe_accessor("sem_search")
class SemSearchDataframe:
    """
    DataFrame accessor for semantic search operations.

    This class provides a pandas DataFrame accessor that enables semantic
    search over DataFrame content using natural language queries. It supports
    both vector-based retrieval and reranking for improved search quality.

    The accessor can be used directly on any pandas DataFrame:
        df.sem_search('text', 'machine learning applications')
    """

    def __init__(self, pandas_obj: Any):
        """
        Initialize the semantic search accessor.

        Args:
            pandas_obj (Any): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (Any): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        col_name: str,
        query: str,
        K: int | None = None,
        n_rerank: int | None = None,
        return_scores: bool = False,
        suffix: str = "_sim_score",
    ) -> pd.DataFrame:
        """
        Perform semantic search on the DataFrame.

        This method performs semantic search over the specified column using
        a natural language query. It can use vector-based retrieval for initial
        results and optional reranking for improved relevance.

        Args:
            col_name (str): The column name to search on. This column should
                contain the text content to be searched.
            query (str): The natural language query string. This should describe
                what you're looking for in the data.
            K (int | None, optional): The number of documents to retrieve using
                vector search. Must be provided if n_rerank is None. Defaults to None.
            n_rerank (int | None, optional): The number of documents to rerank
                using a cross-encoder reranker. Must be provided if K is None.
                Defaults to None.
            return_scores (bool, optional): Whether to return the similarity scores
                from the vector search. Useful for understanding result relevance.
                Defaults to False.
            suffix (str, optional): The suffix to append to the new column containing
                the similarity scores. Only used if return_scores is True.
                Defaults to "_sim_score".

        Returns:
            pd.DataFrame: A DataFrame containing the search results. The returned
                DataFrame will have fewer rows than the original, containing only
                the most relevant matches.

        Raises:
            ValueError: If neither K nor n_rerank is provided, if the retrieval
                model or vector store is not configured, or if the reranker is
                not configured when n_rerank is specified.

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
        assert not (K is None and n_rerank is None), "K or n_rerank must be provided"
        if K is not None:
            # get retriever model and index
            rm = lotus.settings.rm
            vs = lotus.settings.vs
            if rm is None or vs is None:
                raise ValueError(
                    "The retrieval model must be an instance of RM, and the vector store should be an instance of VS. Please configure a valid retrieval model and vector store using lotus.settings.configure()"
                )

            col_index_dir = self._obj.attrs["index_dirs"][col_name]
            if vs.index_dir != col_index_dir:
                vs.load_index(col_index_dir)
            assert vs.index_dir == col_index_dir

            df_idxs = self._obj.index
            cur_min = len(df_idxs)
            K = min(K, cur_min)
            search_K = K
            while True:
                query_vectors = rm.convert_query_to_query_vector(query)
                vs_output: RMOutput = vs(query_vectors, search_K)
                doc_idxs = vs_output.indices[0]
                scores = vs_output.distances[0]
                assert len(doc_idxs) == len(scores)

                postfiltered_doc_idxs = []
                postfiltered_scores = []
                for idx, score in zip(doc_idxs, scores):
                    if idx in df_idxs:
                        postfiltered_doc_idxs.append(idx)
                        postfiltered_scores.append(score)

                postfiltered_doc_idxs = postfiltered_doc_idxs[:K]
                postfiltered_scores = postfiltered_scores[:K]
                if len(postfiltered_doc_idxs) == K:
                    break
                search_K = search_K * 2

            new_df = self._obj.loc[postfiltered_doc_idxs]
            new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)

            if return_scores:
                new_df["vec_scores" + suffix] = postfiltered_scores
        else:
            new_df = self._obj

        if n_rerank is not None:
            if lotus.settings.reranker is None:
                raise ValueError("Reranker not found in settings")

            docs = new_df[col_name].tolist()
            reranked_output: RerankerOutput = lotus.settings.reranker(query, docs, n_rerank)
            reranked_idxs = reranked_output.indices
            new_df = new_df.iloc[reranked_idxs]

        return new_df
