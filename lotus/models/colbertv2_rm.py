import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lotus.types import RMOutput

try:
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
except ImportError:
    pass


class ColBERTv2RM:
    """
    A retrieval model based on ColBERTv2 for dense passage retrieval.

    This class provides functionality to index documents and perform semantic search
    using the ColBERTv2 model. It supports both indexing and searching operations
    with configurable parameters.

    Attributes:
        docs (list[str] | None): The list of documents that have been indexed.
        kwargs (dict[str, Any]): Default configuration parameters for indexing.
        index_dir (str | None): Directory path where the index is stored.
    """

    def __init__(self) -> None:
        """
        Initialize the ColBERTv2RM retrieval model.

        Sets up default configuration parameters for document indexing:
        - doc_maxlen: Maximum document length (default: 300)
        - nbits: Number of bits for quantization (default: 2)
        """
        self.docs: list[str] | None = None
        self.kwargs: dict[str, Any] = {"doc_maxlen": 300, "nbits": 2}
        self.index_dir: str | None = None

    def index(self, docs: list[str], index_dir: str, **kwargs: dict[str, Any]) -> None:
        """
        Index a collection of documents using ColBERTv2.

        This method creates a searchable index from the provided documents.
        The index is stored in the specified directory and can be used for
        subsequent search operations.

        Args:
            docs: List of document strings to be indexed.
            index_dir: Directory path where the index will be stored.
            **kwargs: Additional configuration parameters that override defaults.
                     Supported parameters include:
                     - doc_maxlen: Maximum document length
                     - nbits: Number of bits for quantization
                     - kmeans_niters: Number of k-means iterations (default: 4)

        Raises:
            ImportError: If ColBERT dependencies are not available.
        """
        kwargs = {**self.kwargs, **kwargs}
        checkpoint = "colbert-ir/colbertv2.0"

        with Run().context(RunConfig(nranks=1, experiment="lotus")):
            config = ColBERTConfig(doc_maxlen=kwargs["doc_maxlen"], nbits=kwargs["nbits"], kmeans_niters=4)
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=f"{index_dir}/index", collection=docs, overwrite=True)

        with open(f"experiments/lotus/indexes/{index_dir}/index/docs", "wb") as fp:
            pickle.dump(docs, fp)

        self.docs = docs
        self.index_dir = index_dir

    def load_index(self, index_dir: str) -> None:
        """
        Load an existing index from disk.

        This method loads a previously created index and its associated documents
        into memory for searching operations.

        Args:
            index_dir: Directory path where the index is stored.

        Raises:
            FileNotFoundError: If the index directory or documents file doesn't exist.
            pickle.UnpicklingError: If the documents file is corrupted.
        """
        self.index_dir = index_dir
        with open(f"experiments/lotus/indexes/{index_dir}/index/docs", "rb") as fp:
            self.docs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """
        Extract document vectors from the index for specified document IDs.

        This method is not implemented for ColBERTv2RM as the underlying
        ColBERT implementation doesn't provide direct access to document vectors.

        Args:
            index_dir: Directory path where the index is stored.
            ids: List of document IDs to extract vectors for.

        Raises:
            NotImplementedError: This method is not supported in ColBERTv2RM.
        """
        raise NotImplementedError("This method is not implemented for ColBERTv2RM")

    def __call__(
        self,
        queries: str | Image.Image | list[str] | NDArray[np.float64],
        K: int,
        **kwargs: dict[str, Any],
    ) -> RMOutput:
        """
        Perform semantic search using the indexed documents.

        This method searches for the most similar documents to the given queries
        and returns ranked results with distances and indices.

        Args:
            queries: Query or list of queries to search for. Can be:
                    - A single string query
                    - A list of string queries
                    - An image (not supported in current implementation)
                    - A numpy array of query vectors (not supported in current implementation)
            K: Number of top results to return for each query.
            **kwargs: Additional search parameters (currently unused).

        Returns:
            RMOutput: Object containing search results with:
                     - distances: List of distance scores for each query
                     - indices: List of document indices for each query

        Raises:
            ValueError: If no index has been loaded or created.
            ImportError: If ColBERT dependencies are not available.
            AssertionError: If queries is not a string or list of strings.
        """
        if isinstance(queries, str):
            queries = [queries]

        with Run().context(RunConfig(experiment="lotus")):
            searcher = Searcher(index=f"{self.index_dir}/index", collection=self.docs)

        # make queries a dict with keys as query ids
        assert isinstance(queries, list)
        queries_dict = {i: q for i, q in enumerate(queries)}
        all_results = searcher.search_all(queries_dict, k=K).todict()

        indices = [[result[0] for result in all_results[qid]] for qid in all_results.keys()]
        distances = [[result[2] for result in all_results[qid]] for qid in all_results.keys()]

        return RMOutput(distances=distances, indices=indices)
