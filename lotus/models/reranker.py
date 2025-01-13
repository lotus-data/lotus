from abc import ABC, abstractmethod

from lotus.types import RerankerOutput


class Reranker(ABC):
    """Abstract class for reranker models."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, query: str, docs: list[str], K: int) -> RerankerOutput:
        """Invoke the reranker.

        Args:
            query (str): The query to use for reranking.
            docs (list[str]): A list of documents to rerank.
            K (int): The number of documents to keep after reranking.

        Returns:
            RerankerOutput: The indices of the reranked documents.
        """
        pass
