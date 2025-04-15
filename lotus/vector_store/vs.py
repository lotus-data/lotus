from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lotus.types import RMOutput


class VS(ABC):
    """Abstract class for vector stores."""

    def __init__(self) -> None:
        self.index_dir: str | None = None

    @abstractmethod
    def index(self, docs: list[str], embeddings: NDArray[np.float64], index_dir: str, **kwargs: dict[str, Any]):
        """
        Create index and store it in vector store.
        """
        pass

    @abstractmethod
    def load_index(self, index_dir: str):
        """
        Load the index from the vector store into memory if needed.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        query_vectors: NDArray[np.float64],
        K: int,
        ids: list[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> RMOutput:
        """
        Perform a nearest neighbor search given query vectors.

        Args:
            query_vectors (Any): The query vector(s) used for the search.
            K (int): The number of nearest neighbors to retrieve.
            ids (Optional[list[Any]]): The list of document ids (or index positions) to search over.
                                       If None, search across all indexed vectors.
            **kwargs (dict[str, Any]): Additional parameters.

        Returns:
            RMOutput: The output containing distances and indices.
        """
        pass

    @abstractmethod
    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """
        Retrieve vectors from a stored index given specific ids.
        """
        pass
