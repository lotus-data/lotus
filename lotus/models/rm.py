from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image


class RM(ABC):
    """
    Abstract base class for retrieval models.

    This class defines the interface for retrieval models that can generate
    embeddings for documents and queries. Subclasses must implement the
    `_embed` method to provide the actual embedding functionality.

    Attributes:
        None (abstract base class)
    """

    def __init__(self) -> None:
        """Initialize the retrieval model base class."""
        pass

    @abstractmethod
    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        """
        Generate embeddings for a list of documents.

        This is an abstract method that must be implemented by subclasses.

        Args:
            docs: List of document strings to embed.

        Returns:
            NDArray[np.float64]: Array of embeddings with shape (num_docs, embedding_dim).
        """
        pass

    def __call__(self, docs: list[str]) -> NDArray[np.float64]:
        """
        Generate embeddings for documents by calling the `_embed` method.

        Args:
            docs: List of document strings to embed.

        Returns:
            NDArray[np.float64]: Array of embeddings with shape (num_docs, embedding_dim).
        """
        return self._embed(docs)

    def convert_query_to_query_vector(
        self,
        queries: Union[pd.Series, str, Image.Image, list[str], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Convert various query formats to query vectors.

        This method handles different input types and converts them to embedding vectors:
        - String queries: Converted to list and embedded
        - Image queries: Converted to list and embedded (if supported)
        - Pandas Series: Converted to list and embedded
        - List of strings: Directly embedded
        - Numpy arrays: Returned as-is (assumed to be pre-computed vectors)

        Args:
            queries: Query or queries in various formats.

        Returns:
            NDArray[np.float64]: Array of query vectors with shape (num_queries, embedding_dim).
        """
        if isinstance(queries, (str, Image.Image)):
            queries = [queries]

        # Handle numpy array queries (pre-computed vectors)
        if isinstance(queries, np.ndarray):
            query_vectors = queries
        else:
            # Convert queries to list if needed
            if isinstance(queries, pd.Series):
                queries = queries.tolist()
            # Create embeddings for text queries
            query_vectors = self._embed(queries)
        return query_vectors
