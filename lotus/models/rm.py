from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image


class RM(ABC):
    # Abstract class for retriever models.

    def __init__(self) -> None:
        pass

    @abstractmethod
    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        pass

    def __call__(self, docs: list[str]) -> NDArray[np.float64]:
        return self._embed(docs)

    def convert_query_to_query_vector(
        self,
        queries: Union[pd.Series, str, Image.Image, list, NDArray[np.float64]],
    ):
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
