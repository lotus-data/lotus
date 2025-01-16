from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from lotus.types import RMOutput


class RM(ABC):
    """Abstract class for retriever models."""

    def __init__(self) -> None:
        self.index_dir: str | None = None

    @abstractmethod
    def index(self, docs: pd.Series, index_dir: str, **kwargs: dict[str, Any]) -> None:
        """Create index and store it to a directory.

        Args:
            docs (list[str]): A list of documents to index.
            index_dir (str): The directory to save the index in.
        """
        pass

    @abstractmethod
    def load_index(self, index_dir: str) -> None:
        """Load the index into memory.

        Args:
            index_dir (str): The directory of where the index is stored.
        """
        pass

    @abstractmethod
    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """Get the vectors from the index.

        Args:
            index_dir (str): Directory of the index.
            ids (list[int]): The ids of the vectors to retrieve

        Returns:
            NDArray[np.float64]: The vectors matching the specified ids.
        """

        pass

    @abstractmethod
    def __call__(
        self,
        queries: pd.Series | str | Image.Image | list | NDArray[np.float64],
        K: int,
        **kwargs: dict[str, Any],
    ) -> RMOutput:
        """Run top-k search on the index.

        Args:
            queries (str | list[str] | NDArray[np.float64]): Either a query or a list of queries or a 2D FP32 array.
            K (int): The k to use for top-k search.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            RMOutput: An RMOutput object containing the distances and indices of the top-k vectors.
        """
        pass
    
    def __update_attrs__(self, queries: pd.Series | str | Image.Image | list | NDArray[np.float64], index_dir: str) -> None:
        """Default implementation that does nothing"""
        pass
