import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
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

    def index_exists(self, index_dir: str) -> bool:
        """
        Check if an index exists at the given directory.
        Default implementation checks for common index files.
        Subclasses can override for vector store specific checks.
        """
        index_path = Path(index_dir)
        return index_path.exists() and (index_path / "index").exists()

    def is_data_consistent(self, index_dir: str, data: list, model_name: str | None = None) -> bool:
        """
        Check if the cached index is consistent with the current data.
        Default implementation compares data hash stored in metadata.
        Subclasses can override for more sophisticated consistency checks.
        """
        metadata_path = Path(index_dir) / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Create hash of current data
            content = str(sorted(data))
            if model_name:
                content += f"__{model_name}"
            current_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

            return metadata.get("data_hash") == current_hash
        except (json.JSONDecodeError, KeyError):
            return False
