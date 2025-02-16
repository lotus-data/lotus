from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from lotus.types import RMOutput

"""
MODEL_NAME_TO_CLS = {
    "intfloat/e5-small-v2": lambda model: SentenceTransformer(model_name_or_path=model),
    "mixedbread-ai/mxbai-rerank-xsmall-v1": lambda model: CrossEncoder(model_name=model),
    "text-embedding-3-small": lambda model: lambda batch: embedding(model=model, input=batch),
}


def initialize(model_name):
    if model_name == 'intfloat/e5-small-v2':
        return SentenceTransformer(model_name_or_path=model_name) 
    elif model_name== 'mixedbread-ai/mxbai-rerank-xsmall-v1':
        return CrossEncoder(model_name=model_name) 
    return lambda batch: embedding(model=model_name, input=batch) 
"""

class VS(ABC):
    """Abstract class for vector stores."""

    def __init__(self) -> None:
        self.index_dir: str | None = None 
        self.max_batch_size:int = 64

    @abstractmethod
    def index(self, docs, embeddings: Any, index_dir: str, **kwargs: dict[str, Any]):
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
        query_vectors: Any,
        K: int,
        ids: Optional[list[Any]] = None,
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
    def get_vectors_from_index(self, index_dir: str, ids: list[Any]) -> NDArray[np.float64]:
        """
        Retrieve vectors from a stored index given specific ids.
        """
        pass 
