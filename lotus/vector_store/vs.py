from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from litellm import embedding
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import CrossEncoder, SentenceTransformer

from lotus.dtype_extensions import convert_to_base_data
from lotus.types import RMOutput

MODEL_NAME_TO_CLS = {
    "intfloat/e5-small-v2": lambda model: SentenceTransformer(model_name_or_path=model),
    "mixedbread-ai/mxbai-rerank-xsmall-v1": lambda model: CrossEncoder(model_name=model),
    "text-embedding-3-small": lambda model: lambda batch: embedding(model=model, input=batch),
}


def initialize(model_name):
    if model_name == 'intfloat/e5-small-v2':
        return SentenceTransformer(model_name=model_name) 
    elif model_name== 'mixedbread-ai/mxbai-rerank-xsmall-v1':
        return CrossEncoder(model_name=model_name) 
    return lambda batch: embedding(model=model_name, input=batch) 


class VS(ABC):
    """Abstract class for vector stores."""

    def __init__(self, embedding_model: str) -> None:
        self.collection_name: str | None = None 
        self._embed = initialize(embedding_model) 
        self.max_batch_size:int = 64
        

    @abstractmethod
    def index(self, docs: pd.Series, collection_name: str):
        """
        Create index and store it in vector store 
        """
        pass

    @abstractmethod
    def load_index(self, collection_name: str):
        """Load the index from the vector store into memory ?? (not sure if this is needed )"""

    @abstractmethod 
    def __call__(self,
    queries: pd.Series | str | Image.Image | list | NDArray[np.float64],
    K:int,
    **kwargs: dict[str, Any],
 ) -> RMOutput:
        pass 
    
    @abstractmethod
    def get_vectors_from_index(self, collection_name:str, ids: list[Any]) -> NDArray[np.float64]:
        pass 

    def _batch_embed(self, docs: pd.Series | list) -> NDArray[np.float64]:
        """Create embeddings using the provided embedding model with batching"""
        all_embeddings = []
        for i in tqdm(range(0, len(docs), self.max_batch_size), desc="Creating embeddings"):
            batch = docs[i : i + self.max_batch_size]
            _batch = convert_to_base_data(batch)
            embeddings = self._embed(_batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
