import numpy as np
from litellm import embedding
from litellm.types.utils import EmbeddingResponse
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM


class LiteLLMRM(RM):
    """
    A retrieval model based on LiteLLM embedding models.

    This class provides functionality to generate embeddings for documents using
    various embedding models supported by LiteLLM. It supports batch processing
    and optional text truncation for efficient embedding generation.

    Attributes:
        model (str): Name of the embedding model to use.
        max_batch_size (int): Maximum batch size for embedding requests.
        truncate_limit (int | None): Maximum character limit for text truncation.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 64,
        truncate_limit: int | None = None,
    ) -> None:
        """
        Initialize the LiteLLMRM retrieval model.

        Args:
            model: Name of the embedding model to use. Defaults to "text-embedding-3-small".
            max_batch_size: Maximum batch size for embedding requests. Defaults to 64.
            truncate_limit: Maximum character limit for text truncation.
                           If None, no truncation is applied. Defaults to None.
        """
        super().__init__()
        self.model: str = model
        self.max_batch_size: int = max_batch_size
        self.truncate_limit: int | None = truncate_limit

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        """
        Generate embeddings for a list of documents.

        This method processes documents in batches to generate embeddings using
        the specified embedding model. It supports optional text truncation and
        shows progress with a progress bar.

        Args:
            docs: List of document strings to embed.

        Returns:
            NDArray[np.float64]: Array of embeddings with shape (num_docs, embedding_dim).

        Raises:
            Exception: If the embedding API request fails.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(docs), self.max_batch_size)):
            batch = docs[i : i + self.max_batch_size]
            if self.truncate_limit:
                batch = [doc[: self.truncate_limit] for doc in batch]
            _batch = convert_to_base_data(batch)
            response: EmbeddingResponse = embedding(model=self.model, input=_batch)
            embeddings = np.array([d["embedding"] for d in response.data])
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
