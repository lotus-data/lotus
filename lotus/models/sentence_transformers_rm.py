import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM


class SentenceTransformersRM(RM):
    """
    A retrieval model based on Sentence Transformers.

    This class provides functionality to generate embeddings for documents using
    Sentence Transformers models. It supports batch processing and optional
    embedding normalization for efficient embedding generation.

    Attributes:
        model (str): Name of the Sentence Transformers model to use.
        max_batch_size (int): Maximum batch size for embedding requests.
        normalize_embeddings (bool): Whether to normalize embeddings.
        transformer (SentenceTransformer): The Sentence Transformer model instance.
    """

    def __init__(
        self,
        model: str = "intfloat/e5-base-v2",
        max_batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        """
        Initialize the SentenceTransformersRM retrieval model.

        Args:
            model: Name of the Sentence Transformers model to use.
                   Defaults to "intfloat/e5-base-v2".
            max_batch_size: Maximum batch size for embedding requests. Defaults to 64.
            normalize_embeddings: Whether to normalize embeddings. Defaults to True.
            device: Device to run the model on (e.g., "cuda", "cpu").
                    If None, uses default device. Defaults to None.
        """
        self.model: str = model
        self.max_batch_size: int = max_batch_size
        self.normalize_embeddings: bool = normalize_embeddings
        self.transformer: SentenceTransformer = SentenceTransformer(model, device=device)

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        """
        Generate embeddings for a list of documents using Sentence Transformers.

        This method processes documents in batches to generate embeddings using
        the specified Sentence Transformers model. It supports optional embedding
        normalization and shows progress with a progress bar.

        Args:
            docs: List of document strings to embed.

        Returns:
            NDArray[np.float64]: Array of embeddings with shape (num_docs, embedding_dim).

        Raises:
            Exception: If the embedding generation fails.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(docs), self.max_batch_size)):
            batch = docs[i : i + self.max_batch_size]
            _batch = convert_to_base_data(batch)
            torch_embeddings = self.transformer.encode(
                _batch, convert_to_tensor=True, normalize_embeddings=self.normalize_embeddings, show_progress_bar=False
            )
            assert isinstance(torch_embeddings, torch.Tensor)
            cpu_embeddings = torch_embeddings.cpu().numpy()
            all_embeddings.append(cpu_embeddings)
        return np.vstack(all_embeddings)
