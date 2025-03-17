import numpy as np
from litellm import embedding
from litellm.types.utils import EmbeddingResponse
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM


class LiteLLMRM(RM):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 64,
        truncate_limit: int | None = None,
    ):
        super()
        self.model: str = model
        self.max_batch_size: int = max_batch_size
        self.truncate_limit: int | None = truncate_limit

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
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
