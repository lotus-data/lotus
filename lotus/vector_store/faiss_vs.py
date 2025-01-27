import os
import pickle
from typing import Any

import faiss
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS


class FaissVS(VS):
    def __init__(self, factory_string: str = "Flat", metric=faiss.METRIC_INNER_PRODUCT):
        super().__init__()
        self.factory_string = factory_string
        self.metric = metric
        self.index_dir: str | None = None
        self.faiss_index: faiss.Index | None = None
        self.vecs: NDArray[np.float64] | None = None

    def index(self, docs: pd.Series, embeddings, index_dir: str, **kwargs: dict[str, Any]) -> None:
        self.faiss_index = faiss.index_factory(embeddings.shape[1], self.factory_string, self.metric)
        self.faiss_index.add(embeddings)
        self.index_dir = index_dir

        os.makedirs(index_dir, exist_ok=True)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(embeddings, fp)
        faiss.write_index(self.faiss_index, f"{index_dir}/index")

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.faiss_index = faiss.read_index(f"{index_dir}/index")
        with open(f"{index_dir}/vecs", "rb") as fp:
            self.vecs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs: NDArray[np.float64] = pickle.load(fp)
        return vecs[ids]

    def __call__(
        self, query_vectors, K: int, **kwargs: dict[str, Any]
    ) -> RMOutput:
        
        """
        do this processing in the rm 
        if isinstance(queries, str) or isinstance(queries, Image.Image):
            queries = [queries]

        if not isinstance(queries, np.ndarray):
            embedded_queries = self._embed(queries)
        else:
            embedded_queries = np.asarray(queries, dtype=np.float32)
        """
        if self.faiss_index is None:
            raise ValueError("Index not loaded")

        distances, indices = self.faiss_index.search(query_vectors, K)
        return RMOutput(distances=distances, indices=indices)

