import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lotus.types import RMOutput

try:
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
except ImportError:
    pass


class ColBERTv2RM:
    def __init__(self) -> None:
        self.docs: list[str] | None = None
        self.kwargs: dict[str, Any] = {"doc_maxlen": 300, "nbits": 2}
        self.index_dir: str | None = None

    def index(self, docs: list[str], index_dir: str, **kwargs: dict[str, Any]) -> None:
        kwargs = {**self.kwargs, **kwargs}
        checkpoint = "colbert-ir/colbertv2.0"

        with Run().context(RunConfig(nranks=1, experiment="lotus")):
            config = ColBERTConfig(doc_maxlen=kwargs["doc_maxlen"], nbits=kwargs["nbits"], kmeans_niters=4)
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=f"{index_dir}/index", collection=docs, overwrite=True)

        with open(f"experiments/lotus/indexes/{index_dir}/index/docs", "wb") as fp:
            pickle.dump(docs, fp)

        self.docs = docs
        self.index_dir = index_dir

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        with open(f"experiments/lotus/indexes/{index_dir}/index/docs", "rb") as fp:
            self.docs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        raise NotImplementedError("This method is not implemented for ColBERTv2RM")

    def __call__(
        self,
        queries: str | Image.Image | list | NDArray[np.float64],
        K: int,
        **kwargs: dict[str, Any],
    ) -> RMOutput:
        if isinstance(queries, str):
            queries = [queries]

        with Run().context(RunConfig(experiment="lotus")):
            searcher = Searcher(index=f"{self.index_dir}/index", collection=self.docs)

        # make queries a dict with keys as query ids
        assert isinstance(queries, list)
        queries_dict = {i: q for i, q in enumerate(queries)}
        all_results = searcher.search_all(queries_dict, k=K).todict()

        indices = [[result[0] for result in all_results[qid]] for qid in all_results.keys()]
        distances = [[result[2] for result in all_results[qid]] for qid in all_results.keys()]

        return RMOutput(distances=distances, indices=indices)
