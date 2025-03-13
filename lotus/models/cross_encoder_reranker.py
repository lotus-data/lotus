from sentence_transformers import CrossEncoder

from lotus.models.reranker import Reranker
from lotus.types import RerankerOutput


class CrossEncoderReranker(Reranker):
    """CrossEncoder reranker model.

    Args:
        model (str): The name of the reranker model to use.
        device (str): What device to keep the model on.
        max_batch_size (int): The maximum batch size to use for the model.
    """

    # Use lazy init instead of eager because otherwise we have very slow import times.
    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(model=self.model, device=self.device) # type: ignore # CrossEncoder has wrong type stubs
        return self._model


    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        device: str | None = None,
        max_batch_size: int = 64,
    ):
        self.max_batch_size: int = max_batch_size
        self.model = None  # type: ignore # CrossEncoder has wrong type stubs. Also defer initialization.

    def __call__(self, query: str, docs: list[str], K: int) -> RerankerOutput:
        results = self.model.rank(query, docs, top_k=K, batch_size=self.max_batch_size, show_progress_bar=False)
        indices = [int(result["corpus_id"]) for result in results]
        return RerankerOutput(indices=indices)
