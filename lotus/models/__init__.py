from lotus.models.lm import LM
from lotus.models.reranker import Reranker
from lotus.models.rm import RM

# Heavy RM/reranker backends (torch, sentence_transformers, colbert, faiss)
# are imported lazily so `import lotus` stays fast. LM stays eager because it
# is referenced as a type annotation across the codebase (e.g. lotus.settings).
_LAZY = {
    "LiteLLMRM": "lotus.models.litellm_rm",
    "SentenceTransformersRM": "lotus.models.sentence_transformers_rm",
    "CrossEncoderReranker": "lotus.models.cross_encoder_reranker",
    "ColBERTv2RM": "lotus.models.colbertv2_rm",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        cls = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = cls  # cache so subsequent access bypasses __getattr__
        return cls
    raise AttributeError(f"module 'lotus.models' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "CrossEncoderReranker",
    "LM",
    "RM",
    "Reranker",
    "LiteLLMRM",
    "SentenceTransformersRM",
    "ColBERTv2RM",
]
