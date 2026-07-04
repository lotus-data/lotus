from lotus.vector_store.vs import VS

# Concrete vector stores are imported lazily so `import lotus` does not pull in
# faiss / weaviate / qdrant until a vector store is actually used.
_LAZY = {
    "FaissVS": "lotus.vector_store.faiss_vs",
    "WeaviateVS": "lotus.vector_store.weaviate_vs",
    "QdrantVS": "lotus.vector_store.qdrant_vs",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        cls = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = cls  # cache so subsequent access bypasses __getattr__
        return cls
    raise AttributeError(f"module 'lotus.vector_store' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))


__all__ = ["VS", "FaissVS", "WeaviateVS", "QdrantVS"]
