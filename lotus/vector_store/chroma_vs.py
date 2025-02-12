from lotus.vector_store.vs import VS


class ChromaVS(VS):
    def __init__(self):
        try:
            import chromadb
        except ImportError:
            chromadb = None


        if chromadb is None:
            raise ImportError(
                "The chromadb library is required to use ChromaVS. Install it with `pip install chromadb`",
            )
        pass
