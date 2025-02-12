from lotus.vector_store.vs import VS


class QdrantVS(VS):
    def __init__(self):
        try:
            import qdrant_client
        except ImportError:
            qdrant_client = None


        if qdrant_client is None:
            raise ImportError(
                "The qdrant library is required to use QdrantVS. Install it with `pip install qdrant_client`",
            )
        pass
