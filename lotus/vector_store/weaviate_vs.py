from lotus.vector_store.vs import VS


class WeaviateVS(VS):
    def __init__(self):
        try:
            import weaviate
        except ImportError:
            weaviate = None 
            
        if weaviate is None:
            raise ImportError("Please install the weaviate client")
        pass
