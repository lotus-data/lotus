from lotus.vector_store.vs import VS

try:
    import weaviate
except ImportError as err:
    raise ImportError(
        "Please install the weaviate client"
    )

class WeaviateVS(VS):

    def __init__(self):
        pass 

    