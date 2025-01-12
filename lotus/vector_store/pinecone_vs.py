from lotus.vector_store.vs import VS

try:
    import pinecone
except ImportError:
    pinecone = None 


if pinecone is None:
    raise ImportError(
        "The pinecone library is required to use PineconeVS. Install it with `pip install pinecone`",
    )

class PineconeVS(VS):

    def __init__(self):
        pass 

