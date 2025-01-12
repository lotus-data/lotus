from typing import Optional, Union

from lotus.vector_store.vs import VS

try:
    import weaviate
except ImportError:
    raise ImportError("Please install the weaviate client")


class WeaviateVS(VS):
    def __init__(self, weaviate_collection_name:str, weaviate_client: Union[weaviate.WeaviateClient, weaviate.Client], weaviate_collection_text_key: Optional[str] = "content"):
        pass
