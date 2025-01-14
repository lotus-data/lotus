from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from uuid import uuid4

    import weaviate
    from weaviate.util import get_valid_uuid
except ImportError as err:
    raise ImportError("Please install the weaviate client") from err 

class WeaviateVS(VS):
    def __init__(self, weaviate_client: weaviate.WeaviateClient, embedding_model: Callable[[pd.Series | list], NDArray[np.float64]], max_batch_size: int = 64):
        """Initialize with Weaviate client and embedding model"""
        super().__init__(embedding_model)
        self.client = weaviate_client
        self.max_batch_size = max_batch_size

    def index(self, docs: pd.Series, collection_name: str):
        """Create a collection and add documents with their embeddings"""
        self.collection_name = collection_name
        
        # Get sample embedding to determine vector dimension
        sample_embedding = self._embed([docs.iloc[0]])
        vector_dim = sample_embedding.shape[1]
        
        # Create collection without vectorizer config (we'll provide vectors directly)
        collection = self.client.collections.create(
            name=collection_name,
            properties=[
                {
                    "name": "content",
                    "dataType": ["text"],
                },
                {
                    "name": "doc_id",
                    "dataType": ["int"],
                }
            ],
            vectorizer_config=None,  # No vectorizer needed as we provide vectors
            vector_index_config={"distance": "cosine"},
            vectorIndexConfig={
                "distance": "cosine",
                "dimension": vector_dim
            }
        )

        # Generate embeddings for all documents
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs
        embeddings = self._batch_embed(docs_list)

        # Add documents to collection with their embeddings
        with collection.batch.dynamic() as batch:
            for idx, (doc, embedding) in enumerate(zip(docs_list, embeddings)):
                properties = {
                    "content": doc,
                    "doc_id": idx
                }
                batch.add_object(
                    properties=properties,
                    vector=embedding.tolist(),  # Provide pre-computed vector
                    uuid=get_valid_uuid(str(uuid4()))
                )

    def load_index(self, collection_name: str):
        """Load/set the collection name to use"""
        self.collection_name = collection_name
        # Verify collection exists
        try:
            self.client.collections.get(collection_name)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            raise ValueError(f"Collection {collection_name} not found")

    def __call__(self,
        queries: Union[pd.Series, str, Image.Image, list, NDArray[np.float64]],
        K: int,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using pre-computed query vectors"""
        if self.collection_name is None:
            raise ValueError("No collection loaded. Call load_index first.")

        collection = self.client.collections.get(self.collection_name)

        # Convert single query to list
        if isinstance(queries, (str, Image.Image)):
            queries = [queries]
        
        # Handle numpy array queries (pre-computed vectors)
        if isinstance(queries, np.ndarray):
            query_vectors = queries
        else:
            # Generate embeddings for text queries
            query_vectors = self._batch_embed(queries)

        # Perform searches
        results = []
        for query_vector in query_vectors:
            response = (collection.query
                .near_vector({
                    "vector": query_vector.tolist()
                })
                .with_limit(K)
                .with_additional(['distance'])
                .with_fields(['doc_id'])
                .do())
            results.append(response)

        # Process results into expected format
        all_distances = []
        all_indices = []
        
        for result in results:
            objects = result.get('data', {}).get('Get', {}).get(self.collection_name, [])
            
            distances = []
            indices = []
            for obj in objects:
                indices.append(obj['doc_id'])
                # Convert cosine distance to similarity score
                distance = obj.get('_additional', {}).get('distance', 0)
                distances.append(1 - distance)  # Convert distance to similarity
                
            # Pad results if fewer than K matches
            while len(indices) < K:
                indices.append(-1)
                distances.append(0.0)
                
            all_distances.append(distances)
            all_indices.append(indices)

        return RMOutput(
            distances=np.array(all_distances, dtype=np.float32),
            indices=np.array(all_indices, dtype=np.int64)
        )

    def get_vectors_from_index(self, collection_name: str, ids: list[int]) -> NDArray[np.float64]:
        """Retrieve vectors for specific document IDs"""
        collection = self.client.collections.get(collection_name)
        
        # Query for documents with specific doc_ids
        vectors = []
        for doc_id in ids:
            response = (collection.query
                .with_fields(['_additional {vector}'])
                .with_where({
                    'path': ['doc_id'],
                    'operator': 'Equal',
                    'valueNumber': doc_id
                })
                .do())
            
            # Extract vector from response
            objects = response.get('data', {}).get('Get', {}).get(collection_name, [])
            if objects:
                vector = objects[0].get('_additional', {}).get('vector', [])
                vectors.append(vector)
            else:
                raise ValueError(f"Document with id {doc_id} not found")
                
        return np.array(vectors, dtype=np.float64)
        
        
