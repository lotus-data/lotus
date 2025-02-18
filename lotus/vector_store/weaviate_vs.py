from typing import Any, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth
    from weaviate.classes.query import Filter, MetadataQuery
except ImportError as err:
    raise ImportError("Please install the weaviate client") from err 

class WeaviateVS(VS):
    def __init__(self, max_batch_size: int = 64, vector_index_config = Configure.VectorIndex.hnsw(), API_KEY=None, REST_URL=None):


        weaviate_client: weaviate.WeaviateClient | None = None


        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=REST_URL,                                    
            auth_credentials=Auth.api_key(API_KEY),            
        )

        """Initialize with Weaviate client and embedding model"""
        super()
        self.client = weaviate_client
        self.max_batch_size = max_batch_size
        self.vector_index_config = vector_index_config
        self.embedding_dim: Optional[int] = None 
    def __del__(self):
        self.client.close()

    def get_collection_dimension(self, index_dir):
        if not self.embedding_dim:
            self.embedding_dim = self.client.collections.get(index_dir).config
        return self.embedding_dim
    
    def index(self, docs: pd.Series, embeddings, index_dir: str, **kwargs: dict[str, Any]):
        """Create a collection and add documents with their embeddings"""
        self.index_dir = index_dir

        if self.client.collections.exists(index_dir):
            self.client.collections.delete(index_dir)
            self.embedding_dim = np.reshape(embeddings, (len(embeddings), -1)).shape[1]
        

        collection = self.client.collections.create(
            name=index_dir,
            properties=[
                Property(
                    name='content', 
                    data_type=DataType.TEXT
                ),
                Property(
                    name='doc_id',
                    data_type=DataType.INT,
                )
            ],
            vectorizer_config=None,  # No vectorizer needed as we provide vectors
            vector_index_config=self.vector_index_config
        )

        # Generate embeddings for all documents
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs


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
                )

    def load_index(self, index_dir: str):
        """Load/set the collection name to use"""
        self.index_dir = index_dir
        # Verify collection exists
        try:
            self.client.collections.get(index_dir)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            raise ValueError(f"Collection {index_dir} not found")

    def __call__(self,
        query_vectors,
        K: int,
        ids: Optional[list[int]] = None,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using pre-computed query vectors"""
        if self.index_dir is None:
            raise ValueError("No collection loaded. Call load_index first.")

        collection = self.client.collections.get(self.index_dir)

        # Perform searches
        results = []
        for query_vector in query_vectors:
            response = (collection.query
                .near_vector(
                    near_vector=query_vector.tolist(),
                    limit=K,
                    return_metadata=MetadataQuery(distance=True),
                    filters=Filter.any_of([Filter.by_property('doc_id').equal(id) for id in ids]) if ids is not None else None
                    ))
            results.append(response)

        # Process results into expected format
        all_distances = []
        all_indices = []
        
        for result in results:
            objects = result.objects 
            
            distances:List[float] = []
            indices = []
            for obj in objects:
                indices.append(obj.properties.get('doc_id', -1))
                # Convert cosine distance to similarity score
                distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0
                distances.append(1 - distance)  # Convert distance to similarity                
            # Pad results if fewer than K matches
            while len(indices) < K:
                indices.append(-1)
                distances.append(0.0)
                
            all_distances.append(distances)
            all_indices.append(indices)

        return RMOutput(
            distances=np.array(all_distances, dtype=np.float32).tolist(),
            indices=np.array(all_indices, dtype=np.int64).tolist()
        )

    def get_vectors_from_index(self, index_dir: str, ids: list[Any]) -> NDArray[np.float64]:
        raise NotImplementedError("Weaviate does not support get_vectors_from_index")