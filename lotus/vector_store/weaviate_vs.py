from typing import Any, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
except ImportError as err:
    raise ImportError("Please install the weaviate client") from err 

class WeaviateVS(VS):
    def __init__(self, max_batch_size: int = 64):

        REST_URL = 'https://dovieiknqr20pmgoticrmw.c0.us-west3.gcp.weaviate.cloud'

        API_KEY = 'nwRhjKLulSWbhPjX67WBklmJs7dgUS9XGWrZ'

        weaviate_client: weaviate.WeaviateClient | None = None #  need to set this up 


        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=REST_URL,                                    # Replace with your Weaviate Cloud URL
            auth_credentials=Auth.api_key(API_KEY),             # Replace with your Weaviate Cloud key
        )

        """Initialize with Weaviate client and embedding model"""
        super()
        self.client = weaviate_client
        self.max_batch_size = max_batch_size

    def __del__(self):
        self.client.close()

    def index(self, docs: pd.Series, embeddings, index_dir: str, **kwargs: dict[str, Any]):
        """Create a collection and add documents with their embeddings"""
        self.index_dir = index_dir
        
        # Create collection without vectorizer config (we'll provide vectors directly)
        if not self.client.collections.exists(index_dir):
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
                vector_index_config=Configure.VectorIndex.dynamic()
            )
        else:
            collection = self.client.collections.get(index_dir)

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
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using pre-computed query vectors"""
        if self.index_dir is None:
            raise ValueError("No collection loaded. Call load_index first.")

        collection = self.client.collections.get(self.index_dir)

        """

        do this in the retriever module
        # Convert single query to list
        if isinstance(queries, (str, Image.Image)):
            queries = [queries]
        
        # Handle numpy array queries (pre-computed vectors)
        if isinstance(queries, np.ndarray):
            query_vectors = queries
        else:
            # Generate embeddings for text queries
            query_vectors = self._batch_embed(queries)
        """

        # Perform searches
        results = []
        for query_vector in query_vectors:
            response = (collection.query
                .near_vector(
                    near_vector=query_vector.tolist(),
                    limit=K,
                    return_metadata=MetadataQuery(distance=True)
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
        """Retrieve vectors for specific document IDs"""
        collection = self.client.collections.get(index_dir)
        
        # Query for documents with specific doc_ids
        vectors = []

        for id in ids:
            exists = False 
            for obj in collection.query.fetch_objects().objects:
                if id == obj.properties.get('doc_id', -1):
                    exists = True
                    vectors.append(obj.vector)             
            if not exists:
                raise ValueError(f'{id} does not exist in {index_dir}')
        return np.array(vectors, dtype=np.float64)
        
        
