from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
except ImportError as err:
    raise ImportError("Please install the qdrant client") from err

class QdrantVS(VS):
    def __init__(self, max_batch_size: int = 64):

        API_KEY = '_Mic3dVln2gAkS6NLyia6p-CCyMScK42ayuq8Rapm5-xsV5j5_UlIA'

        URL = "https://6f8b9aec-a788-4aac-9aeb-417d307493e8.europe-west3-0.gcp.cloud.qdrant.io:6333"

        client: QdrantClient = QdrantClient(
            url=URL,
            api_key=API_KEY
        ) 

        """Initialize with Qdrant client and embedding model"""
        super() # Fixed the super() call syntax
        self.client: QdrantClient = client
        self.max_batch_size = max_batch_size

    def __del__(self):
        self.client.close()

    def index(self, docs:pd.Series, embeddings, index_dir: str, **kwargs: dict[str, Any]):
        """Create a collection and add documents with their embeddings"""
        self.index_dir = index_dir

        # Get sample embedding to determine vector dimension
        dimension = np.reshape(embeddings, (len(embeddings), -1)).shape[1]
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(index_dir):
            self.client.create_collection(
                collection_name=index_dir,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
        collection_info = self.client.get_collection(index_dir)
        if (collection_info is not None and collection_info.config is not None and collection_info.config.params and collection_info.config.params.vectors):

            vectors = collection_info.config.params.vectors
            if isinstance(vectors, dict):
                # If it's a dict, decide how to handle it.
                # Here we extract the first vector, but you may need a different logic.
                vector = next(iter(vectors.values()))
                size = vector.size
            elif isinstance(vectors, VectorParams):
                size = vectors.size
            else:
                size = None

            if  size != dimension:
                # If there's a discrepancy, create a new version of that collection 
                self.client.delete_collection(index_dir)
                self.client.create_collection(
                    collection_name=index_dir,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )

        # Convert docs to list if it's a pandas Series
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs
        
        # Prepare points for upload
        points = []
        for idx, (doc, embedding) in enumerate(zip(docs_list, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "content": doc,
                        "doc_id": idx
                    }
                )
            )

        # Upload in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=index_dir,
                points=batch
            )

    def load_index(self, index_dir: str):
        """Set the collection name to use"""
        if not self.client.collection_exists(index_dir):
            raise ValueError(f"Collection {index_dir} not found")
        self.index_dir = index_dir

    def __call__(
        self,
        query_vectors,
        K: int,
        ids: Optional[list[int]] = None,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using Qdrant"""
        if self.index_dir is None:
            raise ValueError("No collection loaded. Call load_index first.")
        
        # Perform searches
        all_distances = []
        all_indices = []

        for query_vector in query_vectors:
            results = self.client.search(
                collection_name=self.index_dir,
                query_vector=query_vector.tolist(),
                limit=K,
                with_payload=True,
                query_filter=Filter(
                    should=[
                        FieldCondition(key="doc_id", match=MatchValue(value=id)) for id in ids
                    ]
                ) if ids is not None else None
            )

            # Extract distances and indices
            distances = []
            indices = []
            
            for result in results:
                indices.append(result.id)
                distances.append(result.score)  # Qdrant returns cosine similarity directly

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

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """Retrieve vectors for specific document IDs"""
        if self.index_dir != index_dir:
            self.load_index(index_dir)

        # Fetch points from Qdrant
        points = self.client.retrieve(
            collection_name=index_dir,
            ids=ids,
            with_vectors=True,
            with_payload=False
        )

        # Extract and return vectors
        vectors = []
        for point in points:
            if point.vector is not None:
                vectors.append(point.vector)
            else:
                raise ValueError(f"Vector not found for id {point.id}")

        return np.array(vectors, dtype=np.float64)