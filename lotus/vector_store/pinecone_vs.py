from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from pinecone import Index, Pinecone, ServerlessSpec
except ImportError as err:
    raise ImportError(
        "The pinecone library is required to use PineconeVS. Install it with `pip install pinecone`",
    ) from err

class PineconeVS(VS):
    def __init__(self, max_batch_size: int = 64):

        api_key = 'pcsk_45ecSY_CW62eJeL4jwj6dUfaqM6j9dL3uwK12rudednzGisWMxJv9bHH2DLz6tWoY91W84' 

        """Initialize Pinecone client with API key and environment"""
        super()
        self.pinecone = Pinecone(api_key=api_key)
        self.pc_index:Index | None = None 
        self.max_batch_size = max_batch_size

    def __del__(self):  
        return 


    def index(self, docs: pd.Series, embeddings: Any,  index_dir: str, **kwargs: dict[str, Any]):
        """Create an index and add documents to it"""
        self.index_dir = index_dir
        
        dimension = embeddings.shape[1]
        
        # Check if index already exists
        if index_dir not in self.pinecone.list_indexes().names():
            # Create new index with the correct dimension
            self.pinecone.create_index(
                name=index_dir,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec( 
                    cloud='aws', 
                    region='us-east-1'
                )
            )
        elif self.pinecone.describe_index(index_dir).dimension != dimension:
            # resolve any potential dimension-mismatch errors
            self.pinecone.delete_index(index_dir) 
            self.pinecone.create_index(
                name=index_dir,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                )
            )
        
        # Connect to index
        self.pc_index = self.pinecone.Index(index_dir)
        
        # Convert docs to list if it's a pandas Series
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs
        
        # Prepare vectors for upsert
        vectors = []
        for idx, (embedding, doc) in enumerate(zip(embeddings, docs_list)):
            vectors.append({
                "id": str(idx),
                "values": embedding.tolist(),  # Pinecone expects lists, not numpy arrays
                "metadata": {
                    "content": doc,
                    "doc_id": idx
                }
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
            batch = vectors[i:i + batch_size]
            self.pc_index.upsert(vectors=batch)

    def load_index(self, index_dir: str):
        """Connect to an existing Pinecone index"""
        if index_dir not in self.pinecone.list_indexes():
            raise ValueError(f"Index {index_dir} not found")
        
        self.index_dir = index_dir
        self.pc_index = self.pinecone.Index(index_dir)

    def __call__(
        self,
        query_vectors,
        K: int,
        ids: Optional[list[int]] = None,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using Pinecone"""
        if self.pc_index is None:
            raise ValueError("No index loaded. Call load_index first.")
        K = min(K, 10000)

        # Perform searches
        all_distances = []
        all_indices = []

        for query_vector in query_vectors:
            # Query Pinecone
            results = self.pc_index.query(
                vector=query_vector.tolist(),
                top_k=max(K, 2),
                include_metadata=True,
                filter={
                    "doc_id": {
                        "$in": ids
                    } ,
                } if ids is not None else None,
                **kwargs
            )


            # Extract distances and indices
            distances = []
            indices = []
            
            for match in results.matches:
                indices.append(int(match.metadata["doc_id"]))
                distances.append(match.score)

            # Pad results if fewer than K matches
            while len(indices) < K:
                indices.append(-1)  # Use -1 for padding
                distances.append(0.0)

            all_distances.append(distances)
            all_indices.append(indices)

        return RMOutput(
            distances=np.array(all_distances, dtype=np.float32).tolist(),
            indices=np.array(all_indices, dtype=np.int64).tolist()
        )

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """Retrieve vectors for specific document IDs"""
        raise ValueError('Not a Pinecone supported operation!')
