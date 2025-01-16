from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from pinecone import Pinecone
except ImportError as err:
    raise ImportError(
        "The pinecone library is required to use PineconeVS. Install it with `pip install pinecone`",
    ) from err

class PineconeVS(VS):
    def __init__(self, api_key: str, embedding_model: Callable[[pd.Series | list], NDArray[np.float64]], max_batch_size: int = 64):
        """Initialize Pinecone client with API key and environment"""
        super().__init__(embedding_model)
        self.pinecone = Pinecone(api_key=api_key)
        self.pc_index = None
        self.max_batch_size = max_batch_size


    def index(self, docs: pd.Series, collection_name: str):
        """Create an index and add documents to it"""
        self.collection_name = collection_name
        
        # Get sample embedding to determine vector dimension
        sample_embedding = self._embed([docs.iloc[0]])
        dimension = sample_embedding.shape[1]
        
        # Check if index already exists
        if collection_name not in self.pinecone.list_indexes():
            # Create new index with the correct dimension
            self.pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric="cosine"
            )
        
        # Connect to index
        self.pc_index = self.pinecone.Index(collection_name)
        
        # Convert docs to list if it's a pandas Series
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs
        
        # Create embeddings using the provided embedding model
        embeddings = self._batch_embed(docs_list)
        
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

    def load_index(self, collection_name: str):
        """Connect to an existing Pinecone index"""
        if collection_name not in self.pinecone.list_indexes():
            raise ValueError(f"Index {collection_name} not found")
        
        self.collection_name = collection_name
        self.pc_index = self.pinecone.Index(collection_name)

    def __call__(
        self,
        queries: Union[pd.Series, str, Image.Image, list, NDArray[np.float64]],
        K: int,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using Pinecone"""
        if self.pc_index is None:
            raise ValueError("No index loaded. Call load_index first.")

        # Convert single query to list
        if isinstance(queries, (str, Image.Image)):
            queries = [queries]

        # Handle numpy array queries (pre-computed vectors)
        if isinstance(queries, np.ndarray):
            query_vectors = queries
        else:
            # Convert queries to list if needed
            if isinstance(queries, pd.Series):
                queries = queries.tolist()
            # Create embeddings for text queries
            query_vectors = self._batch_embed(queries)

        # Perform searches
        all_distances = []
        all_indices = []

        for query_vector in query_vectors:
            # Query Pinecone
            results = self.pc_index.query(
                vector=query_vector.tolist(),
                top_k=K,
                include_metadata=True,
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

    def get_vectors_from_index(self, collection_name: str, ids: list[int]) -> NDArray[np.float64]:
        """Retrieve vectors for specific document IDs"""
        if self.pc_index is None or self.collection_name != collection_name:
            self.load_index(collection_name)

        # Fetch vectors from Pinecone
        vectors = []
        for doc_id in ids:
            response = self.pc_index.fetch(ids=[str(doc_id)])
            if str(doc_id) in response.vectors:
                vector = response.vectors[str(doc_id)].values
                vectors.append(vector)
            else:
                raise ValueError(f"Document with id {doc_id} not found")

        return np.array(vectors, dtype=np.float64)
