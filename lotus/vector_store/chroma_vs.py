from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    import chromadb
    from chromadb.api import Collection
except ImportError as err:
    raise ImportError(
        "The chromadb library is required to use ChromaVS. Install it with `pip install chromadb`"
    ) from err 

class ChromaVS(VS):
    def __init__(self, client: chromadb.Client, embedding_model: Callable[[pd.Series | list], NDArray[np.float64]], max_batch_size: int = 64):
        """Initialize with ChromaDB client and embedding model"""
        super().__init__(embedding_model)
        self.client = client
        self.collection: Collection | None = None
        self.collection_name = None
        self.max_batch_size = max_batch_size


    def index(self, docs: pd.Series, collection_name: str):
        """Create a collection and add documents with their embeddings"""
        self.collection_name = collection_name
        
        # Create collection without embedding function (we'll provide embeddings directly)
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for consistency
        )
        
        # Convert docs to list if it's a pandas Series
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs
        
        # Generate embeddings
        embeddings = self._batch_embed(docs_list)
        
        # Prepare documents for addition
        ids = [str(i) for i in range(len(docs_list))]
        metadatas = [{"doc_id": i} for i in range(len(docs_list))]
        
        # Add documents in batches
        batch_size = 100
        for i in tqdm(range(0, len(docs_list), batch_size), desc="Uploading to ChromaDB"):
            end_idx = min(i + batch_size, len(docs_list))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=docs_list[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx]
            )

    def load_index(self, collection_name: str):
        """Load an existing collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
            self.collection_name = collection_name
        except ValueError as e:
            raise ValueError(f"Collection {collection_name} not found") from e

    def __call__(
        self,
        queries: Union[pd.Series, str, Image.Image, list, NDArray[np.float64]],
        K: int,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using ChromaDB"""
        if self.collection is None:
            raise ValueError("No collection loaded. Call load_index first.")

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
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=K,
                include=['metadatas', 'distances']
            )

            # Extract distances and indices
            distances = []
            indices = []
            
            if results['metadatas']:
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    indices.append(metadata['doc_id'])
                    # ChromaDB returns squared L2 distances, convert to cosine similarity
                    # similarity = 1 - (distance / 2)  # Convert L2 distance to cosine similarity
                    distances.append(1 - (distance / 2))

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
        if self.collection is None or self.collection_name != collection_name:
            self.load_index(collection_name)

        # Convert integer ids to strings for ChromaDB
        str_ids = [str(id) for id in ids]
        
        # Get embeddings from ChromaDB
        results = self.collection.get(
            ids=str_ids,
            include=['embeddings']
        )

        if not results['embeddings']:
            raise ValueError("No vectors found for the given ids")

        return np.array(results['embeddings'], dtype=np.float64)
        
        