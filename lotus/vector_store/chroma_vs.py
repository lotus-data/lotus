from typing import Any, List, Mapping, Optional, Union, cast

import numpy as np
import pandas as pd
from chromadb import Where
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from chromadb import Client, ClientAPI
    from chromadb.api import Collection
    from chromadb.api.types import IncludeEnum
    from chromadb.errors import InvalidDimensionException
except ImportError as err:
    raise ImportError(
        "The chromadb library is required to use ChromaVS. Install it with `pip install chromadb`"
    ) from err 

class ChromaVS(VS):
    def __init__(self, max_batch_size: int = 64):

        client: ClientAPI = Client()

        """Initialize with ChromaDB client and embedding model"""
        super()
        self.client = client
        self.collection: Collection | None = None
        self.index_dir = None
        self.max_batch_size = max_batch_size

    def __del__(self):
        return

    def index(self, docs: Any, embeddings: Any, index_dir: str, **kwargs: dict[str, Any]):
        """Create a collection and add documents with their embeddings"""
        self.index_dir = index_dir
        
        # Create collection without embedding function (we'll provide embeddings directly)
        self.collection = self.client.get_or_create_collection(
            name=index_dir,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for consistency
        )
        
        # Convert docs to list if it's a pandas Series
        docs_list = docs.tolist() if isinstance(docs, pd.Series) else docs
        
        # Prepare documents for addition
        ids = [str(i) for i in range(len(docs_list))]
        metadatas: list[Mapping[str, Union[str, int, float, bool]]] = [{"doc_id": int(i)} for i in range(len(docs_list))]
        
        # Add documents in batches
        batch_size = 100
        for i in tqdm(range(0, len(docs_list), batch_size), desc="Uploading to ChromaDB"):
            end_idx = min(i + batch_size, len(docs_list))
            try:
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=docs_list[i:end_idx],
                    embeddings=embeddings[i:end_idx].tolist(),
                    metadatas=metadatas[i:end_idx]
                )
            except InvalidDimensionException:
                # delete, recreate, then add 
                self.client.delete_collection(index_dir)
                        # Create collection without embedding function (we'll provide embeddings directly)
                self.collection = self.client.get_or_create_collection(
                    name=index_dir,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for consistency
                )
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=docs_list[i:end_idx],
                    embeddings=embeddings[i:end_idx].tolist(),
                    metadatas=metadatas[i:end_idx]
                )

    def load_index(self, index_dir: str):
        """Load an existing collection"""
        try:
            self.collection = self.client.get_collection(index_dir)
            self.index_dir = index_dir
        except ValueError as e:
            raise ValueError(f"Collection {index_dir} not found") from e

    def __call__(
        self,
        query_vectors,
        K: int,
        ids: Optional[list[int]] = None,
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """
        Perform vector search using ChromaDB with optional filtering by document IDs.

        Args:
            query_vectors: Pre-embedded query vectors.
            K (int): Number of nearest neighbors to retrieve.
            ids (Optional[list[Any]]): If provided, the search will be limited to documents with these ids.
            **kwargs: Additional parameters.

        Returns:
            RMOutput: Contains the distances and indices of the nearest neighbors.
        """
        if self.collection is None:
            raise ValueError("No collection loaded. Call load_index first.")

        all_distances: list[list[float]] = []
        all_indices: list[list[int]] = []

        # Process each query vector.
        for query_vector in query_vectors:
            # Prepare the where clause by casting ids to a list of allowed types.
            where_clause: Optional[dict[str, Union[dict[str, List[Union[str, int, float, bool]]]]]] = None
            if ids:
                where_clause = {"doc_id": {"$in": cast(List[Union[str, int, float, bool]], ids)}}
            
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=K,
                include=[IncludeEnum.metadatas, IncludeEnum.distances],
                where=cast(Where, where_clause),
            )

            distances: list[float] = []
            indices: list[int] = []

            # Retrieve and cast search results to help the type checker.
            metadatas = results.get("metadatas")
            dists = results.get("distances")
            if metadatas is not None and dists is not None:
                metadatas = cast(
                    List[List[Mapping[str, Union[str, int, float, bool]]]], metadatas
                )
                dists = cast(List[List[float]], dists)
                for metadata, distance in zip(metadatas[0], dists[0]):
                    if metadata is not None and distance is not None:
                        indices.append(int(metadata["doc_id"]))
                        # Convert squared L2 distances to cosine similarity.
                        distances.append(1 - (distance / 2))

            # Pad results if fewer than K matches are returned.
            while len(indices) < K:
                indices.append(-1)
                distances.append(0.0)

            all_indices.append(indices)
            all_distances.append(distances)

        return RMOutput(
            distances=np.array(all_distances, dtype=np.float32).tolist(),
            indices=np.array(all_indices, dtype=np.int64).tolist()
        )

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """Retrieve vectors for specific document IDs"""
        if self.collection is None or self.index_dir != index_dir:
            self.load_index(index_dir)


        if self.collection is None:  # Add this check after load_index
            raise ValueError(f"Failed to load collection {index_dir}")


        # Convert integer ids to strings for ChromaDB
        str_ids = [str(id) for id in ids]
        
        # Get embeddings from ChromaDB
        results = self.collection.get(
            ids=str_ids,
            include=[IncludeEnum.embeddings]
        )

        if  results['embeddings'] is None:
            raise ValueError("No vectors found for the given ids", results['embeddings'])

        return np.array(results['embeddings'], dtype=np.float64)
        
        