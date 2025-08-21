from typing import Any

import numpy as np
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from qdrant_client.http import models

    qdrant_available = True
except ImportError:
    qdrant_available = False


class QdrantVS(VS):
    def __init__(self, client, max_batch_size: int = 128):
        if not qdrant_available:
            raise ImportError("Please install the qdrant client using `pip install lotus-ai[qdrant]`")

        super().__init__()
        self.client = client
        self.max_batch_size = max_batch_size

        self.index_dir: str | None = None
        self.embedding_dim: int | None = None

    def index(self, docs: list[str], embeddings: NDArray[np.float64], index_dir: str, **kwargs: dict[str, Any]):
        """Create a collection and add documents with their embeddings"""
        self.index_dir = index_dir
        self.embedding_dim = np.reshape(embeddings, (len(embeddings), -1)).shape[1]

        # Delete collection if it already exists
        try:
            self.client.delete_collection(collection_name=index_dir)
        except Exception:
            pass

        # Create the collection with appropriate settings
        self.client.create_collection(
            collection_name=index_dir,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )

        # Prepare points to add to the collection
        points = []
        for idx, (doc, embedding) in enumerate(zip(docs, embeddings)):
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={"content": doc, "doc_id": idx},
                )
            )

        # Add points to the collection in batches
        for i in range(0, len(points), self.max_batch_size):
            batch = points[i : i + self.max_batch_size]
            self.client.upsert(
                collection_name=index_dir,
                points=batch,
                wait=True,
            )

    def load_index(self, index_dir: str):
        """Load/set the collection name to use"""
        self.index_dir = index_dir

        # Verify collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if index_dir not in collection_names:
            raise ValueError(f"Collection {index_dir} not found")

        # Get vector size for future reference
        collection_info = self.client.get_collection(collection_name=index_dir)
        vectors = collection_info.config.params.vectors
        if isinstance(vectors, dict):
            self.embedding_dim = next(iter(vectors.values())).size
        else:
            self.embedding_dim = vectors.size

    def __call__(
        self, query_vectors: NDArray[np.float64], K: int, ids: list[int] | None = None, **kwargs: dict[str, Any]
    ) -> RMOutput:
        """Perform vector search using pre-computed query vectors"""
        if self.index_dir is None:
            raise ValueError("No collection loaded. Call load_index first.")

        results = []
        for query_vector in query_vectors:
            # Create a filter for specific IDs if provided
            id_filter = None
            if ids is not None:
                id_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchAny(any=ids),
                        )
                    ]
                )

            # Perform the search
            search_result = self.client.search(
                collection_name=self.index_dir,
                query_vector=query_vector.tolist(),
                limit=K,
                query_filter=id_filter,
                with_payload=True,
            )
            results.append(search_result)

        # Process results into expected format
        all_distances = []
        all_indices = []

        for result in results:
            distances = []
            indices = []

            for scored_point in result:
                # Get document ID
                doc_id = scored_point.payload.get("doc_id", -1)
                indices.append(doc_id)

                # Convert score to similarity (Qdrant returns a similarity score already)
                similarity = scored_point.score if scored_point.score is not None else 0.0
                distances.append(similarity)

            # Pad results if fewer than K matches
            while len(indices) < K:
                indices.append(-1)
                distances.append(0.0)

            all_distances.append(distances)
            all_indices.append(indices)

        return RMOutput(
            distances=np.array(all_distances, dtype=np.float32).tolist(),  # type: ignore
            indices=np.array(all_indices, dtype=np.int64).tolist(),  # type: ignore
        )

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """Retrieve vectors from Qdrant collection given specific ids"""
        if self.index_dir != index_dir:
            self.load_index(index_dir)

        # Retrieve points by IDs
        points = self.client.retrieve(
            collection_name=index_dir,
            ids=ids,
            with_vectors=True,
        )

        # Extract vectors and ensure order matches the input ids
        assert self.embedding_dim is not None
        vectors = np.zeros((len(ids), self.embedding_dim), dtype=np.float64)
        id_to_idx = {id: idx for idx, id in enumerate(ids)}

        for point in points:
            if point.id in id_to_idx:
                vectors[id_to_idx[point.id]] = np.array(point.vector, dtype=np.float64)

        return vectors
