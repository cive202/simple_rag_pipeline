import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.client import qdrant_client
from app.core.config import settings


class QdrantStore:
    """Thin wrapper around a single Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self._client = client or qdrant_client
        self._collection = collection or settings.qdrant_collection
        self._dimension = dimension or settings.embedding_dimension

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            existing = vectors.size if isinstance(vectors, VectorParams) else None
            if existing != self._dimension:
                raise ValueError(
                    f"Collection '{self._collection}' has size {existing}, "
                    f"but embedder dimension is {self._dimension}"
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def upsert_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        vectors: list[list[float]],
        extra_payload: dict[str, Any] | None = None,
        point_ids: list[str] | None = None,
        batch_size: int = 100,
    ) -> list[str]:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if point_ids and len(point_ids) != len(chunks):
            raise ValueError("point_ids and chunks must have the same length")

        generated_ids = point_ids or [str(uuid.uuid4()) for _ in chunks]
        points = [
            PointStruct(
                id=generated_ids[index],
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "text": text,
                    **(extra_payload or {}),
                },
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors))
        ]
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                self._collection, points=points[start : start + batch_size]
            )
        return generated_ids

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._client.delete(
            self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )

    def search(
        self, vector: list[float], top_k: int = 5, doc_id: str | None = None
    ) -> list[dict[str, Any]]:
        query_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            if doc_id
            else None
        )
        result = self._client.query_points(
            self._collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {"id": str(p.id), "score": p.score, "payload": p.payload or {}}
            for p in result.points
        ]
