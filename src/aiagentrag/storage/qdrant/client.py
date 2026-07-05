"""Qdrant vector store implementation."""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from aiagentrag.core.exceptions import StorageError
from aiagentrag.core.models import VectorSearchResult
from aiagentrag.storage.qdrant.schema import extract_metadata, extract_text


class QdrantVectorStore:
    """Vector database implementation using Qdrant."""

    def __init__(self, client: AsyncQdrantClient, vector_size: int) -> None:
        """Initialize the Qdrant vector store."""
        self._client = client
        self._vector_size = vector_size

    async def collection_exists(self, collection: str) -> bool:
        """Return True when the collection exists in Qdrant."""
        try:
            return await self._client.collection_exists(collection)
        except Exception as exc:
            msg = f"Failed to check collection '{collection}': {exc}"
            raise StorageError(msg) from exc

    async def ensure_collection(self, collection: str) -> None:
        """Create a collection if it does not exist."""
        try:
            exists = await self._client.collection_exists(collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance="Cosine",
                    ),
                )
        except Exception as exc:
            msg = f"Failed to ensure collection '{collection}': {exc}"
            raise StorageError(msg) from exc

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filter_payload: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search a collection by vector similarity."""
        try:
            query_filter = self._build_filter(filter_payload)
            response = await self._client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
            return [
                VectorSearchResult(
                    id=str(point.id),
                    content=extract_text(payload),
                    score=point.score,
                    metadata=extract_metadata(payload),
                )
                for point in response.points
                for payload in [(point.payload or {})]
            ]
        except Exception as exc:
            msg = f"Failed to search collection '{collection}': {exc}"
            raise StorageError(msg) from exc

    async def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, str],
    ) -> None:
        """Insert or update a vector point in a collection."""
        try:
            point = PointStruct(
                id=self._resolve_point_id(point_id),
                vector=vector,
                payload=payload,
            )
            await self._client.upsert(collection_name=collection, points=[point])
        except Exception as exc:
            msg = f"Failed to upsert into collection '{collection}': {exc}"
            raise StorageError(msg) from exc

    @staticmethod
    def _build_filter(filter_payload: dict[str, str] | None) -> Filter | None:
        """Build a Qdrant filter from a payload dictionary."""
        if not filter_payload:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter_payload.items()
        ]
        return Filter(must=conditions)

    @staticmethod
    def _resolve_point_id(point_id: str) -> str | int:
        """Resolve a point id to a valid Qdrant identifier."""
        try:
            return str(uuid.UUID(point_id))
        except ValueError:
            return point_id
