"""Shared test fixtures and in-memory implementations."""

from collections.abc import AsyncIterator
from uuid import UUID

from aiagentrag.core.models import (
    LLMMessage,
    Message,
    VectorSearchResult,
)
from aiagentrag.storage.qdrant.schema import extract_metadata, extract_text


class InMemoryConversationStore:
    """In-memory conversation store for testing."""

    def __init__(self) -> None:
        """Initialize empty message storage."""
        self.messages: list[Message] = []

    async def get_recent_messages(self, user_id: str, limit: int) -> list[Message]:
        """Return the most recent messages for a user."""
        user_messages = [m for m in self.messages if m.user_id == user_id]
        user_messages.sort(key=lambda m: m.created_at)
        return user_messages[-limit:]

    async def get_all_messages(self, user_id: str) -> list[Message]:
        """Return all messages for a user ordered by creation time."""
        user_messages = [m for m in self.messages if m.user_id == user_id]
        user_messages.sort(key=lambda m: m.created_at)
        return user_messages

    async def save_message(self, message: Message) -> Message:
        """Persist a single message."""
        self.messages.append(message)
        return message

    async def delete_messages(self, message_ids: list[UUID]) -> None:
        """Delete messages by their identifiers."""
        id_set = set(message_ids)
        self.messages = [m for m in self.messages if m.id not in id_set]

    async def count_messages(self, user_id: str) -> int:
        """Return the total number of messages for a user."""
        return len([m for m in self.messages if m.user_id == user_id])


class InMemoryVectorStore:
    """In-memory vector store for testing."""

    def __init__(self) -> None:
        """Initialize empty vector storage."""
        self.collections: dict[str, list[dict[str, object]]] = {}

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filter_payload: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search by simple keyword overlap (test stub)."""
        points = self.collections.get(collection, [])
        results: list[VectorSearchResult] = []
        for point in points:
            payload = point.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if filter_payload and not all(
                payload.get(key) == value for key, value in filter_payload.items()
            ):
                continue
            content = extract_text(payload)
            results.append(
                VectorSearchResult(
                    id=str(point.get("id", "")),
                    content=content,
                    score=1.0,
                    metadata=extract_metadata(payload),
                ),
            )
        return results[:top_k]

    async def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, str],
    ) -> None:
        """Insert or update a vector point."""
        if collection not in self.collections:
            self.collections[collection] = []
        self.collections[collection].append(
            {"id": point_id, "vector": vector, "payload": payload},
        )


class FakeEmbeddingProvider:
    """Returns a fixed-dimension zero vector for testing."""

    def __init__(self, dimension: int = 4) -> None:
        """Initialize with embedding dimension."""
        self.dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return [0.0] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        return [[0.0] * self.dimension for _ in texts]


class FakeLLMProvider:
    """Returns predetermined responses for testing."""

    def __init__(
        self,
        stream_tokens: list[str] | None = None,
        complete_response: str = "Summary of conversation.",
    ) -> None:
        """Initialize with optional stream tokens and complete response."""
        self.stream_tokens = stream_tokens or ["Hello", " ", "world"]
        self.complete_response = complete_response
        self.last_messages: list[LLMMessage] = []

    async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream predetermined tokens."""
        self.last_messages = messages
        for token in self.stream_tokens:
            yield token

    async def complete(self, messages: list[LLMMessage]) -> str:
        """Return predetermined complete response."""
        self.last_messages = messages
        return self.complete_response
