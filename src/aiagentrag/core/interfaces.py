"""Abstract interfaces for all external dependencies."""

from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID

from aiagentrag.core.models import (
    KnowledgeChunk,
    LLMMessage,
    MemoryItem,
    Message,
    VectorSearchResult,
)


class ConversationStore(Protocol):
    """Persists and retrieves active conversation messages."""

    async def get_recent_messages(self, user_id: str, limit: int) -> list[Message]:
        """Return the most recent messages for a user."""

    async def get_all_messages(self, user_id: str) -> list[Message]:
        """Return all messages for a user ordered by creation time."""

    async def save_message(self, message: Message) -> Message:
        """Persist a single message and return it with assigned id."""

    async def delete_messages(self, message_ids: list[UUID]) -> None:
        """Delete messages by their identifiers."""

    async def count_messages(self, user_id: str) -> int:
        """Return the total number of messages for a user."""


class VectorStore(Protocol):
    """Vector database for semantic search."""

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filter_payload: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search a collection by vector similarity."""

    async def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, str],
    ) -> None:
        """Insert or update a vector point in a collection."""


class EmbeddingProvider(Protocol):
    """Generates vector embeddings for text."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""


class LLMProvider(Protocol):
    """Large language model provider."""

    def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream text tokens from the LLM."""

    async def complete(self, messages: list[LLMMessage]) -> str:
        """Generate a complete response from the LLM."""


class MemoryRepositoryProtocol(Protocol):
    """Loads conversation history and retrieves long-term memory."""

    async def get_recent_messages(self, user_id: str) -> list[Message]:
        """Load recent messages from active conversation storage."""

    async def retrieve_memories(self, user_id: str, query: str) -> list[MemoryItem]:
        """Retrieve relevant long-term memories for a user."""

    async def save_message(self, message: Message) -> Message:
        """Persist a message to active conversation storage."""


class KnowledgeRetrieverProtocol(Protocol):
    """Retrieves relevant knowledge base chunks."""

    async def retrieve(self, query: str) -> list[KnowledgeChunk]:
        """Retrieve knowledge chunks relevant to the query."""


class PromptBuilderProtocol(Protocol):
    """Assembles the final LLM prompt."""

    async def build(
        self,
        user_message: str,
        recent_messages: list[Message],
        memories: list[MemoryItem],
        knowledge_chunks: list[KnowledgeChunk],
    ) -> list[LLMMessage]:
        """Build the complete prompt message list."""


class ConversationCompressorProtocol(Protocol):
    """Compresses old conversation history into long-term memory."""

    async def compress_if_needed(self, user_id: str) -> bool:
        """Compress old messages when threshold is exceeded. Returns True if compressed."""
