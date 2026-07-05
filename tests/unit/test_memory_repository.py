"""Unit tests for the memory repository."""

import pytest

from aiagentrag.core.models import AgentConfig, Message, MessageRole
from aiagentrag.memory.repository import USER_MEMORY_COLLECTION, MemoryRepository
from tests.conftest import FakeEmbeddingProvider, InMemoryConversationStore, InMemoryVectorStore


@pytest.fixture
def config() -> AgentConfig:
    """Provide a test agent configuration."""
    return AgentConfig(system_prompt="Test", max_recent_messages=5, memory_top_k=3)


@pytest.fixture
def store() -> InMemoryConversationStore:
    """Provide an in-memory conversation store."""
    return InMemoryConversationStore()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Provide an in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def repository(
    store: InMemoryConversationStore,
    vector_store: InMemoryVectorStore,
    config: AgentConfig,
) -> MemoryRepository:
    """Provide a memory repository instance."""
    return MemoryRepository(
        conversation_store=store,
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        config=config,
    )


async def test_get_recent_messages_respects_limit(
    repository: MemoryRepository,
    store: InMemoryConversationStore,
) -> None:
    """Should return only the configured number of recent messages."""
    for i in range(10):
        await store.save_message(
            Message(user_id="u1", role=MessageRole.USER, content=f"Message {i}"),
        )
    messages = await repository.get_recent_messages("u1")
    assert len(messages) == 5


async def test_save_message_persists(
    repository: MemoryRepository,
    store: InMemoryConversationStore,
) -> None:
    """Should persist a message via the conversation store."""
    message = Message(user_id="u1", role=MessageRole.USER, content="Hello")
    saved = await repository.save_message(message)
    assert saved.id == message.id
    assert len(store.messages) == 1


async def test_retrieve_memories_filters_by_user(
    repository: MemoryRepository,
    vector_store: InMemoryVectorStore,
) -> None:
    """Should filter memories by user_id."""
    await vector_store.upsert(
        collection=USER_MEMORY_COLLECTION,
        point_id="p1",
        vector=[0.0, 0.0, 0.0, 0.0],
        payload={"user_id": "u1", "content": "User likes cats."},
    )
    await vector_store.upsert(
        collection=USER_MEMORY_COLLECTION,
        point_id="p2",
        vector=[0.0, 0.0, 0.0, 0.0],
        payload={"user_id": "u2", "content": "User likes dogs."},
    )
    memories = await repository.retrieve_memories("u1", "pets")
    assert len(memories) == 1
    assert memories[0].content == "User likes cats."
