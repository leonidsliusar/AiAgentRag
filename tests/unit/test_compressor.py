"""Unit tests for conversation compression."""

import pytest

from aiagentrag.core.models import AgentConfig, Message, MessageRole
from aiagentrag.memory.compressor import ConversationCompressor
from aiagentrag.memory.repository import USER_MEMORY_COLLECTION
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    InMemoryConversationStore,
    InMemoryVectorStore,
)


@pytest.fixture
def config() -> AgentConfig:
    """Provide a test configuration with low compression threshold."""
    return AgentConfig(
        system_prompt="Test",
        max_recent_messages=2,
        compression_threshold=4,
    )


@pytest.fixture
def store() -> InMemoryConversationStore:
    """Provide an in-memory conversation store."""
    return InMemoryConversationStore()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Provide an in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def llm() -> FakeLLMProvider:
    """Provide a fake LLM provider."""
    return FakeLLMProvider(complete_response="User discussed Python programming.")


@pytest.fixture
def compressor(
    store: InMemoryConversationStore,
    vector_store: InMemoryVectorStore,
    llm: FakeLLMProvider,
    config: AgentConfig,
) -> ConversationCompressor:
    """Provide a conversation compressor instance."""
    return ConversationCompressor(
        conversation_store=store,
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=llm,
        config=config,
    )


async def test_no_compression_below_threshold(
    compressor: ConversationCompressor,
    store: InMemoryConversationStore,
) -> None:
    """Should not compress when message count is below threshold."""
    for i in range(3):
        await store.save_message(
            Message(user_id="u1", role=MessageRole.USER, content=f"Msg {i}"),
        )
    result = await compressor.compress_if_needed("u1")
    assert result is False
    assert await store.count_messages("u1") == 3


async def test_compression_above_threshold(
    compressor: ConversationCompressor,
    store: InMemoryConversationStore,
    vector_store: InMemoryVectorStore,
) -> None:
    """Should compress old messages and keep recent ones."""
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        await store.save_message(
            Message(user_id="u1", role=role, content=f"Msg {i}"),
        )
    result = await compressor.compress_if_needed("u1")
    assert result is True
    assert await store.count_messages("u1") == 2
    memories = vector_store.collections.get(USER_MEMORY_COLLECTION, [])
    assert len(memories) == 1
    payload = memories[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["content"] == "User discussed Python programming."


async def test_compression_preserves_recent_messages(
    compressor: ConversationCompressor,
    store: InMemoryConversationStore,
) -> None:
    """Should keep the most recent messages in PostgreSQL."""
    messages = []
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msg = Message(user_id="u1", role=role, content=f"Msg {i}")
        messages.append(msg)
        await store.save_message(msg)

    await compressor.compress_if_needed("u1")
    remaining = await store.get_all_messages("u1")
    assert len(remaining) == 2
    assert remaining[-1].content == "Msg 5"
    assert remaining[-2].content == "Msg 4"
