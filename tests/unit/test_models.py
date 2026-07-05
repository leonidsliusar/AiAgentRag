"""Unit tests for core domain models."""

from datetime import UTC, datetime
from uuid import uuid4

from aiagentrag.core.models import (
    AgentConfig,
    KnowledgeChunk,
    LLMMessage,
    MemoryItem,
    Message,
    MessageRole,
)


def test_message_defaults() -> None:
    """Message should have generated id and timestamp by default."""
    message = Message(user_id="user-1", role=MessageRole.USER, content="Hello")
    assert message.id is not None
    assert message.created_at.tzinfo == UTC


def test_message_is_immutable() -> None:
    """Message model should be frozen."""
    message = Message(user_id="user-1", role=MessageRole.USER, content="Hello")
    try:
        message.content = "Changed"  # type: ignore[misc]
        raise AssertionError("Expected ValidationError")
    except Exception:
        pass


def test_agent_config_defaults() -> None:
    """AgentConfig should have sensible defaults."""
    config = AgentConfig(system_prompt="You are helpful.")
    assert config.max_recent_messages == 20
    assert config.compression_threshold == 30
    assert config.knowledge_top_k == 5
    assert config.memory_top_k == 5
    assert config.knowledge_collection == "documents"
    assert config.user_memory_collection == "user_memory"


def test_memory_item_creation() -> None:
    """MemoryItem should store retrieval metadata."""
    item = MemoryItem(id="mem-1", user_id="user-1", content="User likes Python.", score=0.95)
    assert item.score == 0.95


def test_knowledge_chunk_with_metadata() -> None:
    """KnowledgeChunk should preserve metadata."""
    chunk = KnowledgeChunk(
        id="chunk-1",
        content="Python is a programming language.",
        score=0.88,
        metadata={"source": "book"},
    )
    assert chunk.metadata["source"] == "book"


def test_llm_message_creation() -> None:
    """LLMMessage should store role and content."""
    msg = LLMMessage(role="system", content="You are helpful.")
    assert msg.role == "system"


def test_message_with_explicit_id() -> None:
    """Message should accept an explicit id."""
    message_id = uuid4()
    message = Message(
        id=message_id,
        user_id="user-1",
        role=MessageRole.ASSISTANT,
        content="Hi",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert message.id == message_id
