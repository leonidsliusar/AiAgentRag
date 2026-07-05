"""Unit tests for the prompt builder."""

import pytest

from aiagentrag.core.models import (
    AgentConfig,
    KnowledgeChunk,
    MemoryItem,
    Message,
    MessageRole,
)
from aiagentrag.prompt.builder import PromptBuilder


@pytest.fixture
def config() -> AgentConfig:
    """Provide a test agent configuration."""
    return AgentConfig(system_prompt="You are a helpful assistant.")


@pytest.fixture
def builder(config: AgentConfig) -> PromptBuilder:
    """Provide a prompt builder instance."""
    return PromptBuilder(config)


async def test_build_includes_system_prompt(builder: PromptBuilder) -> None:
    """Built prompt should start with the system prompt."""
    messages = await builder.build(
        user_message="Hello",
        recent_messages=[],
        memories=[],
        knowledge_chunks=[],
    )
    assert messages[0].role == "system"
    assert "helpful assistant" in messages[0].content


async def test_build_includes_conversation_history(builder: PromptBuilder) -> None:
    """Built prompt should include recent messages."""
    recent = [
        Message(user_id="u1", role=MessageRole.USER, content="Hi"),
        Message(user_id="u1", role=MessageRole.ASSISTANT, content="Hello!"),
    ]
    messages = await builder.build(
        user_message="How are you?",
        recent_messages=recent,
        memories=[],
        knowledge_chunks=[],
    )
    assert len(messages) == 4
    assert messages[1].content == "Hi"
    assert messages[2].content == "Hello!"
    assert messages[3].role == "user"
    assert messages[3].content == "How are you?"


async def test_build_includes_memories(builder: PromptBuilder) -> None:
    """Built prompt should include memory section when memories exist."""
    memories = [
        MemoryItem(id="m1", user_id="u1", content="User prefers dark mode.", score=0.9),
    ]
    messages = await builder.build(
        user_message="Hello",
        recent_messages=[],
        memories=memories,
        knowledge_chunks=[],
    )
    assert "Long-term Memory" in messages[0].content
    assert "dark mode" in messages[0].content


async def test_build_includes_knowledge(builder: PromptBuilder) -> None:
    """Built prompt should include knowledge section when chunks exist."""
    chunks = [
        KnowledgeChunk(id="k1", content="Python was created by Guido.", score=0.85),
    ]
    messages = await builder.build(
        user_message="Tell me about Python",
        recent_messages=[],
        memories=[],
        knowledge_chunks=chunks,
    )
    assert "Knowledge Context" in messages[0].content
    assert "Guido" in messages[0].content


async def test_build_omits_empty_sections(builder: PromptBuilder) -> None:
    """Built prompt should not include empty memory or knowledge sections."""
    messages = await builder.build(
        user_message="Hello",
        recent_messages=[],
        memories=[],
        knowledge_chunks=[],
    )
    assert "Long-term Memory" not in messages[0].content
    assert "Knowledge Context" not in messages[0].content


async def test_build_no_duplicate_context(builder: PromptBuilder) -> None:
    """Memory and knowledge should appear only in the system message."""
    memories = [MemoryItem(id="m1", user_id="u1", content="Memory text.", score=0.9)]
    chunks = [KnowledgeChunk(id="k1", content="Knowledge text.", score=0.85)]
    messages = await builder.build(
        user_message="Hello",
        recent_messages=[],
        memories=memories,
        knowledge_chunks=chunks,
    )
    non_system = [m.content for m in messages if m.role != "system"]
    assert not any("Memory text." in content for content in non_system)
    assert not any("Knowledge text." in content for content in non_system)
