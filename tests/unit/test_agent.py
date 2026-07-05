"""Unit tests for the agent orchestrator."""

from collections.abc import AsyncIterator

import pytest

from aiagentrag.agent.agent import Agent
from aiagentrag.core.events import ErrorEvent, FinishedEvent, StatusEvent, TokenEvent
from aiagentrag.core.models import AgentConfig, LLMMessage
from aiagentrag.knowledge.retriever import KnowledgeRetriever
from aiagentrag.memory.compressor import ConversationCompressor
from aiagentrag.memory.repository import MemoryRepository
from aiagentrag.prompt.builder import PromptBuilder
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    InMemoryConversationStore,
    InMemoryVectorStore,
)


@pytest.fixture
def config() -> AgentConfig:
    """Provide a test agent configuration."""
    return AgentConfig(
        system_prompt="You are a helpful assistant.",
        max_recent_messages=10,
        compression_threshold=100,
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
    return FakeLLMProvider(stream_tokens=["Test", " response"])


@pytest.fixture
def agent(
    config: AgentConfig,
    store: InMemoryConversationStore,
    vector_store: InMemoryVectorStore,
    llm: FakeLLMProvider,
) -> Agent:
    """Provide a fully wired agent instance."""
    embedding = FakeEmbeddingProvider()
    memory_repo = MemoryRepository(store, vector_store, embedding, config)
    knowledge = KnowledgeRetriever(vector_store, embedding, config)
    builder = PromptBuilder(config)
    compressor = ConversationCompressor(store, vector_store, embedding, llm, config)
    return Agent(
        config=config,
        memory_repository=memory_repo,
        knowledge_retriever=knowledge,
        prompt_builder=builder,
        llm_provider=llm,
        conversation_compressor=compressor,
    )


async def test_run_emits_status_events(agent: Agent) -> None:
    """Agent should emit status events during execution."""
    events = [event async for event in agent.run("u1", "Hello")]
    status_messages = [e.message for e in events if isinstance(e, StatusEvent)]
    assert "Loading history" in status_messages
    assert "Retrieving memory" in status_messages
    assert "Retrieving knowledge" in status_messages
    assert "Building prompt" in status_messages
    assert "Calling LLM" in status_messages


async def test_run_emits_token_events(agent: Agent) -> None:
    """Agent should stream token events from the LLM."""
    events = [event async for event in agent.run("u1", "Hello")]
    tokens = [e.content for e in events if isinstance(e, TokenEvent)]
    assert tokens == ["Test", " response"]


async def test_run_emits_finished_event(agent: Agent) -> None:
    """Agent should emit a finished event with the full response."""
    events = [event async for event in agent.run("u1", "Hello")]
    finished = [e for e in events if isinstance(e, FinishedEvent)]
    assert len(finished) == 1
    assert finished[0].response == "Test response"
    assert finished[0].metadata.user_id == "u1"


async def test_run_persists_messages(
    agent: Agent,
    store: InMemoryConversationStore,
) -> None:
    """Agent should persist user and assistant messages."""
    _events = [event async for event in agent.run("u1", "Hello")]
    assert await store.count_messages("u1") == 2
    messages = await store.get_all_messages("u1")
    assert messages[0].content == "Hello"
    assert messages[1].content == "Test response"


async def test_run_handles_llm_error(
    config: AgentConfig,
    store: InMemoryConversationStore,
    vector_store: InMemoryVectorStore,
) -> None:
    """Agent should emit an error event when LLM fails."""

    class FailingLLM(FakeLLMProvider):
        async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
            raise RuntimeError("LLM unavailable")
            yield ""

    embedding = FakeEmbeddingProvider()
    llm = FailingLLM()
    memory_repo = MemoryRepository(store, vector_store, embedding, config)
    knowledge = KnowledgeRetriever(vector_store, embedding, config)
    builder = PromptBuilder(config)
    compressor = ConversationCompressor(store, vector_store, embedding, llm, config)
    failing_agent = Agent(
        config=config,
        memory_repository=memory_repo,
        knowledge_retriever=knowledge,
        prompt_builder=builder,
        llm_provider=llm,
        conversation_compressor=compressor,
    )
    events = [event async for event in failing_agent.run("u1", "Hello")]
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert "LLM unavailable" in errors[0].error
