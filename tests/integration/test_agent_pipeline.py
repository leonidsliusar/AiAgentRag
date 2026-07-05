"""Integration tests for the full agent pipeline."""

import pytest

from aiagentrag.agent.agent import Agent
from aiagentrag.core.events import FinishedEvent, TokenEvent
from aiagentrag.core.models import AgentConfig, Message, MessageRole
from aiagentrag.knowledge.retriever import KNOWLEDGE_COLLECTION, KnowledgeRetriever
from aiagentrag.memory.compressor import ConversationCompressor
from aiagentrag.memory.repository import USER_MEMORY_COLLECTION, MemoryRepository
from aiagentrag.prompt.builder import PromptBuilder
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    InMemoryConversationStore,
    InMemoryVectorStore,
)


@pytest.fixture
def config() -> AgentConfig:
    """Provide integration test configuration."""
    return AgentConfig(
        system_prompt="You are an expert assistant.",
        max_recent_messages=3,
        compression_threshold=5,
        knowledge_top_k=3,
        memory_top_k=3,
    )


@pytest.fixture
def infrastructure(config: AgentConfig) -> dict[str, object]:
    """Set up full in-memory infrastructure."""
    store = InMemoryConversationStore()
    vector_store = InMemoryVectorStore()
    embedding = FakeEmbeddingProvider()
    llm = FakeLLMProvider(stream_tokens=["Integrated", " ", "response"])

    memory_repo = MemoryRepository(store, vector_store, embedding, config)
    knowledge = KnowledgeRetriever(vector_store, embedding, config)
    builder = PromptBuilder(config)
    compressor = ConversationCompressor(store, vector_store, embedding, llm, config)
    agent = Agent(
        config=config,
        memory_repository=memory_repo,
        knowledge_retriever=knowledge,
        prompt_builder=builder,
        llm_provider=llm,
        conversation_compressor=compressor,
    )
    return {
        "store": store,
        "vector_store": vector_store,
        "llm": llm,
        "agent": agent,
    }


async def test_full_pipeline_with_context(infrastructure: dict[str, object]) -> None:
    """Agent should use memory and knowledge context in the prompt."""
    store = infrastructure["store"]
    assert isinstance(store, InMemoryConversationStore)
    vector_store = infrastructure["vector_store"]
    assert isinstance(vector_store, InMemoryVectorStore)
    llm = infrastructure["llm"]
    assert isinstance(llm, FakeLLMProvider)
    agent = infrastructure["agent"]
    assert isinstance(agent, Agent)

    await vector_store.upsert(
        collection=KNOWLEDGE_COLLECTION,
        point_id="k1",
        vector=[0.0] * 4,
        payload={
            "text": "The capital of France is Paris.",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
    )
    await vector_store.upsert(
        collection=USER_MEMORY_COLLECTION,
        point_id="m1",
        vector=[0.0] * 4,
        payload={"user_id": "u1", "content": "User lives in Europe."},
    )
    await store.save_message(
        Message(user_id="u1", role=MessageRole.USER, content="Previous question"),
    )
    await store.save_message(
        Message(user_id="u1", role=MessageRole.ASSISTANT, content="Previous answer"),
    )

    events = [event async for event in agent.run("u1", "What is the capital of France?")]
    tokens = [e.content for e in events if isinstance(e, TokenEvent)]
    assert "".join(tokens) == "Integrated response"

    system_content = llm.last_messages[0].content
    assert "Paris" in system_content
    assert "Europe" in system_content

    finished = [e for e in events if isinstance(e, FinishedEvent)]
    assert finished[0].metadata.knowledge_chunks_retrieved == 1
    assert finished[0].metadata.memories_retrieved == 1


async def test_compression_triggered_after_multiple_turns(
    infrastructure: dict[str, object],
) -> None:
    """Compression should activate after exceeding the threshold."""
    store = infrastructure["store"]
    assert isinstance(store, InMemoryConversationStore)
    vector_store = infrastructure["vector_store"]
    assert isinstance(vector_store, InMemoryVectorStore)
    agent = infrastructure["agent"]
    assert isinstance(agent, Agent)

    for turn in range(3):
        _events = [event async for event in agent.run("u1", f"Question {turn}")]

    assert await store.count_messages("u1") <= 3
    memories = vector_store.collections.get(USER_MEMORY_COLLECTION, [])
    assert len(memories) >= 1


async def test_multiple_users_isolated(infrastructure: dict[str, object]) -> None:
    """Different users should have isolated conversation histories."""
    store = infrastructure["store"]
    assert isinstance(store, InMemoryConversationStore)
    agent = infrastructure["agent"]
    assert isinstance(agent, Agent)

    _events_u1 = [event async for event in agent.run("u1", "Hello from u1")]
    _events_u2 = [event async for event in agent.run("u2", "Hello from u2")]

    u1_messages = await store.get_all_messages("u1")
    u2_messages = await store.get_all_messages("u2")
    assert len(u1_messages) == 2
    assert len(u2_messages) == 2
    assert u1_messages[0].content == "Hello from u1"
    assert u2_messages[0].content == "Hello from u2"
