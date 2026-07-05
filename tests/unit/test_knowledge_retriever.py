"""Unit tests for the knowledge retriever."""

import pytest

from aiagentrag.core.models import AgentConfig
from aiagentrag.knowledge.retriever import KnowledgeRetriever
from tests.conftest import FakeEmbeddingProvider, InMemoryVectorStore


@pytest.fixture
def config() -> AgentConfig:
    """Provide a test agent configuration."""
    return AgentConfig(system_prompt="Test", knowledge_top_k=2, knowledge_collection="documents")


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Provide an in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def retriever(vector_store: InMemoryVectorStore, config: AgentConfig) -> KnowledgeRetriever:
    """Provide a knowledge retriever instance."""
    return KnowledgeRetriever(
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        config=config,
    )


async def test_retrieve_returns_knowledge_chunks(
    retriever: KnowledgeRetriever,
    vector_store: InMemoryVectorStore,
) -> None:
    """Should return knowledge chunks from the vectorizer documents collection."""
    await vector_store.upsert(
        collection="documents",
        point_id="k1",
        vector=[0.0, 0.0, 0.0, 0.0],
        payload={
            "text": "Machine learning basics.",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
    )
    chunks = await retriever.retrieve("machine learning")
    assert len(chunks) == 1
    assert chunks[0].content == "Machine learning basics."
    assert chunks[0].metadata["document_id"] == "doc-1"


async def test_retrieve_respects_top_k(
    retriever: KnowledgeRetriever,
    vector_store: InMemoryVectorStore,
) -> None:
    """Should respect the configured top_k limit."""
    for i in range(5):
        await vector_store.upsert(
            collection="documents",
            point_id=f"k{i}",
            vector=[0.0, 0.0, 0.0, 0.0],
            payload={"text": f"Chunk {i}."},
        )
    chunks = await retriever.retrieve("query")
    assert len(chunks) == 2


async def test_retrieve_empty_collection(retriever: KnowledgeRetriever) -> None:
    """Should return empty list when no knowledge exists."""
    chunks = await retriever.retrieve("anything")
    assert chunks == []
