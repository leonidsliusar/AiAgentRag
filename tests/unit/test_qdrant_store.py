"""Unit tests for Qdrant vector store."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiagentrag.core.exceptions import StorageError
from aiagentrag.storage.qdrant.client import QdrantVectorStore


@pytest.fixture
def mock_client() -> AsyncMock:
    """Provide a mocked Qdrant client."""
    return AsyncMock()


@pytest.fixture
def store(mock_client: AsyncMock) -> QdrantVectorStore:
    """Provide a Qdrant vector store with mocked client."""
    return QdrantVectorStore(client=mock_client, vector_size=4)


async def test_ensure_collection_creates_when_missing(
    store: QdrantVectorStore,
    mock_client: AsyncMock,
) -> None:
    """Should create collection when it does not exist."""
    mock_client.collection_exists.return_value = False
    await store.ensure_collection("knowledge")
    mock_client.create_collection.assert_called_once()


async def test_ensure_collection_skips_existing(
    store: QdrantVectorStore,
    mock_client: AsyncMock,
) -> None:
    """Should not create collection when it already exists."""
    mock_client.collection_exists.return_value = True
    await store.ensure_collection("knowledge")
    mock_client.create_collection.assert_not_called()


async def test_search_returns_vectorizer_payload(
    store: QdrantVectorStore,
    mock_client: AsyncMock,
) -> None:
    """Should read chunk text from vectorizer payload field."""
    point = MagicMock()
    point.id = "point-1"
    point.score = 0.95
    point.payload = {
        "text": "Vectorizer chunk text.",
        "document_id": "doc-1",
        "chunk_index": 0,
        "pages": [1],
    }
    mock_client.query_points.return_value = MagicMock(points=[point])

    results = await store.search(
        collection="documents",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        top_k=5,
    )
    assert results[0].content == "Vectorizer chunk text."
    assert results[0].metadata["document_id"] == "doc-1"


async def test_search_returns_results(
    store: QdrantVectorStore,
    mock_client: AsyncMock,
) -> None:
    """Should map Qdrant search results to domain models."""
    point = MagicMock()
    point.id = "point-1"
    point.score = 0.95
    point.payload = {"content": "Test content", "source": "book"}
    mock_client.query_points.return_value = MagicMock(points=[point])

    results = await store.search(
        collection="knowledge",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        top_k=5,
    )
    assert len(results) == 1
    assert results[0].content == "Test content"
    assert results[0].score == 0.95
    assert results[0].metadata == {"source": "book"}


async def test_search_raises_storage_error(
    store: QdrantVectorStore,
    mock_client: AsyncMock,
) -> None:
    """Should wrap Qdrant errors in StorageError."""
    mock_client.query_points.side_effect = RuntimeError("Connection failed")
    with pytest.raises(StorageError, match="Failed to search"):
        await store.search(
            collection="knowledge",
            query_vector=[0.0, 0.0, 0.0, 0.0],
            top_k=5,
        )


async def test_upsert_calls_client(
    store: QdrantVectorStore,
    mock_client: AsyncMock,
) -> None:
    """Should upsert a point via the Qdrant client."""
    await store.upsert(
        collection="user_memory",
        point_id="abc-123",
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={"user_id": "u1", "content": "Summary"},
    )
    mock_client.upsert.assert_called_once()
