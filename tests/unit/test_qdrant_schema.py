"""Unit tests for vectorizer-compatible Qdrant schema helpers."""

from aiagentrag.storage.qdrant.schema import (
    DEFAULT_KNOWLEDGE_COLLECTION,
    extract_metadata,
    extract_text,
)


def test_default_knowledge_collection_matches_vectorizer() -> None:
    """Knowledge collection name must match vectorizer defaults."""
    assert DEFAULT_KNOWLEDGE_COLLECTION == "documents"


def test_extract_text_from_vectorizer_payload() -> None:
    """Vectorizer stores chunk text in the text payload field."""
    payload = {"text": "Chunk body", "document_id": "doc-1"}
    assert extract_text(payload) == "Chunk body"


def test_extract_text_from_user_memory_payload() -> None:
    """Agent user memory still uses the content payload field."""
    payload = {"content": "Summary", "user_id": "u1"}
    assert extract_text(payload) == "Summary"


def test_extract_metadata_excludes_text_fields() -> None:
    """Metadata should not duplicate primary text content."""
    payload = {"text": "Chunk body", "document_id": "doc-1", "chunk_index": 2}
    assert extract_metadata(payload) == {"document_id": "doc-1", "chunk_index": 2}
