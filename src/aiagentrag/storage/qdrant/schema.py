"""Qdrant payload schema shared with vectorizer."""

DEFAULT_KNOWLEDGE_COLLECTION = "documents"
DEFAULT_USER_MEMORY_COLLECTION = "user_memory"

PAYLOAD_TEXT = "text"
PAYLOAD_CONTENT = "content"
PAYLOAD_CHUNK_ID = "chunk_id"
PAYLOAD_DOCUMENT_ID = "document_id"
PAYLOAD_CHUNK_INDEX = "chunk_index"
PAYLOAD_PAGES = "pages"
PAYLOAD_TOKEN_COUNT = "token_count"
PAYLOAD_USER_ID = "user_id"

TEXT_PAYLOAD_KEYS = (PAYLOAD_TEXT, PAYLOAD_CONTENT)


def extract_text(payload: dict[str, object]) -> str:
    """Extract searchable text from a Qdrant payload."""
    for key in TEXT_PAYLOAD_KEYS:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return ""


def extract_metadata(payload: dict[str, object]) -> dict[str, object]:
    """Return payload fields excluding primary text content."""
    return {key: value for key, value in payload.items() if key not in TEXT_PAYLOAD_KEYS}
