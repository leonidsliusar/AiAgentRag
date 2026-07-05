"""Unit tests for agent events."""

from aiagentrag.core.events import (
    ErrorEvent,
    FinishedEvent,
    StatusEvent,
    TokenEvent,
)
from aiagentrag.core.models import ErrorDetails, FinishedMetadata


def test_status_event_type() -> None:
    """StatusEvent should have type 'status'."""
    event = StatusEvent(message="Loading history")
    assert event.type == "status"
    assert event.message == "Loading history"


def test_token_event_type() -> None:
    """TokenEvent should have type 'token'."""
    event = TokenEvent(content="Hello")
    assert event.type == "token"
    assert event.content == "Hello"


def test_finished_event_type() -> None:
    """FinishedEvent should contain response and metadata."""
    metadata = FinishedMetadata(
        user_id="user-1",
        message_count=5,
        memories_retrieved=2,
        knowledge_chunks_retrieved=3,
    )
    event = FinishedEvent(response="Done", metadata=metadata)
    assert event.type == "finished"
    assert event.response == "Done"
    assert event.metadata.memories_retrieved == 2


def test_error_event_type() -> None:
    """ErrorEvent should contain structured error details."""
    details = ErrorDetails(error_type="LLMError", message="Connection failed")
    event = ErrorEvent(error="Connection failed", details=details)
    assert event.type == "error"
    assert event.details is not None
    assert event.details.error_type == "LLMError"


def test_error_event_without_details() -> None:
    """ErrorEvent should allow None details."""
    event = ErrorEvent(error="Something went wrong")
    assert event.details is None
