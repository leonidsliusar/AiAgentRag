"""Unit tests for domain exceptions."""

from aiagentrag.core.exceptions import (
    AgentError,
    CompressionError,
    LLMError,
    PromptBuildError,
    RetrievalError,
    StorageError,
)


def test_exception_hierarchy() -> None:
    """All domain exceptions should inherit from AgentError."""
    assert issubclass(StorageError, AgentError)
    assert issubclass(RetrievalError, AgentError)
    assert issubclass(LLMError, AgentError)
    assert issubclass(CompressionError, AgentError)
    assert issubclass(PromptBuildError, AgentError)


def test_exception_messages() -> None:
    """Exceptions should preserve error messages."""
    error = StorageError("Database unavailable")
    assert str(error) == "Database unavailable"
