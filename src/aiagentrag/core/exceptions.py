"""Domain-specific exceptions for the AI agent library."""


class AgentError(Exception):
    """Base exception for all agent-related errors."""


class StorageError(AgentError):
    """Raised when a storage operation fails."""


class RetrievalError(AgentError):
    """Raised when a retrieval operation fails."""


class LLMError(AgentError):
    """Raised when an LLM operation fails."""


class CompressionError(AgentError):
    """Raised when conversation compression fails."""


class PromptBuildError(AgentError):
    """Raised when prompt assembly fails."""
