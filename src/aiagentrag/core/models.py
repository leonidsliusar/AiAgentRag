"""Core domain models."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Role of a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single conversation message."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class MemoryItem(BaseModel):
    """A retrieved long-term memory entry."""

    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    content: str
    score: float


class KnowledgeChunk(BaseModel):
    """A retrieved knowledge base chunk."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """A message formatted for LLM consumption."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class AgentConfig(BaseModel):
    """Configuration for the AI agent."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str
    max_recent_messages: int = Field(default=20, ge=1)
    compression_threshold: int = Field(default=30, ge=2)
    knowledge_top_k: int = Field(default=5, ge=1)
    memory_top_k: int = Field(default=5, ge=1)
    knowledge_collection: str = Field(default="documents")
    user_memory_collection: str = Field(default="user_memory")


class VectorSearchResult(BaseModel):
    """Result from a vector similarity search."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinishedMetadata(BaseModel):
    """Metadata returned upon successful agent completion."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    message_count: int
    memories_retrieved: int
    knowledge_chunks_retrieved: int


class ErrorDetails(BaseModel):
    """Structured error information."""

    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
