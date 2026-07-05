"""Event types emitted during agent execution."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from aiagentrag.core.models import ErrorDetails, FinishedMetadata


class StatusEvent(BaseModel):
    """Reports execution progress."""

    model_config = ConfigDict(frozen=True)

    type: Literal["status"] = "status"
    message: str


class TokenEvent(BaseModel):
    """Represents a streamed text fragment from the LLM."""

    model_config = ConfigDict(frozen=True)

    type: Literal["token"] = "token"
    content: str


class FinishedEvent(BaseModel):
    """Signals successful completion with final response metadata."""

    model_config = ConfigDict(frozen=True)

    type: Literal["finished"] = "finished"
    response: str
    metadata: FinishedMetadata


class ErrorEvent(BaseModel):
    """Signals execution failure with structured error information."""

    model_config = ConfigDict(frozen=True)

    type: Literal["error"] = "error"
    error: str
    details: ErrorDetails | None = None


AgentEvent = StatusEvent | TokenEvent | FinishedEvent | ErrorEvent

__all__ = [
    "AgentEvent",
    "ErrorEvent",
    "FinishedEvent",
    "StatusEvent",
    "TokenEvent",
]
