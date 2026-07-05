"""Modal RPC LLM provider."""

from collections.abc import AsyncIterator

from aiagentrag.core.exceptions import LLMError
from aiagentrag.core.models import LLMMessage
from aiagentrag.providers.modal.config import ModalRpcConfig
from aiagentrag.providers.modal.rpc import ModalRpcClient


def _serialize_messages(messages: list[LLMMessage]) -> list[dict[str, str]]:
    """Convert LLM messages to JSON-serializable dicts for Modal RPC."""
    return [{"role": message.role, "content": message.content} for message in messages]


def _coerce_token(chunk: object) -> str:
    """Normalize a streamed RPC chunk to a text token."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        content = chunk.get("content") or chunk.get("text")
        if content is not None:
            return str(content)
    msg = f"Unexpected Modal stream chunk type: {type(chunk).__name__}"
    raise LLMError(msg)


class ModalLLMProvider:
    """Streams and completes text via Modal RPC."""

    def __init__(self, rpc: ModalRpcClient, config: ModalRpcConfig) -> None:
        """Initialize the Modal LLM provider."""
        self._rpc = rpc
        self._config = config

    async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream text tokens from a deployed Modal function or class method."""
        payload = _serialize_messages(messages)
        if self._config.uses_llm_cls:
            assert self._config.llm_cls is not None
            stream = self._rpc.call_class_method_gen(
                self._config.llm_cls,
                self._config.llm_stream_method,
                payload,
            )
        else:
            assert self._config.llm_stream_function is not None
            stream = self._rpc.call_function_gen(self._config.llm_stream_function, payload)

        async for chunk in stream:
            token = _coerce_token(chunk)
            if token:
                yield token

    async def complete(self, messages: list[LLMMessage]) -> str:
        """Generate a complete response via Modal RPC."""
        payload = _serialize_messages(messages)
        if self._config.uses_llm_cls:
            assert self._config.llm_cls is not None
            result = await self._rpc.call_class_method(
                self._config.llm_cls,
                self._config.llm_complete_method,
                payload,
            )
        else:
            assert self._config.llm_complete_function is not None
            result = await self._rpc.call_function(self._config.llm_complete_function, payload)

        if isinstance(result, str):
            if not result:
                msg = "Modal RPC returned empty LLM response"
                raise LLMError(msg)
            return result

        if isinstance(result, dict):
            content = result.get("content") or result.get("text")
            if content is not None:
                return str(content)

        msg = f"Unexpected Modal LLM response type: {type(result).__name__}"
        raise LLMError(msg)
