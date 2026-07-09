"""OpenAI LLM provider implementation."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI, AsyncStream

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aiagentrag.core.exceptions import LLMError
from aiagentrag.core.models import LLMMessage


class OpenAILLMProvider:
    """Streams and completes text using the OpenAI API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the OpenAI LLM provider."""
        self._client = client
        self._model = model
        self._temperature = temperature

    async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream text tokens from the OpenAI API."""
        formatted = [{"role": msg.role, "content": msg.content} for msg in messages]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted,  # type: ignore[arg-type]
                temperature=self._temperature,
                stream=True,
            )
            stream = cast("AsyncStream[ChatCompletionChunk]", response)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            msg = f"OpenAI streaming failed: {exc}"
            raise LLMError(msg) from exc

    async def complete(self, messages: list[LLMMessage]) -> str:
        """Generate a complete response from the OpenAI API."""
        formatted = [{"role": msg.role, "content": msg.content} for msg in messages]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted,  # type: ignore[arg-type]
                temperature=self._temperature,
                stream=False,
            )
            completion = cast("ChatCompletion", response)
            content = completion.choices[0].message.content
            if content is None:
                msg = "OpenAI returned empty response"
                raise LLMError(msg)
            return content
        except LLMError:
            raise
        except Exception as exc:
            msg = f"OpenAI completion failed: {exc}"
            raise LLMError(msg) from exc
