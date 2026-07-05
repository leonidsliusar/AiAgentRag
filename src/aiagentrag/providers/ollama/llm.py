"""Ollama LLM provider implementation."""

from collections.abc import AsyncIterator

import ollama

from aiagentrag.core.exceptions import LLMError
from aiagentrag.core.models import LLMMessage


class OllamaLLMProvider:
    """Streams and completes text using a local Ollama instance."""

    def __init__(
        self,
        client: ollama.AsyncClient,
        model: str,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the Ollama LLM provider."""
        self._client = client
        self._model = model
        self._temperature = temperature

    async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream text tokens from Ollama."""
        formatted = [{"role": msg.role, "content": msg.content} for msg in messages]
        try:
            stream = await self._client.chat(
                model=self._model,
                messages=formatted,
                stream=True,
                options={"temperature": self._temperature},
            )
            async for chunk in stream:
                content = chunk.message.content
                if content:
                    yield content
        except Exception as exc:
            msg = f"Ollama streaming failed: {exc}"
            raise LLMError(msg) from exc

    async def complete(self, messages: list[LLMMessage]) -> str:
        """Generate a complete response from Ollama."""
        formatted = [{"role": msg.role, "content": msg.content} for msg in messages]
        try:
            chat_response = await self._client.chat(
                model=self._model,
                messages=formatted,
                stream=False,
                options={"temperature": self._temperature},
            )
            content = chat_response.message.content
            if not content:
                msg = "Ollama returned empty response"
                raise LLMError(msg)
            return content
        except LLMError:
            raise
        except Exception as exc:
            msg = f"Ollama completion failed: {exc}"
            raise LLMError(msg) from exc
