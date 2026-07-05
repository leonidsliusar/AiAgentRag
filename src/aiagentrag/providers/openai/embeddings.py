"""OpenAI embedding provider implementation."""

from openai import AsyncOpenAI

from aiagentrag.core.exceptions import LLMError


class OpenAIEmbeddingProvider:
    """Generates vector embeddings using the OpenAI API."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        """Initialize the OpenAI embedding provider."""
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            msg = f"OpenAI embedding failed: {exc}"
            raise LLMError(msg) from exc
