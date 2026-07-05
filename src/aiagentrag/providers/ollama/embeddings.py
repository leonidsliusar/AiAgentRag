"""Ollama embedding provider implementation."""

import ollama

from aiagentrag.core.exceptions import LLMError


class OllamaEmbeddingProvider:
    """Generates vector embeddings using a local Ollama instance."""

    def __init__(self, client: ollama.AsyncClient, model: str) -> None:
        """Initialize the Ollama embedding provider."""
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        try:
            embeddings: list[list[float]] = []
            for text in texts:
                response = await self._client.embeddings(model=self._model, prompt=text)
                embedding = response.get("embedding")
                if not embedding:
                    msg = "Ollama returned empty embedding"
                    raise LLMError(msg)
                embeddings.append(embedding)
            return embeddings
        except LLMError:
            raise
        except Exception as exc:
            msg = f"Ollama embedding failed: {exc}"
            raise LLMError(msg) from exc
