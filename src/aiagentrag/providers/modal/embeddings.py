"""Modal RPC embedding provider."""

from aiagentrag.core.exceptions import LLMError
from aiagentrag.providers.modal.config import ModalRpcConfig
from aiagentrag.providers.modal.rpc import ModalRpcClient


class ModalEmbeddingProvider:
    """Generates vector embeddings via Modal RPC."""

    def __init__(self, rpc: ModalRpcClient, config: ModalRpcConfig) -> None:
        """Initialize the Modal embedding provider."""
        self._rpc = rpc
        self._config = config

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        if self._config.uses_embed_cls:
            assert self._config.embed_cls is not None
            result = await self._rpc.call_class_method(
                self._config.embed_cls,
                self._config.embed_method,
                texts,
            )
        else:
            assert self._config.embed_function is not None
            result = await self._rpc.call_function(self._config.embed_function, texts)

        if not isinstance(result, list):
            msg = f"Unexpected Modal embedding response type: {type(result).__name__}"
            raise LLMError(msg)

        embeddings: list[list[float]] = []
        for item in result:
            if not isinstance(item, list):
                msg = "Modal embedding response must be a list of float vectors"
                raise LLMError(msg)
            embeddings.append([float(value) for value in item])

        if len(embeddings) != len(texts):
            msg = "Modal embedding response length does not match input length"
            raise LLMError(msg)

        return embeddings
