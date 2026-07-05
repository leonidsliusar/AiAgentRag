"""Knowledge base retrieval."""

from aiagentrag.core.interfaces import EmbeddingProvider, VectorStore
from aiagentrag.core.models import AgentConfig, KnowledgeChunk
from aiagentrag.storage.qdrant.schema import DEFAULT_KNOWLEDGE_COLLECTION

KNOWLEDGE_COLLECTION = DEFAULT_KNOWLEDGE_COLLECTION


class KnowledgeRetriever:
    """Retrieves relevant chunks from the knowledge base."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        config: AgentConfig,
    ) -> None:
        """Initialize the knowledge retriever with injected dependencies."""
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._config = config

    async def retrieve(self, query: str) -> list[KnowledgeChunk]:
        """Retrieve knowledge chunks relevant to the query."""
        query_vector = await self._embedding_provider.embed(query)
        results = await self._vector_store.search(
            collection=self._config.knowledge_collection,
            query_vector=query_vector,
            top_k=self._config.knowledge_top_k,
        )
        return [
            KnowledgeChunk(
                id=result.id,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
            )
            for result in results
        ]
