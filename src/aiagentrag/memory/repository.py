"""Memory repository for conversation history and long-term memory retrieval."""

from aiagentrag.core.interfaces import ConversationStore, EmbeddingProvider, VectorStore
from aiagentrag.core.models import AgentConfig, MemoryItem, Message
from aiagentrag.storage.qdrant.schema import DEFAULT_USER_MEMORY_COLLECTION, PAYLOAD_USER_ID

USER_MEMORY_COLLECTION = DEFAULT_USER_MEMORY_COLLECTION


class MemoryRepository:
    """Manages active conversation storage and long-term memory retrieval."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        config: AgentConfig,
    ) -> None:
        """Initialize the memory repository with injected dependencies."""
        self._conversation_store = conversation_store
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._config = config

    async def get_recent_messages(self, user_id: str) -> list[Message]:
        """Load recent messages from active conversation storage."""
        return await self._conversation_store.get_recent_messages(
            user_id=user_id,
            limit=self._config.max_recent_messages,
        )

    async def retrieve_memories(self, user_id: str, query: str) -> list[MemoryItem]:
        """Retrieve relevant long-term memories for a user."""
        query_vector = await self._embedding_provider.embed(query)
        results = await self._vector_store.search(
            collection=self._config.user_memory_collection,
            query_vector=query_vector,
            top_k=self._config.memory_top_k,
            filter_payload={PAYLOAD_USER_ID: user_id},
        )
        return [
            MemoryItem(
                id=result.id,
                user_id=user_id,
                content=result.content,
                score=result.score,
            )
            for result in results
        ]

    async def save_message(self, message: Message) -> Message:
        """Persist a message to active conversation storage."""
        return await self._conversation_store.save_message(message)
