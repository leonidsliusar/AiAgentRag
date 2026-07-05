"""Conversation compression into long-term memory."""

from uuid import uuid4

from aiagentrag.core.exceptions import CompressionError
from aiagentrag.core.interfaces import (
    ConversationStore,
    EmbeddingProvider,
    LLMProvider,
    VectorStore,
)
from aiagentrag.core.models import AgentConfig, LLMMessage, Message
from aiagentrag.storage.qdrant.schema import PAYLOAD_CONTENT, PAYLOAD_USER_ID

COMPRESSION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Summarize the following conversation concisely, "
    "preserving key facts, preferences, and context about the user. "
    "Write the summary in third person."
)


class ConversationCompressor:
    """Compresses old conversation messages into long-term memory."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        config: AgentConfig,
    ) -> None:
        """Initialize the conversation compressor with injected dependencies."""
        self._conversation_store = conversation_store
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._config = config

    async def compress_if_needed(self, user_id: str) -> bool:
        """Compress old messages when the threshold is exceeded.

        When the message count exceeds compression_threshold, the oldest
        messages (all except the most recent max_recent_messages) are
        summarized, indexed in user_memory, and deleted from PostgreSQL.
        """
        message_count = await self._conversation_store.count_messages(user_id)
        if message_count <= self._config.compression_threshold:
            return False

        all_messages = await self._conversation_store.get_all_messages(user_id)
        keep_count = self._config.max_recent_messages
        messages_to_compress = all_messages[:-keep_count] if keep_count > 0 else all_messages

        if not messages_to_compress:
            return False

        summary = await self._summarize(messages_to_compress)
        await self._save_to_memory(user_id=user_id, summary=summary)

        message_ids = [message.id for message in messages_to_compress]
        await self._conversation_store.delete_messages(message_ids)
        return True

    async def _summarize(self, messages: list[Message]) -> str:
        """Generate a summary of the given messages using the LLM."""
        conversation_text = self._format_conversation(messages)
        llm_messages = [
            LLMMessage(role="system", content=COMPRESSION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=conversation_text),
        ]
        try:
            return await self._llm_provider.complete(llm_messages)
        except Exception as exc:
            msg = f"Failed to summarize conversation: {exc}"
            raise CompressionError(msg) from exc

    async def _save_to_memory(self, user_id: str, summary: str) -> None:
        """Index the summary in the user_memory Qdrant collection."""
        point_id = str(uuid4())
        vector = await self._embedding_provider.embed(summary)
        try:
            await self._vector_store.upsert(
                collection=self._config.user_memory_collection,
                point_id=point_id,
                vector=vector,
                payload={PAYLOAD_USER_ID: user_id, PAYLOAD_CONTENT: summary},
            )
        except Exception as exc:
            msg = f"Failed to save compressed memory: {exc}"
            raise CompressionError(msg) from exc

    @staticmethod
    def _format_conversation(messages: list[Message]) -> str:
        """Format messages into a text block for summarization."""
        lines: list[str] = []
        for message in messages:
            role_label = message.role.value.capitalize()
            lines.append(f"{role_label}: {message.content}")
        return "\n".join(lines)
