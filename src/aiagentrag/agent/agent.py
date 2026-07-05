"""Agent orchestrator implementing the execution pipeline."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

from aiagentrag.core.events import (
    AgentEvent,
    ErrorEvent,
    FinishedEvent,
    StatusEvent,
    TokenEvent,
)
from aiagentrag.core.exceptions import AgentError
from aiagentrag.core.interfaces import (
    ConversationCompressorProtocol,
    KnowledgeRetrieverProtocol,
    LLMProvider,
    MemoryRepositoryProtocol,
    PromptBuilderProtocol,
)
from aiagentrag.core.models import AgentConfig, ErrorDetails, FinishedMetadata, Message, MessageRole


class Agent:
    """Orchestrates the AI agent execution pipeline."""

    def __init__(
        self,
        config: AgentConfig,
        memory_repository: MemoryRepositoryProtocol,
        knowledge_retriever: KnowledgeRetrieverProtocol,
        prompt_builder: PromptBuilderProtocol,
        llm_provider: LLMProvider,
        conversation_compressor: ConversationCompressorProtocol,
    ) -> None:
        """Initialize the agent with all required dependencies."""
        self._config = config
        self._memory_repository = memory_repository
        self._knowledge_retriever = knowledge_retriever
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._conversation_compressor = conversation_compressor

    async def run(self, user_id: str, message: str) -> AsyncIterator[AgentEvent]:
        """Execute the agent pipeline and stream events.

        Pipeline:
            1. Load recent messages
            2. Retrieve user memory
            3. Retrieve knowledge
            4. Build prompt
            5. Stream LLM response
            6. Persist conversation
            7. Compress history if needed
        """
        try:
            yield StatusEvent(message="Loading history")
            recent_messages = await self._memory_repository.get_recent_messages(user_id)

            yield StatusEvent(message="Retrieving memory")
            memories = await self._memory_repository.retrieve_memories(
                user_id=user_id,
                query=message,
            )

            yield StatusEvent(message="Retrieving knowledge")
            knowledge_chunks = await self._knowledge_retriever.retrieve(message)

            yield StatusEvent(message="Building prompt")
            prompt_messages = await self._prompt_builder.build(
                user_message=message,
                recent_messages=recent_messages,
                memories=memories,
                knowledge_chunks=knowledge_chunks,
            )

            yield StatusEvent(message="Calling LLM")
            full_response = ""
            async for token in self._llm_provider.stream(prompt_messages):
                full_response += token
                yield TokenEvent(content=token)

            await self._persist_conversation(
                user_id=user_id,
                user_message=message,
                assistant_response=full_response,
            )

            yield StatusEvent(message="Compressing history")
            await self._conversation_compressor.compress_if_needed(user_id)

            message_count = len(recent_messages) + 2
            yield FinishedEvent(
                response=full_response,
                metadata=FinishedMetadata(
                    user_id=user_id,
                    message_count=message_count,
                    memories_retrieved=len(memories),
                    knowledge_chunks_retrieved=len(knowledge_chunks),
                ),
            )
        except AgentError as exc:
            yield ErrorEvent(
                error=str(exc),
                details=ErrorDetails(
                    error_type=type(exc).__name__,
                    message=str(exc),
                ),
            )
        except Exception as exc:
            yield ErrorEvent(
                error=str(exc),
                details=ErrorDetails(
                    error_type=type(exc).__name__,
                    message=str(exc),
                ),
            )

    async def _persist_conversation(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Save the user message and assistant response to storage."""
        now = datetime.now(tz=UTC)
        user_msg = Message(
            id=uuid4(),
            user_id=user_id,
            role=MessageRole.USER,
            content=user_message,
            created_at=now,
        )
        assistant_msg = Message(
            id=uuid4(),
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=assistant_response,
            created_at=now,
        )
        await self._memory_repository.save_message(user_msg)
        await self._memory_repository.save_message(assistant_msg)
