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
    EmbeddingProvider,
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

    @classmethod
    def from_components(
        cls,
        config: AgentConfig,
        memory_repository: MemoryRepositoryProtocol,
        knowledge_retriever: KnowledgeRetrieverProtocol,
        prompt_builder: PromptBuilderProtocol,
        llm_provider: LLMProvider,
        conversation_compressor: ConversationCompressorProtocol,
    ) -> "Agent":
        """Create an Agent synchronously from already-instantiated components."""
        return cls(
            config=config,
            memory_repository=memory_repository,
            knowledge_retriever=knowledge_retriever,
            prompt_builder=prompt_builder,
            llm_provider=llm_provider,
            conversation_compressor=conversation_compressor,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        database_url: str,
        qdrant_url: str,
    ) -> "Agent":
        """Convenience constructor that wires default implementations synchronously.

        Important: this constructor does NOT run database migrations. Use the
        PostgresConversationStore.initialize(...) helper elsewhere when you need
        migrations to be applied.
        """
        # Local imports to avoid heavy dependencies at package import time.
        from qdrant_client import AsyncQdrantClient
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from aiagentrag.knowledge.retriever import KnowledgeRetriever
        from aiagentrag.memory.compressor import ConversationCompressor
        from aiagentrag.memory.repository import MemoryRepository
        from aiagentrag.prompt.builder import PromptBuilder
        from aiagentrag.storage.postgres.store import PostgresConversationStore
        from aiagentrag.storage.qdrant.client import QdrantVectorStore

        # Create async engine and session factory but DO NOT run migrations here.
        engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        conversation_store = PostgresConversationStore(session_factory)

        # Create Qdrant client and vector store (synchronous client construction).
        # Use a safe default vector size here; `agent.init()` will probe the
        # real embedding dimensionality and overwrite the store's `_vector_size`.
        qdrant_client = AsyncQdrantClient(url=qdrant_url)
        vector_store = QdrantVectorStore(qdrant_client)

        # Wire repository and other components.
        memory_repo = MemoryRepository(
            conversation_store=conversation_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            config=config,
        )
        knowledge = KnowledgeRetriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            config=config,
        )
        prompt = PromptBuilder(config)
        compressor = ConversationCompressor(
            conversation_store=conversation_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            config=config,
        )

        return cls.from_components(
            config=config,
            memory_repository=memory_repo,
            knowledge_retriever=knowledge,
            prompt_builder=prompt,
            llm_provider=llm_provider,
            conversation_compressor=compressor,
        )

    @classmethod
    def from_openai(
        cls,
        config: AgentConfig,
        *,
        openai_client: LLMProvider,
        embedding_provider: EmbeddingProvider,
        database_url: str,
        qdrant_url: str,
    ) -> "Agent":
        """Create an Agent wired for OpenAI-based providers.

        This is a thin wrapper around `from_config` that accepts an OpenAI client
        instance for the LLM and reuses the provided embedding provider.
        """
        # The openai_client is expected to implement the LLMProvider protocol.
        return cls.from_config(
            config,
            embedding_provider=embedding_provider,
            llm_provider=openai_client,
            database_url=database_url,
            qdrant_url=qdrant_url,
        )

    @classmethod
    def from_ollama(
        cls,
        config: AgentConfig,
        *,
        ollama_client: LLMProvider,
        embedding_provider: EmbeddingProvider,
        database_url: str,
        qdrant_url: str,
    ) -> "Agent":
        """Create an Agent wired for Ollama-based providers."""
        return cls.from_config(
            config,
            embedding_provider=embedding_provider,
            llm_provider=ollama_client,
            database_url=database_url,
            qdrant_url=qdrant_url,
        )

    @classmethod
    def from_modal(
        cls,
        config: AgentConfig,
        *,
        modal_client: LLMProvider,
        embedding_provider: EmbeddingProvider,
        database_url: str,
        qdrant_url: str,
    ) -> "Agent":
        """Create an Agent wired for Modal-based providers."""
        return cls.from_config(
            config,
            embedding_provider=embedding_provider,
            llm_provider=modal_client,
            database_url=database_url,
            qdrant_url=qdrant_url,
        )

    async def init(self) -> None:
        """Perform async initialization tasks.

        This method is responsible for:
        - probing the embedding provider to determine vector dimensionality,
        - setting the vector size on all Qdrant vector stores,
        - creating the knowledge and user-memory collections when missing,
        - raising clear AgentError on any failure.
        """
        try:
            # Resolve embedding provider from wired components.
            embedding = getattr(self._memory_repository, "_embedding_provider", None)
            if embedding is None:
                embedding = getattr(self._knowledge_retriever, "_embedding_provider", None)

            memory_vector_store = getattr(self._memory_repository, "_vector_store", None)
            knowledge_vector_store = getattr(self._knowledge_retriever, "_vector_store", None)

            if embedding is None:
                raise AgentError("Embedding provider is not available for probing vector size.")

            # Probe embedding dimensionality and set it on vector stores.
            probe = await embedding.embed("dimension probe")
            if not isinstance(probe, list) or len(probe) == 0:
                raise AgentError("Failed to determine embedding dimensionality from provider.")
            vec_size = len(probe)

            for vs in (memory_vector_store, knowledge_vector_store):
                if vs is None:
                    continue
                try:
                    vs._vector_size = vec_size
                except Exception as exc:
                    raise AgentError(f"Failed to set vector size on vector store: {exc}") from exc

            # Create knowledge collection if missing.
            knowledge_collection = self._config.knowledge_collection
            if knowledge_vector_store is not None:
                await knowledge_vector_store.ensure_collection(knowledge_collection)

            # Create user memory collection if missing.
            user_mem_col = self._config.user_memory_collection
            if memory_vector_store is not None:
                await memory_vector_store.ensure_collection(user_mem_col)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Agent initialization failed: {exc}") from exc

    # Note: vector size is determined and applied during `init()`; no separate setter.

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
