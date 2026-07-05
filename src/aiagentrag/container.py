"""Dependency injection container for wiring agent components."""

from dishka import AsyncContainer, Provider, Scope, make_async_container, provide
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from aiagentrag.agent.agent import Agent
from aiagentrag.core.models import AgentConfig
from aiagentrag.knowledge.retriever import KnowledgeRetriever
from aiagentrag.memory.compressor import ConversationCompressor
from aiagentrag.memory.repository import MemoryRepository
from aiagentrag.prompt.builder import PromptBuilder
from aiagentrag.storage.postgres.store import PostgresConversationStore
from aiagentrag.storage.qdrant.client import QdrantVectorStore


class AgentProvider(Provider):
    """Dishka provider that wires all agent dependencies."""

    def __init__(
        self,
        config: AgentConfig,
        embedding_provider: object,
        llm_provider: object,
        database_url: str,
        qdrant_url: str,
        vector_size: int,
    ) -> None:
        """Initialize the provider with configuration and connection settings."""
        super().__init__()
        self._config = config
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._database_url = database_url
        self._qdrant_url = qdrant_url
        self._vector_size = vector_size

    @provide(scope=Scope.APP)
    def config(self) -> AgentConfig:
        """Provide agent configuration."""
        return self._config

    @provide(scope=Scope.APP)
    def engine(self) -> AsyncEngine:
        """Provide the async database engine."""
        return create_async_engine(self._database_url)

    @provide(scope=Scope.APP)
    def session_factory(self, engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
        """Provide the async session factory."""
        return async_sessionmaker(engine, expire_on_commit=False)

    @provide(scope=Scope.APP)
    def conversation_store(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> PostgresConversationStore:
        """Provide the PostgreSQL conversation store."""
        return PostgresConversationStore(session_factory)

    @provide(scope=Scope.APP)
    def qdrant_client(self) -> AsyncQdrantClient:
        """Provide the Qdrant async client."""
        return AsyncQdrantClient(url=self._qdrant_url)

    @provide(scope=Scope.APP)
    def vector_store(self, qdrant_client: AsyncQdrantClient) -> QdrantVectorStore:
        """Provide the Qdrant vector store."""
        return QdrantVectorStore(qdrant_client, vector_size=self._vector_size)

    @provide(scope=Scope.APP)
    def embedding_provider(self) -> object:
        """Provide the embedding provider."""
        return self._embedding_provider

    @provide(scope=Scope.APP)
    def llm_provider(self) -> object:
        """Provide the LLM provider."""
        return self._llm_provider

    @provide(scope=Scope.APP)
    def memory_repository(
        self,
        conversation_store: PostgresConversationStore,
        vector_store: QdrantVectorStore,
        embedding_provider: object,
        config: AgentConfig,
    ) -> MemoryRepository:
        """Provide the memory repository."""
        return MemoryRepository(
            conversation_store=conversation_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,  # type: ignore[arg-type]
            config=config,
        )

    @provide(scope=Scope.APP)
    def knowledge_retriever(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: object,
        config: AgentConfig,
    ) -> KnowledgeRetriever:
        """Provide the knowledge retriever."""
        return KnowledgeRetriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,  # type: ignore[arg-type]
            config=config,
        )

    @provide(scope=Scope.APP)
    def prompt_builder(self, config: AgentConfig) -> PromptBuilder:
        """Provide the prompt builder."""
        return PromptBuilder(config)

    @provide(scope=Scope.APP)
    def conversation_compressor(
        self,
        conversation_store: PostgresConversationStore,
        vector_store: QdrantVectorStore,
        embedding_provider: object,
        llm_provider: object,
        config: AgentConfig,
    ) -> ConversationCompressor:
        """Provide the conversation compressor."""
        return ConversationCompressor(
            conversation_store=conversation_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,  # type: ignore[arg-type]
            llm_provider=llm_provider,  # type: ignore[arg-type]
            config=config,
        )

    @provide(scope=Scope.APP)
    def agent(
        self,
        config: AgentConfig,
        memory_repository: MemoryRepository,
        knowledge_retriever: KnowledgeRetriever,
        prompt_builder: PromptBuilder,
        llm_provider: object,
        conversation_compressor: ConversationCompressor,
    ) -> Agent:
        """Provide the fully wired agent."""
        return Agent(
            config=config,
            memory_repository=memory_repository,
            knowledge_retriever=knowledge_retriever,
            prompt_builder=prompt_builder,
            llm_provider=llm_provider,  # type: ignore[arg-type]
            conversation_compressor=conversation_compressor,
        )


def create_container(
    config: AgentConfig,
    embedding_provider: object,
    llm_provider: object,
    database_url: str,
    qdrant_url: str,
    vector_size: int,
) -> AsyncContainer:
    """Create a Dishka async container with all agent dependencies wired."""
    provider = AgentProvider(
        config=config,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        database_url=database_url,
        qdrant_url=qdrant_url,
        vector_size=vector_size,
    )
    return make_async_container(provider)
