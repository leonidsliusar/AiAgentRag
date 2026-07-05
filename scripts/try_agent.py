#!/usr/bin/env python3
"""Run the agent against real PostgreSQL, Qdrant, and Ollama.

Usage:
    # Load documents into Qdrant via vectorizer first, then:
    export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag"
    export QDRANT_URL="http://localhost:6333"
    export QDRANT_COLLECTION="documents"
    export OLLAMA_HOST="http://localhost:11434"

    poetry run python -m scripts.try_agent -m "Расскажи коротко про Python."
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from aiagentrag.agent.agent import Agent
from aiagentrag.core.events import ErrorEvent, FinishedEvent, StatusEvent, TokenEvent
from aiagentrag.core.models import AgentConfig
from aiagentrag.knowledge.retriever import KnowledgeRetriever
from aiagentrag.memory.compressor import ConversationCompressor
from aiagentrag.memory.repository import MemoryRepository
from aiagentrag.prompt.builder import PromptBuilder
from aiagentrag.storage.postgres.store import PostgresConversationStore
from aiagentrag.storage.qdrant.client import QdrantVectorStore
from aiagentrag.storage.qdrant.schema import DEFAULT_KNOWLEDGE_COLLECTION

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = DEFAULT_KNOWLEDGE_COLLECTION
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the AI agent with real infrastructure.")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--message",
        "-m",
        required=True,
        help="User message to send",
    )
    parser.add_argument(
        "--user-id",
        default="local-user",
        help="User identifier",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
        help="Qdrant HTTP URL",
    )
    parser.add_argument(
        "--knowledge-collection",
        default=os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        help="Qdrant collection populated by vectorizer (default: documents)",
    )
    return parser.parse_args()


async def build_agent(
    config: AgentConfig,
    provider: str,
    database_url: str,
    qdrant_url: str,
    knowledge_collection: str,
) -> Agent:
    """Build an agent wired to PostgreSQL, Qdrant, and the selected LLM provider."""
    if provider == "ollama":
        import ollama

        from aiagentrag.providers.ollama.embeddings import OllamaEmbeddingProvider
        from aiagentrag.providers.ollama.llm import OllamaLLMProvider

        host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        chat_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
        client = ollama.AsyncClient(host=host)
        llm = OllamaLLMProvider(client=client, model=chat_model)
        embedding = OllamaEmbeddingProvider(client=client, model=embed_model)
    else:
        from openai import AsyncOpenAI

        from aiagentrag.providers.openai.embeddings import OpenAIEmbeddingProvider
        from aiagentrag.providers.openai.llm import OpenAILLMProvider

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = "OPENAI_API_KEY is required for --provider openai"
            raise RuntimeError(msg)

        chat_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        embed_model = os.getenv("OPENAI_EMBED_MODEL", DEFAULT_OPENAI_EMBED_MODEL)
        client = AsyncOpenAI(api_key=api_key)
        llm = OpenAILLMProvider(client=client, model=chat_model)
        embedding = OpenAIEmbeddingProvider(client=client, model=embed_model)

    vector_size = len(await embedding.embed("dimension probe"))
    conversation_store = await PostgresConversationStore.initialize(database_url)

    from qdrant_client import AsyncQdrantClient

    qdrant_client = AsyncQdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(qdrant_client, vector_size=vector_size)

    if not await vector_store.collection_exists(knowledge_collection):
        msg = (
            f"Knowledge collection '{knowledge_collection}' was not found in Qdrant. "
            "Load documents via vectorizer before running the agent."
        )
        raise RuntimeError(msg)

    await vector_store.ensure_collection(config.user_memory_collection)

    return Agent(
        config=config,
        memory_repository=MemoryRepository(
            conversation_store,
            vector_store,
            embedding,
            config,
        ),
        knowledge_retriever=KnowledgeRetriever(vector_store, embedding, config),
        prompt_builder=PromptBuilder(config),
        llm_provider=llm,
        conversation_compressor=ConversationCompressor(
            conversation_store,
            vector_store,
            embedding,
            llm,
            config,
        ),
    )


async def run_agent(agent: Agent, user_id: str, message: str) -> int:
    """Run the agent and print streamed events. Returns exit code."""
    print(f"User ({user_id}): {message}\n")
    print("Agent: ", end="", flush=True)

    exit_code = 0
    async for event in agent.run(user_id=user_id, message=message):
        if isinstance(event, StatusEvent):
            print(f"\n[{event.message}]", flush=True)
            print("Agent: ", end="", flush=True)
        elif isinstance(event, TokenEvent):
            print(event.content, end="", flush=True)
        elif isinstance(event, FinishedEvent):
            print("\n")
            print(
                f"Done. Messages in context: {event.metadata.message_count}, "
                f"memories: {event.metadata.memories_retrieved}, "
                f"knowledge: {event.metadata.knowledge_chunks_retrieved}",
            )
        elif isinstance(event, ErrorEvent):
            print(f"\nError: {event.error}", file=sys.stderr)
            exit_code = 1

    return exit_code


async def main() -> int:
    """Entry point."""
    args = parse_args()
    config = AgentConfig(
        system_prompt=(
            "You are a helpful assistant. "
            "Answer concisely using provided knowledge and memory when relevant."
        ),
        max_recent_messages=10,
        compression_threshold=20,
        knowledge_collection=args.knowledge_collection,
    )

    agent = await build_agent(
        config=config,
        provider=args.provider,
        database_url=args.database_url,
        qdrant_url=args.qdrant_url,
        knowledge_collection=args.knowledge_collection,
    )

    return await run_agent(agent, user_id=args.user_id, message=args.message)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
