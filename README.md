# AIAgentRag

Python library for an AI agent with RAG, streaming responses, and long-term memory.  
Transport-independent — wire it into Telegram, FastAPI, CLI, or anything else on your side.

Compatible with [vectorizer](https://github.com/leonidsliusar/vectorizer): load documents into Qdrant via vectorizer; the agent reads the same `documents` collection.

---

## 1. Dependencies

The library does **not** start external services. You need:

| Service | Purpose | Typical setup |
|---------|---------|---------------|
| **PostgreSQL** | Conversation history | `docker compose up -d` in this repo, or your own instance |
| **Qdrant** | RAG + long-term memory | `docker compose up -d` in this repo, or your own instance |
| **LLM + embeddings** | Generation and vectorization | **Ollama** (local) or **OpenAI API** |

For RAG:

- **`documents`** collection in Qdrant — created and populated by **vectorizer** (the agent is read-only).
- **`user_memory`** collection — created automatically by the agent.
- Embedding model must match vectorizer: **`nomic-embed-text`** (Ollama).

Python **3.13+**.

---

## 2. Installation

### From PyPI

```bash
pip install aiagentrag
```

```bash
poetry add aiagentrag
```

### From source (development)

```bash
git clone https://github.com/leonidsliusar/AIAgentRag
cd AIAgentRag
poetry install
```

---

## 3. Local script (`scripts/try_agent`)

Script to test the agent against real infrastructure. Lives in this repository only — not shipped on PyPI.

### One-time setup

```bash
docker compose up -d          # PostgreSQL :5432 + Qdrant :6333
ollama pull nomic-embed-text  # embeddings (same as vectorizer)
ollama pull qwen3:8b          # chat model (or set OLLAMA_MODEL)
# ingest documents into Qdrant via vectorizer (collection: documents)
```

### Run

From the repository root:

```bash
poetry run python -m scripts.try_agent -m "Your question"
```

PostgreSQL migrations run automatically inside the script — no manual step required.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--message` | — | **Required.** User message |
| `--provider` | `ollama` | `ollama` or `openai` |
| `--user-id` | `local-user` | User identifier |
| `--database-url` | see below | PostgreSQL URL |
| `--qdrant-url` | see below | Qdrant URL |
| `--knowledge-collection` | see below | Document collection (vectorizer) |

### Environment variables

Used when the corresponding flag is not passed:

| Variable | Default |
|----------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag` |
| `QDRANT_URL` | `http://localhost:6333` |
| `QDRANT_COLLECTION` | `documents` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen3:8b` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` |
| `OPENAI_API_KEY` | — (required for `--provider openai`) |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` |

### Examples

```bash
# minimal run
poetry run python -m scripts.try_agent -m "What do the documents say about X?"

# explicit URLs
poetry run python -m scripts.try_agent \
  -m "Hello" \
  --database-url "postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag" \
  --qdrant-url "http://localhost:6333" \
  --knowledge-collection documents

# OpenAI
export OPENAI_API_KEY=sk-...
poetry run python -m scripts.try_agent --provider openai -m "Hello"
```

---

## 4. PostgreSQL migrations

Migrations are embedded in the package. Consumer projects do **not** need `alembic.ini`, an `alembic/` folder, or manual SQL.

The PostgreSQL **database** must exist beforehand; the library creates tables.

### In this repository

```bash
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag"
poetry run aiagentrag-migrate upgrade head
```

Check status:

```bash
poetry run aiagentrag-migrate current
poetry run aiagentrag-migrate history
```

### In a project with the installed package

**CLI** (after `pip install aiagentrag`):

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/mydb"
aiagentrag-migrate upgrade head
aiagentrag-migrate current
aiagentrag-migrate downgrade base   # rollback
```

**Python API:**

```python
from aiagentrag.storage.postgres import upgrade_head, PostgresConversationStore

DATABASE_URL = "postgresql+asyncpg://user:pass@host:5432/mydb"

# migrations only
upgrade_head(DATABASE_URL)

# migrations + ready-to-use store
store = await PostgresConversationStore.initialize(DATABASE_URL)
```

`upgrade_head` is idempotent — safe to call on every application startup.

---

## 5. Using the library in your project

The library does not start infrastructure. You connect PostgreSQL, Qdrant, and an LLM provider, build an `Agent`, and call `run()`.

Full wiring example: [`scripts/try_agent.py`](scripts/try_agent.py).

Minimal outline:

```python
import asyncio

from aiagentrag import Agent, AgentConfig, TokenEvent, FinishedEvent, ErrorEvent
from aiagentrag.knowledge.retriever import KnowledgeRetriever
from aiagentrag.memory.compressor import ConversationCompressor
from aiagentrag.memory.repository import MemoryRepository
from aiagentrag.prompt.builder import PromptBuilder
from aiagentrag.storage.postgres import PostgresConversationStore
from aiagentrag.storage.qdrant.client import QdrantVectorStore

# + your LLM provider (Ollama / OpenAI)
# + QdrantVectorStore, embedding provider


async def main() -> None:
    config = AgentConfig(
        system_prompt="You are a helpful assistant.",
        knowledge_collection="documents",  # vectorizer collection
    )

    database_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/mydb"
    conversation_store = await PostgresConversationStore.initialize(database_url)

    # vector_store, embedding, llm — see scripts/try_agent.py

    agent = Agent(
        config=config,
        memory_repository=MemoryRepository(conversation_store, vector_store, embedding, config),
        knowledge_retriever=KnowledgeRetriever(vector_store, embedding, config),
        prompt_builder=PromptBuilder(config),
        llm_provider=llm,
        conversation_compressor=ConversationCompressor(
            conversation_store, vector_store, embedding, llm, config
        ),
    )

    async for event in agent.run(user_id="user-1", message="Hello"):
        match event:
            case TokenEvent(content=token):
                print(token, end="", flush=True)
            case FinishedEvent(metadata=meta):
                print(f"\nchunks: {meta.knowledge_chunks_retrieved}")
            case ErrorEvent(error=err):
                print(f"Error: {err}")


asyncio.run(main())
```

### DI (optional)

```python
from aiagentrag import create_container

container = create_container(
    config=config,
    embedding_provider=embedding,
    llm_provider=llm,
    database_url=database_url,
    qdrant_url="http://localhost:6333",
    vector_size=768,
)
```

### Events

`agent.run()` yields an async stream:

| Event | When |
|-------|------|
| `StatusEvent` | Pipeline step (load history, RAG, LLM, …) |
| `TokenEvent` | Streamed response token |
| `FinishedEvent` | Success + metadata |
| `ErrorEvent` | Failure |

### Qdrant collections

| Collection | Written by | Read by |
|------------|------------|---------|
| `documents` | vectorizer | agent (RAG) |
| `user_memory` | agent | agent |

Document payload (vectorizer): `text` field plus metadata (`document_id`, `chunk_index`, …).

---

## 6. Local development

```bash
git clone https://github.com/leonidsliusar/AIAgentRag
cd AIAgentRag
poetry install

docker compose up -d    # PostgreSQL + Qdrant
# Ollama — local; documents — via vectorizer

poetry run python -m scripts.try_agent -m "test"

poetry run pytest
poetry run ruff check src tests scripts
poetry run mypy src
poetry build              # verify migrations are included in the wheel
```

In-memory test fakes live in `tests/conftest.py` only — not used for local runs.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Knowledge collection 'documents' was not found` | Ingest documents via vectorizer first |
| `connection refused` on :5432 / :6333 | `docker compose up -d` |
| Ollama streaming / embedding failed | `ollama serve`, `ollama pull nomic-embed-text`, `ollama pull qwen3:8b` |
| Poor RAG quality | Same embedding model as vectorizer (`nomic-embed-text`) |
