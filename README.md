# AIAgentRag

Reusable Python library: AI agent with RAG, streaming responses, and long-term memory.

Transport-independent — connect to Telegram, FastAPI, Discord, CLI, etc.  
**Works with [vectorizer](https://github.com/leonidsliusar/vectorizer)**: load documents into Qdrant via vectorizer, run this agent against the same Qdrant — RAG works immediately.

## Local run (step by step)

Everything below assumes you are in the project root after `poetry install`.

### What you need running

| Service | Port | How to start |
|---------|------|--------------|
| PostgreSQL | 5432 | `docker compose up -d` (included) |
| Qdrant | 6333 | `docker compose up -d` (included) |
| Ollama | 11434 | Already on your machine (same as vectorizer), or `docker compose --profile ollama up -d` |
| vectorizer data | — | Documents already loaded into Qdrant collection `documents` |

### Step 1 — Install

```bash
git clone https://github.com/leonidsliusar/AIAgentRag
cd AIAgentRag
poetry install
```

### Step 2 — Start PostgreSQL and Qdrant

```bash
docker compose up -d
```

Docker Compose creates database `aiagentrag` automatically (`postgres` / `postgres`).

Check that services are up:

```bash
docker compose ps
curl -s http://localhost:6333/readyz
```

### Step 3 — Ollama models

Use the same Ollama instance as vectorizer. Pull models if missing:

```bash
ollama pull nomic-embed-text   # embeddings — must match vectorizer
ollama pull qwen3:8b          # chat model (or set OLLAMA_MODEL)
```

### Step 4 — Load documents via vectorizer

In you ingest documents into Qdrant (collection `documents`, embedding `nomic-embed-text`) before.

AIAgentRag reads that collection and **does not create or overwrite it**.  
If `documents` is empty or missing, the agent will fail with a clear error.

Verify collection exists:

```bash
curl -s http://localhost:6333/collections/documents | head
```

### Step 5 — Run the agent

One command (migrations run automatically inside the script):

```bash
poetry run python -m scripts.try_agent -m "<YOUR QUESTION HERE>"
```

Defaults match Docker Compose and vectorizer:

- PostgreSQL: `postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag`
- Qdrant: `http://localhost:6333`
- Collection: `documents`
- Ollama: `http://localhost:11434`

Expected output:

```
User (local-user): Ваш вопрос...

[Loading history]
[Retrieving memory]
[Retrieving knowledge]
[Building prompt]
[Calling LLM]
Agent: ... streamed answer ...

Done. Messages in context: 2, memories: 0, knowledge: 3
```

### Optional flags

```bash
poetry run python -m scripts.try_agent \
  -m "Hello!" \
  --user-id my-user \
  --knowledge-collection documents \
  --database-url "postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag" \
  --qdrant-url "http://localhost:6333"
```

OpenAI instead of Ollama:

```bash
export OPENAI_API_KEY=sk-...
poetry run python -m scripts.try_agent --provider openai -m "Hello"
```

### Environment variables (all optional if defaults fit)

```bash
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag"
export QDRANT_URL="http://localhost:6333"
export QDRANT_COLLECTION="documents"
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
```

| Flag | Default |
|------|---------|
| `--provider` | `ollama` |
| `--message`, `-m` | **required** |
| `--user-id` | `local-user` |
| `--database-url` | `DATABASE_URL` or postgres default |
| `--qdrant-url` | `QDRANT_URL` or `http://localhost:6333` |
| `--knowledge-collection` | `QDRANT_COLLECTION` or `documents` |

---

## Features

- Streaming LLM with events: `StatusEvent`, `TokenEvent`, `FinishedEvent`, `ErrorEvent`
- RAG from Qdrant collection `documents` (vectorizer-compatible schema)
- Long-term user memory in Qdrant collection `user_memory`
- Conversation compression
- Providers: OpenAI, Ollama
- Storage: PostgreSQL + Qdrant
- Embedded PostgreSQL migrations
- DI via Dishka (`create_container`)

## Requirements

- Python 3.13+
- PostgreSQL, Qdrant, Ollama (or OpenAI)
- Documents in Qdrant (via vectorizer)

## Installation

```bash
pip install aiagentrag / poetry add aiagentrag  # PyPI
poetry install                                  # from source
```

## Architecture

```
User message
    → PostgreSQL (recent history)
    → Qdrant user_memory (long-term memory)
    → Qdrant documents (RAG chunks from vectorizer)
    → prompt → LLM stream → save messages → compress old history
```

| Storage | Collection | Owner | Agent action |
|---------|------------|-------|--------------|
| PostgreSQL | `messages` | AIAgentRag | read/write |
| Qdrant | `documents` | vectorizer | **read only** |
| Qdrant | `user_memory` | AIAgentRag | read/write (auto-created) |

### vectorizer payload (shared)

| Field | Description |
|-------|-------------|
| `text` | Chunk text |
| `document_id` | Source document |
| `chunk_index` | Index in document |
| `chunk_id`, `pages`, `token_count` | optional metadata |

Embedding model must match vectorizer: **`nomic-embed-text`**.

```python
from aiagentrag import AgentConfig

config = AgentConfig(
    system_prompt="You are a helpful assistant.",
    knowledge_collection="documents",  # default
)
```

## Database migrations

For **`try_agent`** — migrations run automatically, nothing to do manually.

For your own app:

```bash
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag"
poetry run aiagentrag-migrate upgrade head
```

Or in Python:

```python
from aiagentrag.storage.postgres import PostgresConversationStore

store = await PostgresConversationStore.initialize(DATABASE_URL)
```

No `alembic.ini` or `alembic/` folder needed in consumer projects.

## Public API

```python
from aiagentrag import Agent, AgentConfig, TokenEvent, FinishedEvent

async for event in agent.run(user_id="user-1", message="Hello"):
    match event:
        case TokenEvent(content=token):
            print(token, end="", flush=True)
        case FinishedEvent(metadata=meta):
            print(meta.knowledge_chunks_retrieved)
```

Full wiring example: `scripts/try_agent.py`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Knowledge collection 'documents' was not found` | Run vectorizer ingest first; check `QDRANT_URL` |
| `connection refused` :5432 | `docker compose up -d` |
| `connection refused` :6333 | `docker compose up -d` |
| `Ollama streaming failed` | `ollama serve`, then `ollama pull llama3.2` |
| `Ollama embedding failed` | `ollama pull nomic-embed-text` |
| Bad RAG results | Same embed model as vectorizer; re-index if changed |

## Development

```bash
poetry run ruff check src tests scripts
poetry run mypy src
poetry run pytest
```

Test fakes live in `tests/conftest.py` only — not used for local runs.
