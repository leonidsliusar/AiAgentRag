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
| **LLM + embeddings** | Generation and vectorization | **Ollama**, **OpenAI API**, or **Modal RPC** |

For RAG:

- **`documents`** collection in Qdrant — created and populated by **vectorizer** (the agent is read-only).
- **`user_memory`** collection — created automatically by the agent.
- With Ollama: embedding model must match vectorizer (`nomic-embed-text`).
- With Modal: your deployed Modal app must expose compatible `embed` RPC (see below).

Python **3.13+**.

---

## 2. Installation

### From PyPI

```bash
pip install aiagentrag
pip install "aiagentrag[modal]"   # Modal RPC provider
```

```bash
poetry add aiagentrag
poetry add aiagentrag --extras modal
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
| `--provider` | `ollama` | `ollama`, `openai`, or `modal` |
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
| `MODAL_TOKEN_ID` | — (required unless set in `~/.modal.toml`) |
| `MODAL_TOKEN_SECRET` | — (required unless set in `~/.modal.toml`) |
| `MODAL_APP_NAME` | — (required for `--provider modal`) |
| `MODAL_ENVIRONMENT` | — (optional Modal environment) |
| `MODAL_LLM_CLS` | — (Modal class with `complete` / `stream` methods) |
| `MODAL_LLM_COMPLETE_METHOD` | `complete` |
| `MODAL_LLM_STREAM_METHOD` | `stream` |
| `MODAL_EMBED_CLS` | same as `MODAL_LLM_CLS` |
| `MODAL_EMBED_METHOD` | `embed` |
| `MODAL_LLM_COMPLETE_FUNCTION` | — (alternative to class: deployed Function name) |
| `MODAL_LLM_STREAM_FUNCTION` | — |
| `MODAL_EMBED_FUNCTION` | — |

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

### Modal RPC (full setup)

Modal calls your **deployed** app over RPC. You need **credentials**, a **deployed app**, and **target names**.

#### Step 1 — Modal account and token

1. Register at [modal.com](https://modal.com).
2. Create an API token at [modal.com/settings](https://modal.com/settings).

#### Step 2 — Authenticate locally (pick one)

**Option A — CLI** (writes `~/.modal.toml`):

```bash
pip install modal
modal token set --token-id ak-... --token-secret as-...
```

**Option B — environment variables** (CI, Docker, no config file):

```bash
export MODAL_TOKEN_ID=ak-...
export MODAL_TOKEN_SECRET=as-...
```

Verify:

```bash
modal profile current
```

#### Step 3 — Deploy the LLM service on Modal

Uses the same credentials as above:

```bash
poetry install --extras modal
modal deploy scripts/modal_llm_service.py
```

This deploys app `aiagentrag-llm` with class `LLMService`. Replace method bodies in that file with your real inference code.

#### Step 4 — Run the agent (local machine or server)

Still needs PostgreSQL, Qdrant, and vectorizer data — only LLM/embeddings go through Modal RPC:

```bash
export MODAL_TOKEN_ID=ak-...          # skip if already in ~/.modal.toml
export MODAL_TOKEN_SECRET=as-...      # skip if already in ~/.modal.toml
export MODAL_APP_NAME=aiagentrag-llm  # must match deployed app name
export MODAL_LLM_CLS=LLMService       # must match @app.cls class name

poetry run python -m scripts.try_agent --provider modal -m "Hello"
```

If credentials are missing, the script fails immediately with a clear error (before any RPC call).

#### Modal RPC contract

Modal is used as **RPC** to your deployed app — not HTTP, not Ollama.

Your deployed Modal app must expose these RPC endpoints (see [`scripts/modal_llm_service.py`](scripts/modal_llm_service.py)).

**Class-based** (typical for GPU models with `@modal.cls`):

| Method | Input | Output |
|--------|-------|--------|
| `complete` | `list[dict]` with `role`, `content` | `str` (or `{"content": "..."}`) |
| `stream` | same | generator yielding `str` tokens |
| `embed` | `list[str]` | `list[list[float]]` |

**Function-based** alternative: set `MODAL_LLM_COMPLETE_FUNCTION`, `MODAL_LLM_STREAM_FUNCTION`, `MODAL_EMBED_FUNCTION` instead of `MODAL_LLM_CLS`.

Client-side usage:

```python
from aiagentrag.providers.modal import ModalRpcClient, ModalRpcConfig, ModalLLMProvider, ModalEmbeddingProvider

config = ModalRpcConfig(app_name="aiagentrag-llm", llm_cls="LLMService", embed_cls="LLMService")
rpc = ModalRpcClient(config.app_name)
llm = ModalLLMProvider(rpc, config)
embedding = ModalEmbeddingProvider(rpc, config)
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

### Upgrade migrations
```bash
aiagentrag-migrate upgrade head
```

### Ollama (local Ollama server)

```python
from aiagentrag.core.models import AgentConfig
from aiagentrag.agent.agent import Agent
from aiagentrag.providers.ollama import OllamaEmbeddingProvider, OllamaLLMProvider
from ollama import AsyncClient

ollama_client = AsyncClient(base_url="http://localhost:11434")
embedding = OllamaEmbeddingProvider(ollama_client, model="nomic-embed-text")
llm = OllamaLLMProvider(ollama_client, model="qwen3:8b")
cfg = AgentConfig(system_prompt="You are a helpful assistant.")
agent = Agent.from_ollama(
    cfg,
    ollama_client=llm,
    embedding_provider=embedding,
    database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/aiagentrag",
    qdrant_url="http://localhost:6333",
)
```
### OpenAI (official async client)

```python
from aiagentrag.providers.openai import OpenAIEmbeddingProvider, OpenAILLMProvider
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key="sk-...")
embedding = OpenAIEmbeddingProvider(openai_client, model="text-embedding-3-small")
llm = OpenAILLMProvider(openai_client, model="gpt-4o-mini")

agent = Agent.from_openai(
    cfg,
    openai_client=llm,
    embedding_provider=embedding,
    database_url=DATABASE_URL,
    qdrant_url=QDRANT_URL,
)
```

### Modal RPC (deployed Modal app)

```python
from aiagentrag.providers.modal import (
    ModalRpcConfig,
    ModalRpcClient,
    ModalEmbeddingProvider,
    ModalLLMProvider,
)

rpc_cfg = ModalRpcConfig(
    app_name="aiagentrag-llm",
    llm_cls="LLMService",
    embed_cls="LLMService",
)
rpc = ModalRpcClient(app_name=rpc_cfg.app_name, environment_name=rpc_cfg.environment_name)

embedding = ModalEmbeddingProvider(rpc, rpc_cfg)
llm = ModalLLMProvider(rpc, rpc_cfg)

agent = Agent.from_modal(
    cfg,
    modal_client=llm,
    embedding_provider=embedding,
    database_url=DATABASE_URL,
    qdrant_url=QDRANT_URL,
)
```

### Client call
```python
async def get_answer(user_id: int, message: str) -> AsyncGenerator[str, None]:
   await agent.init()
   async for event in agent.run(user_id=str(user_id), message=message):
        match event:
            case TokenEvent(content=tok):
                yield tok
            case FinishedEvent(metadata=meta):
                print("\nDone:", meta)
                raise StopIteration
            case ErrorEvent(error=err):
                raise RuntimeError(err)
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

poetry install --extras modal

docker compose up -d    # PostgreSQL + Qdrant
# LLM: Ollama, OpenAI, or Modal RPC; documents via vectorizer

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
| Poor RAG quality | Same embedding vectors as used when ingesting documents |
| Modal RPC failed | Credentials set (`modal token set` or `MODAL_TOKEN_*`), app deployed (`modal deploy`) |
| `Token missing` | Set `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` or run `modal token set` |
