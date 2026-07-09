# AI Agent Development Guide

## Project Goal

Develop a reusable Python library implementing an AI agent with Retrieval-Augmented Generation (RAG), streaming responses and long-term memory.

The library must not depend on any transport layer such as Telegram, Discord, FastAPI or CLI.

The project must expose a clean, stable and reusable public API.

---

## Responsibilities

The agent is responsible for:

- loading recent conversation history
- retrieving long-term memory
- retrieving knowledge via RAG
- building prompts
- streaming LLM responses
- persisting conversations
- compressing old conversations
- exposing events during execution

---

## Architecture Principles

- Clean Architecture
- SOLID
- Dependency Injection
- Composition over inheritance

Business logic must never depend on infrastructure.

---

## External Systems

Infrastructure implementations may include:

- PostgreSQL
- Qdrant
- OpenAI
- Ollama

Business logic must communicate only through interfaces.

---

## Development Workflow

Before implementing any feature:

1. Read architecture.md
2. Read project_structure.md
3. Read tech_stack.md
4. Read public_api.md
5. Read prompts.md
6. Follow coding_rules.md
7. Complete milestones sequentially

Never skip milestones.

---

## Code Quality

Every generated file must:

- pass Ruff
- pass mypy
- include type hints
- include docstrings
- contain no TODOs
- contain no placeholder implementations
- be production ready

Always prefer readability over cleverness.
