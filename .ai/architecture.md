# Architecture

The project is a reusable AI Agent library.

Modules:

- Agent
- Prompt Builder
- Memory
- Knowledge Retrieval
- LLM
- Compression
- Storage
- Providers

---

Execution Pipeline

User Message

↓

Load Recent Messages

↓

Retrieve User Memory

↓

Retrieve Knowledge

↓

Build Prompt

↓

LLM Streaming

↓

Persist Conversation

↓

Compress History (optional)

---

Dependency Rule

Domain

↑

Application

↑

Infrastructure

Infrastructure must never be imported inside Domain.

The Agent orchestrates the flow but does not know implementation details.

All external systems are injected.