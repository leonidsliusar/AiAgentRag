# Public API

The library exposes a transport-independent API.

Example:

```python
agent = Agent(...)

async for event in agent.run(
    user_id=user_id,
    message=message,
):
    ...
```

---

Execution

The run() method returns an asynchronous stream of events.

The caller decides how to present them.

Examples:

- Telegram
- Discord
- FastAPI
- CLI
- WebSocket

---

Event Types

StatusEvent

Reports execution progress.

Examples:

- Loading history
- Retrieving knowledge
- Building prompt
- Calling LLM

---

TokenEvent

Represents a streamed text fragment from the LLM.

---

FinishedEvent

Signals successful completion.

Contains the final response metadata.

---

ErrorEvent

Signals execution failure.

Contains structured error information.

---

Design Rules

The public API must remain stable.

Transport-specific logic must never appear inside the library.

The library must expose interfaces instead of implementations whenever possible.
