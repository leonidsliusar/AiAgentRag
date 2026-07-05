"""Example Modal app for AIAgentRag RPC integration.

Deploy:
    modal deploy scripts/modal_llm_service.py

Environment for try_agent:
    export MODAL_APP_NAME=aiagentrag-llm
    export MODAL_LLM_CLS=LLMService
    poetry run python -m scripts.try_agent --provider modal -m "Hello"

Replace the method bodies with your own GPU inference logic.
This example does not use Ollama — only demonstrates the RPC contract.
"""

from __future__ import annotations

import modal

app = modal.App("aiagentrag-llm")

EMBED_DIM = 768


@app.cls()
class LLMService:
    """Modal class exposing LLM and embedding RPC methods for AIAgentRag."""

    @modal.method()
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a full assistant response for the given chat messages."""
        user_text = messages[-1]["content"] if messages else ""
        return f"Modal RPC reply to: {user_text}"

    @modal.method()
    def stream(self, messages: list[dict[str, str]]):
        """Yield streamed text chunks for the given chat messages."""
        response = self.complete(messages)
        for word in response.split():
            yield word + " "

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""
        return [[float(index) / EMBED_DIM for index in range(EMBED_DIM)] for _ in texts]
