"""Assembles structured prompts for LLM consumption."""

from aiagentrag.core.models import (
    AgentConfig,
    KnowledgeChunk,
    LLMMessage,
    MemoryItem,
    Message,
    MessageRole,
)


class PromptBuilder:
    """Builds deterministic, consistently formatted LLM prompts."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the prompt builder with agent configuration."""
        self._config = config

    async def build(
        self,
        user_message: str,
        recent_messages: list[Message],
        memories: list[MemoryItem],
        knowledge_chunks: list[KnowledgeChunk],
    ) -> list[LLMMessage]:
        """Build the complete prompt message list.

        Structure:
            1. System prompt with memory and knowledge context
            2. Recent conversation history
            3. Current user message
        """
        system_content = self._build_system_content(memories, knowledge_chunks)
        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_content)]

        for message in recent_messages:
            messages.append(
                LLMMessage(role=message.role.value, content=message.content),
            )

        messages.append(LLMMessage(role=MessageRole.USER.value, content=user_message))
        return messages

    def _build_system_content(
        self,
        memories: list[MemoryItem],
        knowledge_chunks: list[KnowledgeChunk],
    ) -> str:
        """Assemble the system prompt with optional context sections."""
        sections = [self._config.system_prompt]

        memory_section = self._format_memories(memories)
        if memory_section:
            sections.append(memory_section)

        knowledge_section = self._format_knowledge(knowledge_chunks)
        if knowledge_section:
            sections.append(knowledge_section)

        return "\n\n".join(sections)

    def _format_memories(self, memories: list[MemoryItem]) -> str:
        """Format retrieved long-term memories into a prompt section."""
        if not memories:
            return ""

        lines = ["## Long-term Memory", ""]
        for index, memory in enumerate(memories, start=1):
            lines.append(f"{index}. {memory.content}")
        return "\n".join(lines)

    def _format_knowledge(self, knowledge_chunks: list[KnowledgeChunk]) -> str:
        """Format retrieved knowledge chunks into a prompt section."""
        if not knowledge_chunks:
            return ""

        lines = ["## Knowledge Context", ""]
        for index, chunk in enumerate(knowledge_chunks, start=1):
            lines.append(f"{index}. {chunk.content}")
        return "\n".join(lines)
