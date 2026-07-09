# Prompt Design

Every request sent to the LLM must follow the same structure.

---

System Prompt

Defines:

- agent role
- behavioral rules
- response style
- safety requirements

---

Conversation History

Contains recent dialogue from PostgreSQL.

---

Long-term Memory

Contains retrieved summaries from the user_memory collection.

Only relevant memories should be included.

---

Knowledge Context

Contains retrieved chunks from the knowledge collection.

Only relevant chunks should be included.

---

Current User Message

Always appended last.

---

Rules

Never inject irrelevant memories.

Never inject irrelevant knowledge.

Keep prompts deterministic.

Keep formatting consistent.

Avoid duplicated context.

The Prompt Builder is responsible for assembling the final prompt.

Other modules must not construct prompts manually.
