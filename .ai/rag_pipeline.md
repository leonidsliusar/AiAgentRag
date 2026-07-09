# Retrieval Pipeline

The project contains two independent retrieval systems.

---

Knowledge Retrieval

Collection:

knowledge

Contains:

- psychology books
- articles
- PDF chunks

---

Memory Retrieval

Collection:

user_memory

Contains:

- compressed summaries
- long-term memories

---

Recent Conversation

Stored inside PostgreSQL.

---

Prompt Context

The prompt builder combines:

- recent conversation
- retrieved memories
- retrieved knowledge
- current user message

Only relevant information should be included.
