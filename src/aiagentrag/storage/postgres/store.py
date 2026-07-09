"""PostgreSQL conversation store implementation."""

from typing import Self
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aiagentrag.core.exceptions import StorageError
from aiagentrag.core.models import Message, MessageRole
from aiagentrag.storage.postgres.models import MessageRecord


class PostgresConversationStore:
    """Persists active conversation messages in PostgreSQL."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Initialize the store with an async session factory."""
        self._session_factory = session_factory

    @classmethod
    async def initialize(cls, database_url: str) -> Self:
        """Return a ready conversation store."""
        engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return cls(session_factory)

    async def get_recent_messages(self, user_id: str, limit: int) -> list[Message]:
        """Return the most recent messages for a user."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(MessageRecord)
                    .where(MessageRecord.user_id == user_id)
                    .order_by(MessageRecord.created_at.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                records = list(result.scalars().all())
                records.reverse()
                return [self._to_domain(record) for record in records]
        except Exception as exc:
            msg = f"Failed to get recent messages: {exc}"
            raise StorageError(msg) from exc

    async def get_all_messages(self, user_id: str) -> list[Message]:
        """Return all messages for a user ordered by creation time."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(MessageRecord)
                    .where(MessageRecord.user_id == user_id)
                    .order_by(MessageRecord.created_at.asc())
                )
                result = await session.execute(stmt)
                records = list(result.scalars().all())
                return [self._to_domain(record) for record in records]
        except Exception as exc:
            msg = f"Failed to get all messages: {exc}"
            raise StorageError(msg) from exc

    async def save_message(self, message: Message) -> Message:
        """Persist a single message."""
        try:
            async with self._session_factory() as session:
                record = MessageRecord(
                    id=message.id,
                    user_id=message.user_id,
                    role=message.role.value,
                    content=message.content,
                    created_at=message.created_at,
                )
                session.add(record)
                await session.commit()
                return message
        except Exception as exc:
            msg = f"Failed to save message: {exc}"
            raise StorageError(msg) from exc

    async def delete_messages(self, message_ids: list[UUID]) -> None:
        """Delete messages by their identifiers."""
        if not message_ids:
            return
        try:
            async with self._session_factory() as session:
                stmt = delete(MessageRecord).where(MessageRecord.id.in_(message_ids))
                await session.execute(stmt)
                await session.commit()
        except Exception as exc:
            msg = f"Failed to delete messages: {exc}"
            raise StorageError(msg) from exc

    async def count_messages(self, user_id: str) -> int:
        """Return the total number of messages for a user."""
        try:
            async with self._session_factory() as session:
                stmt = (
                    select(func.count())
                    .select_from(MessageRecord)
                    .where(MessageRecord.user_id == user_id)
                )
                result = await session.execute(stmt)
                return int(result.scalar_one())
        except Exception as exc:
            msg = f"Failed to count messages: {exc}"
            raise StorageError(msg) from exc

    @staticmethod
    def _to_domain(record: MessageRecord) -> Message:
        """Convert an ORM record to a domain message."""
        return Message(
            id=record.id,
            user_id=record.user_id,
            role=MessageRole(record.role),
            content=record.content,
            created_at=record.created_at,
        )
