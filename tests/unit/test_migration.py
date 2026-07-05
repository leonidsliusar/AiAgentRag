"""Unit tests for embedded PostgreSQL migrations."""

from unittest.mock import MagicMock, patch

import pytest

from aiagentrag.storage.postgres.migration import (
    current,
    downgrade,
    history,
    migration_resources_available,
    upgrade,
    upgrade_head,
)


def test_migration_resources_available() -> None:
    """Embedded Alembic resources must be discoverable in the package."""
    assert migration_resources_available()


def test_upgrade_head_requires_database_url() -> None:
    """Migration API must fail when database URL is missing."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="DATABASE_URL is not configured"),
    ):
        upgrade_head(None)


@patch("aiagentrag.storage.postgres.migration.command.upgrade")
def test_upgrade_head_calls_alembic(mock_upgrade: MagicMock) -> None:
    """upgrade_head must delegate to Alembic with injected configuration."""
    upgrade_head("postgresql+asyncpg://user:pass@localhost:5432/testdb")
    mock_upgrade.assert_called_once()
    config = mock_upgrade.call_args.args[0]
    revision = mock_upgrade.call_args.args[1]
    assert revision == "head"
    assert config.get_main_option("sqlalchemy.url") == (
        "postgresql+asyncpg://user:pass@localhost:5432/testdb"
    )


@patch("aiagentrag.storage.postgres.migration.command.upgrade")
def test_upgrade_calls_alembic_with_revision(mock_upgrade: MagicMock) -> None:
    """upgrade must pass the requested revision to Alembic."""
    upgrade("postgresql+asyncpg://user:pass@localhost:5432/testdb", "001_initial_messages")
    mock_upgrade.assert_called_once()
    assert mock_upgrade.call_args.args[1] == "001_initial_messages"


@patch("aiagentrag.storage.postgres.migration.command.downgrade")
def test_downgrade_calls_alembic(mock_downgrade: MagicMock) -> None:
    """downgrade must pass the requested revision to Alembic."""
    downgrade("postgresql+asyncpg://user:pass@localhost:5432/testdb", "base")
    mock_downgrade.assert_called_once()
    assert mock_downgrade.call_args.args[1] == "base"


def test_history_lists_revisions() -> None:
    """history must return embedded migration revision identifiers."""
    revisions = history("postgresql+asyncpg://user:pass@localhost:5432/testdb")
    assert "001_initial_messages" in revisions


@patch("aiagentrag.storage.postgres.migration.asyncio.run")
def test_current_delegates_to_async_lookup(mock_run: MagicMock) -> None:
    """current must resolve the revision via Alembic migration context."""
    mock_run.return_value = "001_initial_messages"
    revision = current("postgresql+asyncpg://user:pass@localhost:5432/testdb")
    assert revision == "001_initial_messages"
    mock_run.assert_called_once()


@patch("aiagentrag.storage.postgres.store.upgrade_head")
def test_initialize_store_runs_migrations(mock_upgrade_head: MagicMock) -> None:
    """PostgresConversationStore.initialize must apply migrations before use."""
    import asyncio

    from aiagentrag.storage.postgres.store import PostgresConversationStore

    async def _run() -> None:
        with (
            patch(
                "aiagentrag.storage.postgres.store.create_async_engine",
                return_value=MagicMock(),
            ),
            patch(
                "aiagentrag.storage.postgres.store.async_sessionmaker",
                return_value=MagicMock(),
            ),
        ):
            await PostgresConversationStore.initialize(
                "postgresql+asyncpg://user:pass@localhost:5432/testdb",
            )

    asyncio.run(_run())
    mock_upgrade_head.assert_called_once_with(
        "postgresql+asyncpg://user:pass@localhost:5432/testdb",
    )
