"""Public PostgreSQL migration API."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from contextlib import contextmanager
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, TypeVar

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

T = TypeVar("T")


def _resolve_database_url(database_url: str | None) -> str:
    """Resolve the database URL from argument or environment."""
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        msg = "DATABASE_URL is not configured"
        raise RuntimeError(msg)
    return url


def _run_blocking_task(task: Callable[[], T]) -> T:
    """Run a blocking task safely from sync or async callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return task()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(task).result()


@contextmanager
def _alembic_config(database_url: str) -> Iterator[Config]:
    """Build an Alembic config using package resources."""
    postgres_root = files("aiagentrag.storage.postgres")
    with (
        as_file(postgres_root / "alembic.ini") as ini_path,
        as_file(postgres_root / "alembic") as script_path,
    ):
        alembic_cfg = Config(str(ini_path))
        alembic_cfg.set_main_option("script_location", str(script_path))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        yield alembic_cfg


def upgrade_head(database_url: str | None = None) -> None:
    """Apply all pending migrations."""

    def _task() -> None:
        url = _resolve_database_url(database_url)
        with _alembic_config(url) as cfg:
            command.upgrade(cfg, "head")

    _run_blocking_task(_task)


def upgrade(database_url: str | None, revision: str) -> None:
    """Apply migrations up to the given revision."""

    def _task() -> None:
        url = _resolve_database_url(database_url)
        with _alembic_config(url) as cfg:
            command.upgrade(cfg, revision)

    _run_blocking_task(_task)


def downgrade(database_url: str | None, revision: str) -> None:
    """Revert migrations down to the given revision."""

    def _task() -> None:
        url = _resolve_database_url(database_url)
        with _alembic_config(url) as cfg:
            command.downgrade(cfg, revision)

    _run_blocking_task(_task)


async def _fetch_current_revision(database_url: str) -> str | None:
    """Fetch the current Alembic revision from the database."""
    with _alembic_config(database_url) as cfg:
        configuration = cfg.get_section(cfg.config_ini_section, {})
        configuration["sqlalchemy.url"] = database_url
        connectable = async_engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        try:
            async with connectable.connect() as connection:

                def _get_revision(sync_connection: object) -> str | None:
                    context = MigrationContext.configure(sync_connection)  # type: ignore[arg-type]
                    return context.get_current_revision()

                return await connection.run_sync(_get_revision)
        finally:
            await connectable.dispose()


def current(database_url: str | None = None) -> str | None:
    """Return the current database revision."""
    url = _resolve_database_url(database_url)
    return _run_blocking_task(lambda: asyncio.run(_fetch_current_revision(url)))


def history(database_url: str | None = None) -> list[str]:
    """Return all available migration revision identifiers."""
    url = _resolve_database_url(database_url)
    with _alembic_config(url) as cfg:
        script = ScriptDirectory.from_config(cfg)
        return [revision.revision for revision in script.walk_revisions()]


def migration_resources_available() -> bool:
    """Check that embedded Alembic resources are discoverable."""
    postgres_root = files("aiagentrag.storage.postgres")
    return (
        (postgres_root / "alembic.ini").is_file()
        and (postgres_root / "alembic" / "env.py").is_file()
        and (postgres_root / "alembic" / "versions" / "001_initial_messages.py").is_file()
    )
