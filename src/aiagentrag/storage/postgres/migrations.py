"""Backward-compatible migration module."""

from aiagentrag.storage.postgres.migration import (
    current,
    downgrade,
    history,
    upgrade,
    upgrade_head,
)

__all__ = [
    "current",
    "downgrade",
    "history",
    "upgrade",
    "upgrade_head",
]
