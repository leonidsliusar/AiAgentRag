"""CLI for PostgreSQL schema migrations."""

from __future__ import annotations

import argparse
import sys

from aiagentrag.storage.postgres.migration import (
    current,
    downgrade,
    history,
    upgrade,
    upgrade_head,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="aiagentrag-migrate",
        description="Manage AIAgentRag PostgreSQL schema migrations.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL URL (defaults to DATABASE_URL environment variable)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    upgrade_parser = subparsers.add_parser("upgrade", help="Apply migrations")
    upgrade_parser.add_argument(
        "revision",
        nargs="?",
        default="head",
        help="Target revision (default: head)",
    )

    downgrade_parser = subparsers.add_parser("downgrade", help="Revert migrations")
    downgrade_parser.add_argument("revision", help="Target revision")

    subparsers.add_parser("current", help="Show current revision")
    subparsers.add_parser("history", help="List available revisions")

    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    database_url: str | None = args.database_url

    try:
        if args.command == "upgrade":
            if args.revision == "head":
                upgrade_head(database_url)
            else:
                upgrade(database_url, args.revision)
        elif args.command == "downgrade":
            downgrade(database_url, args.revision)
        elif args.command == "current":
            revision = current(database_url)
            print(revision or "(empty)")
        elif args.command == "history":
            for revision in history(database_url):
                print(revision)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
