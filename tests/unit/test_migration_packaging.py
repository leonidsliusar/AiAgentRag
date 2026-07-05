"""Tests for migration files included in built distributions."""

import subprocess
import zipfile
from pathlib import Path


def test_wheel_contains_embedded_alembic_files() -> None:
    """Built wheel must ship embedded Alembic resources."""
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["poetry", "build", "--format", "wheel"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_paths = sorted(project_root.glob("dist/*.whl"))
    assert wheel_paths, "Expected a wheel artifact in dist/"

    expected_suffixes = (
        "aiagentrag/storage/postgres/alembic.ini",
        "aiagentrag/storage/postgres/alembic/env.py",
        "aiagentrag/storage/postgres/alembic/versions/001_initial_messages.py",
    )

    with zipfile.ZipFile(wheel_paths[-1]) as wheel:
        names = wheel.namelist()
        for suffix in expected_suffixes:
            assert any(name.endswith(suffix) for name in names), suffix
