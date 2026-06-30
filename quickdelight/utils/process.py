from __future__ import annotations

"""Subprocess helper."""

import subprocess
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, env=env, check=True)

