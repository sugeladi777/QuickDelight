from __future__ import annotations

"""File-system helpers.

These helpers deliberately stay tiny.  They hide repetitive path handling, but
they do not know anything about input building, GT building, or training.
"""

import shutil
from pathlib import Path


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def remove_dir(path: Path) -> None:
    resolved = resolve_path(path)
    if resolved.is_dir():
        shutil.rmtree(resolved)


def reset_dir(path: Path) -> Path:
    remove_dir(path)
    return ensure_dir(path)


def copy_file(source: Path, destination: Path) -> Path:
    src = resolve_path(source)
    dst = resolve_path(destination)
    if not src.is_file():
        raise FileNotFoundError(f"missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_files(pairs: list[tuple[Path, Path]] | tuple[tuple[Path, Path], ...]) -> tuple[Path, ...]:
    return tuple(copy_file(source, destination) for source, destination in pairs)

