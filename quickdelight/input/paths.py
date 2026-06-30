from __future__ import annotations

"""Minimal path helpers for current QuickDelight flows."""

from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".avif")


def normalize_camera_id(value: str) -> str:
    return Path(str(value)).stem.removeprefix("cam")


def _camera_sort_key(camera_id: str) -> tuple[int, str]:
    return (int(camera_id), camera_id) if camera_id.isdigit() else (10**12, camera_id)


def raw_image_path(image_dir: Path, camera_id: str) -> Path | None:
    camera_id = normalize_camera_id(camera_id)
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"cam{camera_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def discover_camera_ids(image_dir: Path, max_views: int | None = None) -> list[str]:
    camera_ids = sorted(
        {
            normalize_camera_id(path.name)
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        },
        key=_camera_sort_key,
    )
    return camera_ids[:max_views] if max_views is not None else camera_ids


def discover_sample_ids(raw_root: Path) -> tuple[str, ...]:
    root = raw_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing raw root: {root}")
    return tuple(child.name for child in sorted(root.iterdir()) if child.is_dir() and not child.name.startswith("."))


def load_sample_ids(path: Path) -> tuple[str, ...]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing sample list: {resolved}")
    sample_ids: list[str] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            sample_ids.append(Path(text).name)
    return tuple(sample_ids)


def select_one_frame_per_capture(sample_ids: tuple[str, ...]) -> tuple[str, ...]:
    selected: dict[str, str] = {}
    for sample_id in sorted(sample_ids):
        capture = sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
        selected.setdefault(capture, sample_id)
    return tuple(sorted(selected.values()))


def shard_sample_ids(sample_ids: tuple[str, ...], num_shards: int, shard_index: int) -> tuple[str, ...]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return tuple(sample_id for index, sample_id in enumerate(sample_ids) if index % num_shards == shard_index)
