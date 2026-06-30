from __future__ import annotations

"""Shared sample selection helpers for dataset builders."""

from pathlib import Path

from quickdelight.input.paths import discover_sample_ids, load_sample_ids, select_one_frame_per_capture, shard_sample_ids
from quickdelight.utils.batch import BatchSelection, validate_batch_selection


DatasetSelection = BatchSelection


def collect_sample_ids(raw_root: Path, selection: DatasetSelection) -> tuple[str, ...]:
    validate_batch_selection(selection)
    sample_ids = load_sample_ids(selection.sample_list) if selection.sample_list else discover_sample_ids(raw_root)
    if selection.one_frame_per_capture:
        sample_ids = select_one_frame_per_capture(sample_ids)
    sample_ids = shard_sample_ids(sample_ids, selection.num_shards, selection.shard_index)
    if selection.limit is not None:
        sample_ids = sample_ids[: selection.limit]
    return sample_ids
