from __future__ import annotations

"""Shared helpers for batch-style sample processing."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class BatchSelection:
    sample_list: Path | None = None
    limit: int | None = None
    one_frame_per_capture: bool = False
    num_shards: int = 1
    shard_index: int = 0
    jobs: int = 1
    continue_on_error: bool = False


@dataclass(frozen=True)
class BatchItemResult:
    sample_id: str
    ok: bool
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class BatchRunResult:
    total: int
    succeeded: tuple[str, ...]
    failed: tuple[str, ...]
    item_results: tuple[BatchItemResult, ...]
    wall_seconds: float


def validate_batch_selection(selection: BatchSelection) -> None:
    if selection.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if selection.shard_index < 0 or selection.shard_index >= selection.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if selection.jobs <= 0:
        raise ValueError("jobs must be positive")
    if selection.limit is not None and selection.limit < 0:
        raise ValueError("limit must be non-negative")


def format_batch_run_result(result: BatchRunResult) -> str:
    lines = [
        f"total: {result.total}",
        f"succeeded: {len(result.succeeded)}",
        f"failed: {len(result.failed)}",
        f"wall_seconds: {result.wall_seconds:.2f}",
    ]
    if result.failed:
        lines.append("failed samples:")
        for item in result.item_results:
            if item.ok:
                continue
            suffix = f" ({item.error})" if item.error else ""
            lines.append(f"  - {item.sample_id}: {item.elapsed_seconds:.2f}s{suffix}")
    return "\n".join(lines)


def _run_one(build_one: Callable[[str], object], sample_id: str) -> BatchItemResult:
    start = perf_counter()
    try:
        build_one(sample_id)
    except Exception as exc:
        return BatchItemResult(
            sample_id=sample_id,
            ok=False,
            elapsed_seconds=perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
    return BatchItemResult(sample_id=sample_id, ok=True, elapsed_seconds=perf_counter() - start)


def run_batch(
    sample_ids: tuple[str, ...],
    build_one: Callable[[str], object],
    jobs: int,
    continue_on_error: bool,
) -> BatchRunResult:
    wall_start = perf_counter()
    if jobs == 1:
        item_results: list[BatchItemResult] = []
        for sample_id in sample_ids:
            result = _run_one(build_one, sample_id)
            item_results.append(result)
            status = "ok" if result.ok else "failed"
            print(f"[batch] {len(item_results)}/{len(sample_ids)} {sample_id} {status} {result.elapsed_seconds:.2f}s", flush=True)
            if not result.ok and not continue_on_error:
                break
        return _finalize_batch_result(sample_ids, item_results, wall_start)

    item_results = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_sample = {executor.submit(_run_one, build_one, sample_id): sample_id for sample_id in sample_ids}
        for future in as_completed(future_to_sample):
            result = future.result()
            item_results.append(result)
            status = "ok" if result.ok else "failed"
            print(f"[batch] {len(item_results)}/{len(sample_ids)} {result.sample_id} {status} {result.elapsed_seconds:.2f}s", flush=True)
            if not result.ok and not continue_on_error:
                executor.shutdown(wait=False, cancel_futures=True)
                break
    return _finalize_batch_result(sample_ids, item_results, wall_start)


def _finalize_batch_result(
    sample_ids: tuple[str, ...],
    item_results: list[BatchItemResult],
    wall_start: float,
) -> BatchRunResult:
    ordered = sorted(item_results, key=lambda item: sample_ids.index(item.sample_id))
    succeeded = tuple(item.sample_id for item in ordered if item.ok)
    failed = tuple(item.sample_id for item in ordered if not item.ok)
    return BatchRunResult(
        total=len(sample_ids),
        succeeded=succeeded,
        failed=failed,
        item_results=tuple(ordered),
        wall_seconds=perf_counter() - wall_start,
    )
