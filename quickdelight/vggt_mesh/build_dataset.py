from __future__ import annotations

"""Batch builder for image-only self-supervised inputs."""

from functools import partial

from quickdelight.dataset_selection import DatasetSelection, collect_sample_ids
from quickdelight.utils.batch import BatchRunResult, run_batch

from .image_pipeline import ImageOnlyInputConfig, build_input_from_images


def _build_one(sample_id: str, config: ImageOnlyInputConfig) -> None:
    build_input_from_images(
        ImageOnlyInputConfig(
            image_root=config.image_root / sample_id,
            output_root=config.output_root,
            sample_id=sample_id,
            views=config.views,
            image_size=config.image_size,
            texture_size=config.texture_size,
            device=config.device,
            pixel3dmm_root=config.pixel3dmm_root,
            vggtface2_root=config.vggtface2_root,
            overwrite=config.overwrite,
        )
    )


def build_image_only_dataset(config: ImageOnlyInputConfig, selection: DatasetSelection) -> BatchRunResult:
    sample_ids = collect_sample_ids(config.image_root, selection)
    if not sample_ids:
        raise RuntimeError(f"no samples selected under {config.image_root}")
    return run_batch(
        sample_ids,
        partial(_build_one, config=config),
        jobs=selection.jobs,
        continue_on_error=selection.continue_on_error,
    )
