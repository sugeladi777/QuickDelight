from __future__ import annotations

"""QuickDelight command line interface."""

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "quickdelight_dataset"


def _add_dataset_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-list", "--sample_list", dest="sample_list", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--one-frame-per-capture", "--one_frame_per_capture", dest="one_frame_per_capture", action="store_true")
    parser.add_argument("--num-shards", "--num_shards", dest="num_shards", type=int, default=1)
    parser.add_argument("--shard-index", "--shard_index", dest="shard_index", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--continue-on-error", "--continue_on_error", dest="continue_on_error", action="store_true")


def _dataset_selection(args):
    from quickdelight.dataset_selection import DatasetSelection

    return DatasetSelection(
        sample_list=args.sample_list,
        limit=args.limit,
        one_frame_per_capture=args.one_frame_per_capture,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        jobs=args.jobs,
        continue_on_error=args.continue_on_error,
    )


def run_build_input(args) -> int:
    from quickdelight.vggt_mesh import DEFAULT_PIXEL3DMM_ROOT, DEFAULT_VGGTFACE2_ROOT, ImageOnlyInputConfig, build_input_from_images

    image_root = args.image_root if args.image_root is not None else args.raw_root / args.sample_id
    sample_root = build_input_from_images(
        ImageOnlyInputConfig(
            image_root=image_root,
            output_root=args.dataset_root,
            sample_id=args.sample_id,
            views=args.views,
            image_size=args.image_size,
            texture_size=args.texture_size,
            device=args.device,
            pixel3dmm_root=DEFAULT_PIXEL3DMM_ROOT,
            vggtface2_root=DEFAULT_VGGTFACE2_ROOT,
            overwrite=args.overwrite,
        )
    )
    print(f"[build-input] wrote {sample_root}")
    return 0


def run_build_dataset(args) -> int:
    from quickdelight.utils.batch import format_batch_run_result
    from quickdelight.vggt_mesh import DEFAULT_PIXEL3DMM_ROOT, DEFAULT_VGGTFACE2_ROOT, ImageOnlyInputConfig, build_image_only_dataset

    result = build_image_only_dataset(
        ImageOnlyInputConfig(
            image_root=args.raw_root,
            output_root=args.dataset_root,
            sample_id="",
            views=args.views,
            image_size=args.image_size,
            texture_size=args.texture_size,
            device=args.device,
            pixel3dmm_root=DEFAULT_PIXEL3DMM_ROOT,
            vggtface2_root=DEFAULT_VGGTFACE2_ROOT,
            overwrite=args.overwrite,
        ),
        _dataset_selection(args),
    )
    print("[build-dataset]")
    print(format_batch_run_result(result))
    return 0 if not result.failed or args.continue_on_error else 1


def run_train_selfsup(args) -> int:
    from quickdelight.selfsup import SelfSupervisedTrainingConfig, run_self_supervised_training

    run_self_supervised_training(
        SelfSupervisedTrainingConfig(
            dataset_root=args.dataset_root,
            save_root=args.save_root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            base_channels=args.base_channels,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_samples=args.max_samples,
            crop_to_uv_mask=args.crop_to_uv_mask,
            use_mask=args.use_mask,
            use_amp=args.use_amp,
            preview_every=args.preview_every,
            reprojection_weight=args.reprojection_weight,
            grad_clip_norm=args.grad_clip_norm,
            save_every=args.save_every,
            use_scheduler=args.use_scheduler,
        )
    )
    return 0


def _add_mesh_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", "--raw_root", dest="raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--dataset-root", "--dataset_root", "--output_root", dest="dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--views", "--max_views", dest="views", type=int, default=16)
    parser.add_argument("--image-size", "--image_size", dest="image_size", type=int, default=512)
    parser.add_argument("--texture-size", "--texture_size", "--uv_size", dest="texture_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m quickdelight")
    subparsers = parser.add_subparsers(dest="command")

    build_input = subparsers.add_parser("build-input")
    build_input.add_argument("sample_id")
    build_input.add_argument("--image-root", "--image_root", dest="image_root", type=Path, default=None)
    _add_mesh_build_args(build_input)
    build_input.set_defaults(func=run_build_input)

    build_dataset = subparsers.add_parser("build-dataset")
    _add_mesh_build_args(build_dataset)
    _add_dataset_selection(build_dataset)
    build_dataset.set_defaults(func=run_build_dataset)

    train_selfsup = subparsers.add_parser("train-selfsup")
    train_selfsup.add_argument("--dataset-root", "--dataset_root", "--data_root", dest="dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    train_selfsup.add_argument("--save-root", "--save_root", dest="save_root", type=Path, default=REPO_ROOT / "data" / "runs" / "quickdelight_selfsup")
    train_selfsup.add_argument("--device", type=str, default="cuda:0")
    train_selfsup.add_argument("--epochs", type=int, default=50)
    train_selfsup.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    train_selfsup.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    train_selfsup.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=3e-4)
    train_selfsup.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    train_selfsup.add_argument("--base-channels", "--base_channels", dest="base_channels", type=int, default=32)
    train_selfsup.add_argument("--val-ratio", "--val_ratio", dest="val_ratio", type=float, default=0.1)
    train_selfsup.add_argument("--seed", type=int, default=42)
    train_selfsup.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=None)
    train_selfsup.add_argument("--crop-to-uv-mask", "--crop_to_uv_mask", dest="crop_to_uv_mask", action="store_true", default=True)
    train_selfsup.add_argument("--no-crop-to-uv-mask", "--no_crop_to_uv_mask", dest="crop_to_uv_mask", action="store_false")
    train_selfsup.add_argument("--use-mask", "--use_mask", dest="use_mask", action="store_true", default=True)
    train_selfsup.add_argument("--no-use-mask", "--no_use_mask", dest="use_mask", action="store_false")
    train_selfsup.add_argument("--amp", dest="use_amp", action="store_true", default=True)
    train_selfsup.add_argument("--no-amp", dest="use_amp", action="store_false")
    train_selfsup.add_argument("--preview-every", "--preview_every", dest="preview_every", type=int, default=1)
    train_selfsup.add_argument("--reprojection-weight", "--reprojection_weight", dest="reprojection_weight", type=float, default=1.0)
    train_selfsup.add_argument("--grad-clip-norm", "--grad_clip_norm", dest="grad_clip_norm", type=float, default=1.0)
    train_selfsup.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=1)
    train_selfsup.add_argument("--scheduler", dest="use_scheduler", action="store_true", default=True)
    train_selfsup.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    train_selfsup.set_defaults(func=run_train_selfsup)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)
