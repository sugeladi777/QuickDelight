from __future__ import annotations

"""Build QuickDelight partial maps from internally generated UV buffers."""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from quickdelight.input.projection import project_to_uv
from quickdelight.utils.image import load_binary_mask
from quickdelight.utils.image import save_mask, save_rgb
from quickdelight.utils.uv_mask import load_fixed_uv_mask



def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_reprojection_view(image: np.ndarray, uv: np.ndarray, mask: np.ndarray, root: Path, view_name: str) -> None:
    save_rgb(image, root / "image" / f"{view_name}.png")
    np.save(root / "uv" / f"{view_name}.npy", uv.astype(np.float32))
    save_mask(mask, root / "mask" / f"{view_name}.png")


def _load_rgb_aligned(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_uv(path: Path) -> np.ndarray:
    uv = np.load(path).astype(np.float32)
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError(f"expected UV buffer with shape (H,W,2), got {uv.shape}: {path}")
    return np.clip(uv, 0.0, 1.0)


def _build_partial_view(
    image: np.ndarray,
    uv: np.ndarray,
    mask: np.ndarray,
    texture_size: int,
    fixed_uv_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    projection = project_to_uv(
        np.clip(image.astype(np.float32), 0.0, 1.0),
        np.clip(uv.astype(np.float32), 0.0, 1.0),
        mask > 0,
        texture_size=texture_size,
    )
    partial_mask = projection.mask & fixed_uv_mask
    partial_rgb = projection.texture * partial_mask[..., None].astype(np.float32)
    nonzero = np.any(partial_rgb > (1.0 / 255.0), axis=-1)
    return partial_rgb, partial_mask, {
        "source_mask_coverage": float((mask > 0).mean()),
        "projection_mask_coverage": float(projection.mask.mean()),
        "partial_mask_coverage": float(partial_mask.mean()),
        "partial_nonzero_coverage": float(nonzero.mean()),
    }


def _build_from_uv_buffers(
    sample_root: Path,
    image_dir: Path,
    uv_dir: Path,
    mask_dir: Path,
    texture_size: int,
    *,
    preserve_reproject: bool = False,
) -> None:
    input_root = sample_root / "input"
    reproject_root = sample_root / "reproject"
    _reset_dir(input_root / "partial")
    _reset_dir(input_root / "mask")
    if not preserve_reproject:
        _reset_dir(reproject_root / "image")
        _reset_dir(reproject_root / "uv")
        _reset_dir(reproject_root / "mask")

    fixed_uv_mask = load_fixed_uv_mask(texture_size)
    quality_records: list[dict[str, float | str | int]] = []
    image_paths = sorted(image_dir.glob("cam*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no rendered images found under {image_dir}")
    for image_path in image_paths:
        view_name = image_path.stem
        uv_path = uv_dir / f"{view_name}.npy"
        mask_path = mask_dir / f"{view_name}.png"
        if not uv_path.is_file():
            raise FileNotFoundError(f"missing UV buffer: {uv_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing reprojection mask: {mask_path}")

        uv = _load_uv(uv_path)
        image_size = (uv.shape[1], uv.shape[0])
        image = _load_rgb_aligned(image_path, image_size)
        mask = load_binary_mask(mask_path, image_size=image_size)

        if not preserve_reproject:
            _save_reprojection_view(image, uv, mask, reproject_root, view_name)
        partial_rgb, partial_mask, quality = _build_partial_view(image, uv, mask, texture_size, fixed_uv_mask)
        save_rgb(partial_rgb, input_root / "partial" / f"{view_name}.png")
        save_mask(partial_mask, input_root / "mask" / f"{view_name}.png")
        quality_records.append(
            {
                "camera_id": view_name.removeprefix("cam"),
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "uv_width": int(uv.shape[1]),
                "uv_height": int(uv.shape[0]),
                **quality,
            }
        )

    (input_root / "partial_quality.json").write_text(
        json.dumps(
            {
                "sampling_mode": "mesh_projected_uv",
                "texture_size": int(texture_size),
                "views": quality_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
