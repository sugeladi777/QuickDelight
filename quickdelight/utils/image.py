from __future__ import annotations

"""Image loading, saving, and AVA-256 color conversion."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_binary_mask(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    with Image.open(path) as image:
        image = image.convert("L").resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _ava256_tonemap_constants() -> tuple[float, np.ndarray, float, np.ndarray]:
    gamma = 1.5254
    black = np.array([4.4 / 255.0, 3.1 / 255.0, 4.2 / 255.0], dtype=np.float32).reshape(1, 1, 3)
    scale = 1.0 / 1.1059
    color_scale = np.array([1.279545, 1.1059, 1.6], dtype=np.float32).reshape(1, 1, 3)
    return gamma, black, scale, color_scale


def ava256_linear_to_srgb(image: Image.Image) -> Image.Image:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gamma, black, scale, color_scale = _ava256_tonemap_constants()
    image_array = image_array * (color_scale * (scale / (1.0 - black))) - (black * (scale / (1.0 - black)))
    image_array = np.clip(np.power(np.clip(image_array, 1e-6, None), 1.0 / gamma), 0.0, 1.0)
    return Image.fromarray((image_array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def convert_to_png(input_path: Path, output_path: Path, overwrite: bool = False) -> Path:
    if output_path.is_file() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ava256_linear_to_srgb(Image.open(input_path)).save(output_path)
    return output_path

