from __future__ import annotations

"""Shared fixed UV mask utilities."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXED_UV_MASK = REPO_ROOT / "quickdelight" / "assets" / "template_valid_mask_1024.pt"


def load_fixed_uv_mask(size: int, mask_path: Path = DEFAULT_FIXED_UV_MASK) -> np.ndarray:
    if mask_path.suffix == ".pt":
        mask = torch.load(mask_path, map_location="cpu", weights_only=True)
        mask_array = np.asarray(mask, dtype=np.uint8) > 0
        if mask_array.shape != (size, size):
            image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
            image = image.resize((size, size), Image.Resampling.NEAREST)
            mask_array = np.asarray(image, dtype=np.uint8) > 0
        return mask_array

    with Image.open(mask_path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0
