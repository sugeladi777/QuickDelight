from __future__ import annotations

"""Save texture training previews."""

from pathlib import Path

import torch
from torchvision.utils import save_image


def _uncrop_tensor(image: torch.Tensor, crop_box: torch.Tensor, full_size: int) -> torch.Tensor:
    output = image.new_zeros((image.shape[0], full_size, full_size))
    top, bottom, left, right = [int(x) for x in crop_box.tolist()]
    output[:, top:bottom, left:right] = image
    return output


def save_preview(
    preview_dir: Path,
    inputs: torch.Tensor,
    prediction: torch.Tensor,
    crop_box: torch.Tensor,
    full_size: int,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_image(inputs[0], preview_dir / "input.png", nrow=min(4, inputs.shape[1]))
    save_image(_uncrop_tensor(prediction[0], crop_box[0], full_size=full_size), preview_dir / "pred.png")
