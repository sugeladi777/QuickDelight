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
    masks: torch.Tensor,
    prediction: torch.Tensor,
    crop_box: torch.Tensor,
    full_size: int,
    target_images: torch.Tensor | None = None,
    uv: torch.Tensor | None = None,
    reproject_masks: torch.Tensor | None = None,
    max_views: int = 4,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    view_count = min(max_views, inputs.shape[1])
    save_image(inputs[0, :view_count], preview_dir / "partial_inputs.png", nrow=view_count)
    save_image(masks[0, :view_count], preview_dir / "partial_masks.png", nrow=view_count)
    save_image(_uncrop_tensor(prediction[0], crop_box[0], full_size=full_size), preview_dir / "pred_texture.png")
    if target_images is not None:
        save_image(target_images[0, :view_count], preview_dir / "target_images.png", nrow=view_count)
    if uv is not None and target_images is not None:
        from quickdelight.selfsup.loss import sample_texture_to_views

        rendered = sample_texture_to_views(prediction[:1], uv[:1], crop_box=crop_box[:1], full_size=full_size)[0, :view_count]
        if reproject_masks is not None:
            rendered = rendered * reproject_masks[0, :view_count]
        save_image(rendered, preview_dir / "rendered_images.png", nrow=view_count)
    if reproject_masks is not None:
        save_image(reproject_masks[0, :view_count], preview_dir / "reproject_masks.png", nrow=view_count)
