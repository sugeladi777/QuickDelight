from __future__ import annotations

"""Losses for self-supervised UV texture training."""

import torch
import torch.nn.functional as F


def _uncrop_texture(texture: torch.Tensor, crop_box: torch.Tensor, full_size: int) -> torch.Tensor:
    batch, channels, _, _ = texture.shape
    output = texture.new_zeros((batch, channels, full_size, full_size))
    for index in range(batch):
        top, bottom, left, right = [int(value) for value in crop_box[index].tolist()]
        output[index, :, top:bottom, left:right] = texture[index]
    return output


def sample_texture_to_views(texture: torch.Tensor, uv: torch.Tensor, crop_box: torch.Tensor | None = None, full_size: int = 1024) -> torch.Tensor:
    if crop_box is not None and texture.shape[-1] != full_size:
        texture = _uncrop_texture(texture, crop_box, full_size=full_size)
    batch, views, _, height, width = uv.shape
    texture_rep = texture[:, None].expand(-1, views, -1, -1, -1).reshape(batch * views, texture.shape[1], texture.shape[2], texture.shape[3])
    grid = uv.permute(0, 1, 3, 4, 2).reshape(batch * views, height, width, 2)
    grid = grid.clamp(0.0, 1.0) * 2.0 - 1.0
    sampled = F.grid_sample(texture_rep, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.reshape(batch, views, texture.shape[1], height, width)


def visible_texture_l1(
    texture: torch.Tensor,
    partials: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor the prediction to observed partial-map texels."""
    valid = masks.float()
    loss = (torch.abs(texture[:, None] - partials) * valid).sum() / (valid.sum() * texture.shape[1] + eps)
    return loss, valid.mean()


def masked_reprojection_l1(
    texture: torch.Tensor,
    target_images: torch.Tensor,
    uv: torch.Tensor,
    masks: torch.Tensor,
    crop_box: torch.Tensor | None = None,
    full_size: int = 1024,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = sample_texture_to_views(texture, uv, crop_box=crop_box, full_size=full_size)
    valid = masks.float()
    loss = (torch.abs(rendered - target_images) * valid).sum() / (valid.sum() * texture.shape[1] + eps)
    return loss, valid.mean()


def masked_tv_loss(texture: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """A light smoothness prior inside the valid UV region."""
    if mask is None:
        return torch.abs(texture[..., 1:, :] - texture[..., :-1, :]).mean() + torch.abs(texture[..., :, 1:] - texture[..., :, :-1]).mean()

    mask = mask.float()
    y_valid = mask[..., 1:, :] * mask[..., :-1, :]
    x_valid = mask[..., :, 1:] * mask[..., :, :-1]
    y_loss = (torch.abs(texture[..., 1:, :] - texture[..., :-1, :]) * y_valid).sum() / (y_valid.sum() * texture.shape[1] + eps)
    x_loss = (torch.abs(texture[..., :, 1:] - texture[..., :, :-1]) * x_valid).sum() / (x_valid.sum() * texture.shape[1] + eps)
    return x_loss + y_loss
