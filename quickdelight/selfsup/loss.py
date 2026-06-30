from __future__ import annotations

"""Self-supervised image reprojection losses."""

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


def _erode_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask.float()
    batch, views, channels, height, width = mask.shape
    flat = mask.float().reshape(batch * views, channels, height, width)
    kernel_size = pixels * 2 + 1
    eroded = 1.0 - F.max_pool2d(1.0 - flat, kernel_size=kernel_size, stride=1, padding=pixels)
    return eroded.reshape(batch, views, channels, height, width)


def _masked_mean_std(image: torch.Tensor, mask: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    mean = (image * mask).sum(dim=(-2, -1), keepdim=True) / count
    var = (((image - mean) ** 2) * mask).sum(dim=(-2, -1), keepdim=True) / count
    return mean, torch.sqrt(var + eps)


def _masked_color_normalize(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    rendered_mean, rendered_std = _masked_mean_std(rendered, mask, eps)
    target_mean, target_std = _masked_mean_std(target, mask, eps)
    return (rendered - rendered_mean) / rendered_std, (target - target_mean) / target_std


def _pointwise_loss(diff: torch.Tensor, loss_type: str, charbonnier_eps: float) -> torch.Tensor:
    if loss_type == "l1":
        return torch.abs(diff)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    if loss_type == "charbonnier":
        return torch.sqrt(diff * diff + charbonnier_eps * charbonnier_eps)
    raise ValueError(f"unsupported reprojection loss type: {loss_type}")


def _masked_average(values: torch.Tensor, mask: torch.Tensor, channels: int, eps: float) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() * channels + eps)


def _masked_gradient_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
    charbonnier_eps: float,
    eps: float,
) -> torch.Tensor:
    x_mask = mask[..., :, 1:] * mask[..., :, :-1]
    y_mask = mask[..., 1:, :] * mask[..., :-1, :]
    rendered_dx = rendered[..., :, 1:] - rendered[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    rendered_dy = rendered[..., 1:, :] - rendered[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    x_loss = _masked_average(_pointwise_loss(rendered_dx - target_dx, loss_type, charbonnier_eps), x_mask, rendered.shape[2], eps)
    y_loss = _masked_average(_pointwise_loss(rendered_dy - target_dy, loss_type, charbonnier_eps), y_mask, rendered.shape[2], eps)
    return x_loss + y_loss


def masked_reprojection_loss(
    texture: torch.Tensor,
    target_images: torch.Tensor,
    uv: torch.Tensor,
    masks: torch.Tensor,
    crop_box: torch.Tensor | None = None,
    full_size: int = 1024,
    loss_type: str = "l1",
    mask_erode_pixels: int = 0,
    color_normalize: bool = False,
    gradient_weight: float = 0.0,
    charbonnier_eps: float = 1e-3,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    rendered = sample_texture_to_views(texture, uv, crop_box=crop_box, full_size=full_size)
    valid = _erode_mask(masks, mask_erode_pixels)
    loss_rendered, loss_target = (rendered, target_images)
    if color_normalize:
        loss_rendered, loss_target = _masked_color_normalize(rendered, target_images, valid, eps)
    photometric = _masked_average(_pointwise_loss(loss_rendered - loss_target, loss_type, charbonnier_eps), valid, texture.shape[1], eps)
    gradient = rendered.new_tensor(0.0)
    if gradient_weight > 0.0:
        gradient = _masked_gradient_loss(loss_rendered, loss_target, valid, loss_type, charbonnier_eps, eps)
    total = photometric + gradient_weight * gradient
    return {
        "total": total,
        "photometric": photometric,
        "gradient": gradient,
        "valid_ratio": valid.mean(),
    }


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
