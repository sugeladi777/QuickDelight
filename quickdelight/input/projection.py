from __future__ import annotations

"""Project image colors into UV texture space."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    texture: np.ndarray
    mask: np.ndarray
    support: np.ndarray


def _empty_projection(texture_size: int) -> ProjectionResult:
    return ProjectionResult(
        texture=np.zeros((texture_size, texture_size, 3), dtype=np.float32),
        mask=np.zeros((texture_size, texture_size), dtype=bool),
        support=np.zeros((texture_size, texture_size), dtype=np.float32),
    )


def _valid_uv_mask(uv: np.ndarray, visibility_mask: np.ndarray) -> np.ndarray:
    valid_mask = np.asarray(visibility_mask) > 0
    finite_mask = np.isfinite(uv).all(axis=-1)
    range_mask = (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    return valid_mask & finite_mask & range_mask


def _flatten_texel_coords(x: np.ndarray, y: np.ndarray, texture_size: int) -> np.ndarray:
    return y * texture_size + x


def _collect_valid_samples(colors: np.ndarray, uv_coords: np.ndarray, visibility_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(colors, dtype=np.float32)
    uv = np.asarray(uv_coords, dtype=np.float32)
    valid = _valid_uv_mask(uv, visibility_mask)
    return np.clip(uv[valid], 0.0, 1.0), rgb[valid]


def _bilinear_neighbors(valid_uv: np.ndarray, texture_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = valid_uv[:, 0] * (texture_size - 1)
    y = valid_uv[:, 1] * (texture_size - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, texture_size - 1)
    y1 = np.clip(y0 + 1, 0, texture_size - 1)
    wx1 = x - x0.astype(np.float32)
    wy1 = y - y0.astype(np.float32)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    indices = np.concatenate(
        [
            _flatten_texel_coords(x0, y0, texture_size),
            _flatten_texel_coords(x0, y1, texture_size),
            _flatten_texel_coords(x1, y0, texture_size),
            _flatten_texel_coords(x1, y1, texture_size),
        ],
        axis=0,
    )
    weights = np.concatenate([wx0 * wy0, wx0 * wy1, wx1 * wy0, wx1 * wy1], axis=0).astype(np.float32)
    return indices, weights


def _repeat_rgb_samples(valid_rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([valid_rgb, valid_rgb, valid_rgb, valid_rgb], axis=0).astype(np.float32)


def _accumulate_support(flat_indices: np.ndarray, flat_weights: np.ndarray, texel_count: int) -> np.ndarray:
    support = np.zeros((texel_count,), dtype=np.float32)
    np.add.at(support, flat_indices, flat_weights)
    return support


def _accumulate_weighted_rgb(flat_indices: np.ndarray, flat_weights: np.ndarray, flat_rgb: np.ndarray, texel_count: int) -> np.ndarray:
    weighted_rgb = np.zeros((texel_count, 3), dtype=np.float32)
    np.add.at(weighted_rgb, flat_indices, flat_rgb * flat_weights[:, None])
    return weighted_rgb


def project_to_uv(
    colors: np.ndarray,
    uv_coords: np.ndarray,
    visibility_mask: np.ndarray,
    texture_size: int,
    eps: float = 1e-8,
    valid_weight_threshold: float = 1e-1,
) -> ProjectionResult:
    valid_uv, valid_rgb = _collect_valid_samples(colors, uv_coords, visibility_mask)
    if valid_uv.shape[0] == 0:
        return _empty_projection(texture_size)

    texel_count = texture_size * texture_size
    flat_indices, flat_weights = _bilinear_neighbors(valid_uv, texture_size)
    flat_rgb = _repeat_rgb_samples(valid_rgb)
    support = _accumulate_support(flat_indices, flat_weights, texel_count)
    weighted_rgb = _accumulate_weighted_rgb(flat_indices, flat_weights, flat_rgb, texel_count)
    texture = weighted_rgb / np.maximum(support[:, None], eps)
    mask = support > max(valid_weight_threshold, eps)
    return ProjectionResult(
        texture=texture.reshape(texture_size, texture_size, 3),
        mask=mask.reshape(texture_size, texture_size),
        support=np.clip(support, 0.0, 1.0).reshape(texture_size, texture_size),
    )
