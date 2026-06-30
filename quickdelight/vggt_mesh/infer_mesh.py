from __future__ import annotations

"""Infer a UV mesh from VGGTFace2-style point-map inputs."""

import importlib
import json
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGGTFACE2_ROOT = REPO_ROOT / "third_party" / "vggtface2-dev"


@dataclass(frozen=True)
class VGGTFace2MeshConfig:
    pkl_path: Path
    output_mesh_path: Path
    vggtface2_root: Path = DEFAULT_VGGTFACE2_ROOT
    uv_size: int = 512
    num_views: int | None = 16
    device: str = "cuda:0"
    blur_point_map: bool = True
    overwrite: bool = False


@contextmanager
def _third_party_imports(root: Path):
    root = root.expanduser().resolve()
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def _load_pkl(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    required = {"uvs", "masks", "point_maps"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"missing VGGTFace2 pkl keys for mesh inference: {sorted(missing)}")
    return {key: np.asarray(value) for key, value in data.items()}


def _load_third_party_modules(root: Path):
    with _third_party_imports(root):
        models = importlib.import_module("models")
        reconstruct = importlib.import_module("reconstruct_mesh_utils")
    return models, reconstruct


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape:
        raise ValueError(f"shape mismatch: {source.shape} vs {target.shape}")
    n, dim = source.shape
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = source_centered.T @ target_centered / n
    u, singular_values, vt = np.linalg.svd(covariance)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        singular_values[-1] = -singular_values[-1]
        u[:, -1] = -u[:, -1]
    rotation = u @ vt
    scale = singular_values.sum() / np.var(source, axis=0).sum()
    translation = target.mean(axis=0) - source.mean(axis=0).dot(scale * rotation)
    return float(scale), rotation.astype(np.float32), translation.astype(np.float32)


def _masked_gaussian_blur(image: np.ndarray, mask: np.ndarray, ksize: int = 3, sigma: float = 0.0, eps: float = 1e-8) -> np.ndarray:
    import cv2

    image = image.astype(np.float32)
    mask = (mask.astype(np.float32) > 0).astype(np.float32)
    if image.ndim == 2:
        weighted = image * mask
        num = cv2.GaussianBlur(weighted, (ksize, ksize), sigma)
        den = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
        return num / (den + eps)
    if image.ndim == 3:
        mask_3 = mask[..., None]
        weighted = image * mask_3
        num = cv2.GaussianBlur(weighted, (ksize, ksize), sigma)
        den = cv2.GaussianBlur(mask, (ksize, ksize), sigma)[..., None]
        return num / (den + eps)
    raise ValueError(f"unsupported image shape: {image.shape}")


def _load_model(root: Path, device: torch.device):
    models, _ = _load_third_party_modules(root)
    model = models.MultiViewFusionNetDeep(
        in_channels=3,
        use_mask=True,
        base_channels=64,
        out_channels=3,
        norm_type="gn",
    ).to(device)
    checkpoint_path = root / "checkpoints" / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _fuse_uv_single_view(
    uv_map: torch.Tensor,
    point_map: torch.Tensor,
    mask: torch.Tensor,
    uv_size: int,
    eps: float = 1e-8,
    chunk: int = 2_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = uv_map.device
    height, width, _ = uv_map.shape
    uv = uv_map.unsqueeze(0).clamp(1e-7, 1.0 - 1e-7)
    attr = point_map.unsqueeze(0)
    weight = mask.unsqueeze(0).to(torch.float32)

    x = uv[..., 0] * (uv_size - 1)
    y = uv[..., 1] * (uv_size - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=uv_size - 1)
    y1 = (y0 + 1).clamp(max=uv_size - 1)
    wx1 = x - x0.to(x.dtype)
    wy1 = y - y0.to(y.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    weights = [
        (y0 * uv_size + x0).reshape(-1),
        (y1 * uv_size + x0).reshape(-1),
        (y0 * uv_size + x1).reshape(-1),
        (y1 * uv_size + x1).reshape(-1),
    ]
    splat_weights = [
        ((wx0 * wy0) * weight).reshape(-1),
        ((wx0 * wy1) * weight).reshape(-1),
        ((wx1 * wy0) * weight).reshape(-1),
        ((wx1 * wy1) * weight).reshape(-1),
    ]
    attr_flat = attr.reshape(height * width, 3).to(torch.float32)
    accum = torch.zeros((uv_size * uv_size, 3), device=device, dtype=torch.float32)
    support = torch.zeros((uv_size * uv_size,), device=device, dtype=torch.float32)

    for indices, flat_weight in zip(weights, splat_weights):
        for start in range(0, attr_flat.shape[0], chunk):
            end = min(start + chunk, attr_flat.shape[0])
            idx = indices[start:end]
            w = flat_weight[start:end].float()
            accum.index_add_(0, idx, w[:, None] * attr_flat[start:end])
            support.index_add_(0, idx, w)

    point_map_uv = accum / (support[:, None] + eps)
    return point_map_uv.view(uv_size, uv_size, 3), support.view(uv_size, uv_size)


def _template_valid_mask(root: Path, device: torch.device, uv_size: int, vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor) -> torch.Tensor:
    import nvdiffrast.torch as dr

    ctx = dr.RasterizeCudaContext(device=str(device))
    faces = faces.to(torch.int32)
    pos_clip = torch.zeros((uvs.shape[0], 4), device=device, dtype=torch.float32)
    pos_clip[:, 0:2] = uvs.to(torch.float32) * 2.0 - 1.0
    pos_clip[:, 2] = 0.0
    pos_clip[:, 3] = 1.0
    rast, _ = dr.rasterize(ctx, pos_clip[None, ...], faces, (uv_size, uv_size))
    mask_uv = rast[0, ..., 3] > 0
    return mask_uv


def _canonicalize_partial_uv_point_map(
    root: Path,
    pred_uv_point_map: torch.Tensor,
    valid_mask: torch.Tensor,
    uvs_array: torch.Tensor,
    canonical_vertices: np.ndarray,
    template_valid_mask: torch.Tensor,
    radius: int = 1,
    sigma: float = 0.8,
) -> torch.Tensor:
    _, reconstruct = _load_third_party_modules(root)
    zero_mask = torch.all(pred_uv_point_map == 0, dim=-1)
    valid_mask = torch.logical_and(valid_mask.bool(), ~zero_mask)
    sampled_vertices_pred, vertices_valid, _ = reconstruct.sample_uv_pointmap_local_masked(
        pred_uv_point_map.float(),
        valid_mask,
        uvs_array,
        radius=radius,
        sigma=sigma,
        min_valid_neighbors=1,
    )
    vertices_mask = (vertices_valid > 1.0).detach().cpu().numpy()
    if vertices_mask.sum() < 10:
        scale = 1.0
        rotation = np.eye(3, dtype=np.float32)
        translation = np.zeros((3,), dtype=np.float32)
    else:
        scale, rotation, translation = _umeyama(
            sampled_vertices_pred.detach().cpu().numpy()[vertices_mask],
            canonical_vertices[vertices_mask],
        )
        rotation = rotation.astype(np.float32)
        translation = translation.astype(np.float32)

    pred_np = pred_uv_point_map.detach().cpu().numpy()
    pred_np = scale * (pred_np @ rotation) + translation
    valid_np = valid_mask.detach().cpu().numpy().astype(np.float32)
    template_np = template_valid_mask.detach().cpu().numpy().astype(np.float32)
    pred_np = pred_np * valid_np[..., None] * template_np[..., None]
    return torch.from_numpy(np.clip(pred_np, -1.0, 1.0)).float().to(pred_uv_point_map.device)


@torch.no_grad()
def _infer_full_uv_point_map(
    root: Path,
    model: torch.nn.Module,
    uvs: np.ndarray,
    masks: np.ndarray,
    point_maps: np.ndarray,
    canonical_vertices: np.ndarray,
    uvs_array: np.ndarray,
    template_mask: torch.Tensor,
    device: torch.device,
    uv_size: int,
    num_views: int | None,
) -> np.ndarray:
    uvs_t = torch.from_numpy(uvs.astype(np.float32)).to(device)
    masks_t = torch.from_numpy(masks.astype(np.float32)).to(device)
    point_maps_t = torch.from_numpy(point_maps.astype(np.float32)).to(device)
    uvs_array_t = torch.from_numpy(uvs_array.astype(np.float32)).to(device)

    selected = list(range(uvs_t.shape[0] if num_views is None else min(num_views, uvs_t.shape[0])))
    pred_uv_list: list[torch.Tensor] = []
    valid_mask_list: list[torch.Tensor] = []
    for view_index in selected:
        mask = (masks_t[view_index] > 0.5).float()
        pred_uv_point_map, support = _fuse_uv_single_view(uvs_t[view_index], point_maps_t[view_index], mask, uv_size=uv_size)
        valid_mask = support > 1e-1
        pred_uv_list.append(
            _canonicalize_partial_uv_point_map(
                root,
                pred_uv_point_map=pred_uv_point_map,
                valid_mask=valid_mask,
                uvs_array=uvs_array_t,
                canonical_vertices=canonical_vertices,
                template_valid_mask=template_mask,
            )
        )
        valid_mask_list.append(valid_mask & template_mask)

    pred_t = torch.stack(pred_uv_list, dim=0).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    mask_t = torch.stack(valid_mask_list, dim=0).unsqueeze(1).unsqueeze(0).float().to(device)
    out_t = model(pred_t, mask_t)
    return out_t[0].permute(1, 2, 0).detach().cpu().numpy()


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# VGGTFace2 inferred mesh\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in uvs:
            handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for face in faces.astype(np.int64):
            a, b, c = face + 1
            handle.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def infer_vggtface2_mesh(config: VGGTFace2MeshConfig) -> Path:
    output_mesh_path = config.output_mesh_path.expanduser().resolve()
    if output_mesh_path.is_file() and not config.overwrite:
        return output_mesh_path

    root = config.vggtface2_root.expanduser().resolve()
    device = torch.device(config.device if torch.cuda.is_available() or not str(config.device).startswith("cuda") else "cpu")
    data = _load_pkl(config.pkl_path.expanduser().resolve())
    uvs = np.asarray(data["uvs"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    point_maps = np.asarray(data["point_maps"], dtype=np.float32)
    if uvs.ndim != 4 or uvs.shape[-1] != 2 or point_maps.shape[:3] != uvs.shape[:3] or masks.shape != uvs.shape[:3]:
        raise ValueError(f"invalid VGGTFace2 pkl shapes: uvs={uvs.shape}, masks={masks.shape}, point_maps={point_maps.shape}")

    canonical_mesh = trimesh.load_mesh(root / "canonical_mesh_flame.ply", process=False)
    canonical_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float32)
    canonical_faces = np.asarray(canonical_mesh.faces, dtype=np.int64)
    uvs_array = np.load(root / "canonical_mesh_flame_uvs.npy").astype(np.float32)
    if len(canonical_vertices) != len(uvs_array):
        raise ValueError(f"canonical vertex/UV count mismatch: vertices={len(canonical_vertices)} uvs={len(uvs_array)}")

    model = _load_model(root, device)
    vertices_t = torch.from_numpy(canonical_vertices).float().to(device)
    faces_t = torch.from_numpy(canonical_faces).long().to(device)
    uvs_t = torch.from_numpy(uvs_array).float().to(device)
    template_mask = _template_valid_mask(root, device, config.uv_size, vertices_t, faces_t, uvs_t)
    full_uv_point_map = _infer_full_uv_point_map(
        root=root,
        model=model,
        uvs=uvs,
        masks=masks,
        point_maps=point_maps,
        canonical_vertices=canonical_vertices,
        uvs_array=uvs_array,
        template_mask=template_mask,
        device=device,
        uv_size=config.uv_size,
        num_views=config.num_views,
    )

    _, reconstruct = _load_third_party_modules(root)
    if config.blur_point_map:
        full_uv_point_map = _masked_gaussian_blur(full_uv_point_map, template_mask.detach().cpu().numpy(), ksize=3, sigma=0)
    sampled_vertices, _, _ = reconstruct.sample_uv_pointmap_local_masked(
        torch.from_numpy(full_uv_point_map).float().to(device),
        template_mask.to(device),
        uvs_t,
        radius=1,
        sigma=0.8,
    )
    vertices = sampled_vertices.detach().cpu().numpy().astype(np.float32)
    _write_obj(output_mesh_path, vertices, canonical_faces, uvs_array)
    (output_mesh_path.parent / f"{output_mesh_path.stem}_meta.json").write_text(
        json.dumps(
            {
                "source_pkl": str(config.pkl_path),
                "vggtface2_root": str(root),
                "uv_size": int(config.uv_size),
                "num_views": None if config.num_views is None else int(config.num_views),
                "vertices": int(vertices.shape[0]),
                "faces": int(canonical_faces.shape[0]),
                "blur_point_map": bool(config.blur_point_map),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_mesh_path
