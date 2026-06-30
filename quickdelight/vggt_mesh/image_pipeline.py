from __future__ import annotations

"""Image-only VGGTFace2 input construction.

The public pipeline accepts only images.  UV maps, face masks, VGGT point maps,
the VGGTFace2 mesh, and QuickDelight partial maps are all generated internally.
"""

import json
import pickle
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from quickdelight.input.paths import IMAGE_EXTENSIONS, discover_camera_ids, raw_image_path

from .infer_mesh import DEFAULT_VGGTFACE2_ROOT, VGGTFace2MeshConfig, infer_vggtface2_mesh
from .run import _build_from_uv_buffers, _reset_dir, _save_reprojection_view


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIXEL3DMM_ROOT = REPO_ROOT / "third_party" / "pixel3dmm"
DEFAULT_CROP_SCALE = 1.0
FACE_LABELS = {"face", "rb", "lb", "re", "le", "nose", "ulip", "imouth", "llip"}
DEFAULT_PARSER_MODEL = "farl/lapa/448"


@dataclass(frozen=True)
class ImageOnlyInputConfig:
    image_root: Path
    output_root: Path
    sample_id: str = ""
    views: int = 16
    image_size: int = 512
    texture_size: int = 1024
    device: str = "cuda:0"
    pixel3dmm_root: Path = DEFAULT_PIXEL3DMM_ROOT
    vggtface2_root: Path = DEFAULT_VGGTFACE2_ROOT
    overwrite: bool = False


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _prepend_sys_path(path: Path):
    resolved = str(path.expanduser().resolve())
    sys.path.insert(0, resolved)
    try:
        yield
    finally:
        try:
            sys.path.remove(resolved)
        except ValueError:
            pass


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _resolve_image_dir(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing image root: {root}")
    image_select = root / "image_select"
    if image_select.is_dir():
        return image_select
    if any(_is_image_file(child) for child in root.iterdir()):
        return root
    raise FileNotFoundError(f"no images found under {root} or {image_select}")


def _crop_record_from_box(image_size: tuple[int, int], box: np.ndarray, scale: float = DEFAULT_CROP_SCALE) -> dict[str, int | float | str]:
    width, height = image_size
    x0, y0, x1, y1 = [float(value) for value in box]
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    side = max(max(x1 - x0, y1 - y0) * scale, 64.0)
    left = int(round(center_x - side * 0.5))
    top = int(round(center_y - side * 0.5))
    right = int(round(center_x + side * 0.5))
    bottom = int(round(center_y + side * 0.5))
    return {
        "method": "detected",
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "source_width": width,
        "source_height": height,
        "crop_scale": float(scale),
    }


def _fallback_center_crop_record(image_size: tuple[int, int]) -> dict[str, int | float | str]:
    width, height = image_size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return {
        "method": "center",
        "left": left,
        "top": top,
        "right": left + side,
        "bottom": top + side,
        "source_width": width,
        "source_height": height,
        "crop_scale": 1.0,
    }


def _crop_image_with_record(image: Image.Image, record: dict[str, int | float | str]) -> Image.Image:
    left = int(record["left"])
    top = int(record["top"])
    right = int(record["right"])
    bottom = int(record["bottom"])
    width, height = image.size
    crop = Image.new("RGB", (right - left, bottom - top), (0, 0, 0))
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(width, right)
    src_bottom = min(height, bottom)
    if src_right <= src_left or src_bottom <= src_top:
        return crop
    crop.paste(image.crop((src_left, src_top, src_right, src_bottom)), (max(0, -left), max(0, -top)))
    return crop


def _load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_rgb_array(path: Path) -> np.ndarray:
    with _load_rgb_image(path) as image:
        return np.asarray(image, dtype=np.uint8).copy()


def _detect_faces(image_rgbs: list[np.ndarray], device: torch.device):
    import batch_face

    detector = batch_face.RetinaFace(device=str(device), network="resnet50", return_dict=True)
    return detector.detect(image_rgbs, batch_size=min(8, max(1, len(image_rgbs))))


def _detect_crop_records(
    image_paths: list[Path],
    image_rgbs: list[np.ndarray],
    detections,
    crop_scale: float,
) -> dict[str, dict[str, int | float | str]]:
    records: dict[str, dict[str, int | float | str]] = {}
    for path, image_rgb, faces in zip(image_paths, image_rgbs, detections):
        camera_id = path.stem.removeprefix("cam")
        image_size = (int(image_rgb.shape[1]), int(image_rgb.shape[0]))
        if not faces:
            records[camera_id] = _fallback_center_crop_record(image_size)
            continue
        best_face = max(
            faces,
            key=lambda item: float(item["score"])
            * max(1.0, float((item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]))),
        )
        records[camera_id] = _crop_record_from_box(image_size, np.asarray(best_face["box"], dtype=np.float32), scale=crop_scale)
    return records


def _load_images(
    image_dir: Path,
    views: int,
    image_size: int,
    device: torch.device,
) -> tuple[
    list[str],
    list[Path],
    list[np.ndarray],
    object,
    np.ndarray,
    dict[str, dict[str, int | float | str]],
]:
    camera_ids = discover_camera_ids(image_dir, max_views=views)
    if not camera_ids:
        raise FileNotFoundError(f"no images found under {image_dir}")
    image_paths: list[Path] = []
    for camera_id in camera_ids:
        path = raw_image_path(image_dir, camera_id)
        if path is None:
            raise FileNotFoundError(f"missing image for camera {camera_id} under {image_dir}")
        image_paths.append(path)

    image_rgbs = [_load_rgb_array(path) for path in image_paths]
    detections = _detect_faces(image_rgbs, device)
    crop_records = _detect_crop_records(image_paths, image_rgbs, detections, crop_scale=DEFAULT_CROP_SCALE)

    images: list[np.ndarray] = []
    for camera_id, image_rgb in zip(camera_ids, image_rgbs):
        crop_record = crop_records[camera_id]
        crop = _crop_image_with_record(Image.fromarray(image_rgb), crop_record).resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
        images.append(np.asarray(crop, dtype=np.float32) / 255.0)
    return camera_ids, image_paths, image_rgbs, detections, np.stack(images, axis=0).astype(np.float32), crop_records


def _empty_masks(image_rgbs: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros(image.shape[:2], dtype=bool) for image in image_rgbs]


def _crop_mask_with_record(mask: np.ndarray, record: dict[str, int | float | str], image_size: int) -> np.ndarray:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")
    cropped = _crop_image_with_record(mask_image, record).convert("L")
    resized = cropped.resize((image_size, image_size), Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.float32)


def _parse_raw_face_masks(image_rgbs: list[np.ndarray], detections, device: torch.device) -> list[np.ndarray]:
    import facer
    from batch_face.face_parsing.farl import convert2facer

    if not image_rgbs:
        return []

    parser = facer.face_parser(DEFAULT_PARSER_MODEL, device=device)
    images = torch.stack([torch.from_numpy(image).permute(2, 0, 1) for image in image_rgbs], dim=0).to(device)

    with torch.inference_mode():
        if not detections or not any(detections):
            return _empty_masks(image_rgbs)
        faces = convert2facer(detections, device=str(device))
        if "image_ids" in faces:
            faces["image_ids"] = faces["image_ids"].long()
        faces = parser(images, faces)
        labels = faces["seg"]["logits"].argmax(dim=1)
        label_names = faces["seg"]["label_names"]

    keep_ids = [index for index, name in enumerate(label_names) if name in FACE_LABELS]
    if not keep_ids:
        return _empty_masks(image_rgbs)
    keep_tensor = torch.tensor(keep_ids, device=labels.device)
    parsed_masks = torch.isin(labels, keep_tensor).detach().cpu().numpy().astype(bool)

    output_masks = _empty_masks(image_rgbs)
    image_ids = faces["image_ids"].detach().cpu().numpy()
    for face_index, image_index in enumerate(image_ids):
        output_masks[int(image_index)] |= parsed_masks[face_index]
    return output_masks


def _predict_masks(
    image_rgbs: list[np.ndarray],
    detections,
    camera_ids: list[str],
    crop_records: dict[str, dict[str, int | float | str]],
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    raw_masks = _parse_raw_face_masks(image_rgbs, detections, device)
    masks = [
        _crop_mask_with_record(raw_mask, crop_records[camera_id], image_size=image_size)
        for camera_id, raw_mask in zip(camera_ids, raw_masks)
    ]
    return np.stack(masks, axis=0).astype(np.float32)


def _load_pixel3dmm_uv_model(pixel3dmm_root: Path, device: torch.device):
    pixel3dmm_root = pixel3dmm_root.expanduser().resolve()
    runtime_root = pixel3dmm_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PIXEL3DMM_CODE_BASE", str(pixel3dmm_root))
    os.environ.setdefault("PIXEL3DMM_PREPROCESSED_DATA", str(runtime_root / "preprocessed"))
    os.environ.setdefault("PIXEL3DMM_TRACKING_OUTPUT", str(runtime_root / "tracking_output"))

    import timm
    from torchvision import transforms
    from pixel3dmm.lightning import p3dmm_network
    from pixel3dmm.lightning.p3dmm_system import system as p3dmm_system

    # Newer torch builds can hit a native MHA fast path incompatibility in the
    # vendored Pixel3DMM runtime. Patch it at runtime instead of editing
    # third_party sources.
    p3dmm_network.MultiheadAttention = p3dmm_network.MultiheadAttention_cstm

    def build_dino_offline(model_name: str, proxy_error_retries: int = 0, proxy_error_cooldown: int = 0):
        model = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
        data_config = timm.data.resolve_model_data_config(model)
        processor = transforms.Normalize(mean=data_config["mean"], std=data_config["std"])
        return model, processor

    p3dmm_network.DinoWrapper._build_dino = staticmethod(build_dino_offline)
    checkpoint = pixel3dmm_root / "pretrained_weights" / "uv.ckpt"
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    cfg = checkpoint_data.get("hyper_parameters", {}).get("cfg")
    if cfg is None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(pixel3dmm_root / "configs" / "base.yaml")
        cfg.model.prediction_type = ["uv_map"]
    model = p3dmm_system.load_from_checkpoint(str(checkpoint), cfg=cfg, strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_uvs(images: np.ndarray, masks: np.ndarray, pixel3dmm_root: Path, device: torch.device) -> np.ndarray:
    model = _load_pixel3dmm_uv_model(pixel3dmm_root, device)
    try:
        uvs: list[np.ndarray] = []
        for image, mask in zip(images, masks):
            image_t = torch.from_numpy(image).to(device=device).float()[None, None]
            mask_t = torch.from_numpy(mask > 0.5).to(device=device).long()[None, None]
            batch = {"tar_msk": mask_t, "tar_rgb": image_t}
            mirrored = {"tar_rgb": torch.flip(image_t, dims=[3]), "tar_msk": torch.flip(mask_t, dims=[3])}
            output, _ = model.net(batch)
            output_mirrored, _ = model.net(mirrored)
            flipped_uv = torch.flip(output_mirrored["uv_map"], dims=[4])
            flipped_uv[:, :, 0, :, :] *= -1
            flipped_uv[:, :, 0, :, :] += 2 * 0.0075
            uv = torch.clamp((output["uv_map"] + flipped_uv) * 0.25 + 0.5, 0.0, 1.0)
            uvs.append(uv[0, 0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32))
        return np.stack(uvs, axis=0)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _resize_batch(images: np.ndarray, target_size: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR) for image in images],
        axis=0,
    ).astype(np.float32)


def _load_vggt_model(vggtface2_root: Path, device: torch.device):
    with _prepend_sys_path(vggtface2_root):
        from vggt.models.vggt import VGGT

    model = VGGT()
    model.load_state_dict(torch.load(vggtface2_root / "vggt_weights.pt", map_location="cpu"))
    model.eval()
    return model.to(device)


@torch.no_grad()
def _predict_point_maps(images: np.ndarray, vggtface2_root: Path, device: torch.device, image_size: int) -> np.ndarray:
    with _prepend_sys_path(vggtface2_root):
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = _load_vggt_model(vggtface2_root, device)
    try:
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        images_518 = _resize_batch(images, 518)
        image_t = torch.from_numpy(images_518).to(device).permute(0, 3, 1, 2)[None]
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(image_t.float())
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_t.shape[-2:])
        depth_map, _ = model.depth_head(aggregated_tokens_list, image_t, ps_idx)
        point_maps = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
        return _resize_batch(point_maps, image_size)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _write_internal_pkl(path: Path, images: np.ndarray, uvs: np.ndarray, masks: np.ndarray, point_maps: np.ndarray) -> None:
    _ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "imgs": images.astype(np.float32),
                "uvs": uvs.astype(np.float32),
                "masks": masks.astype(np.float32),
                "point_maps": point_maps.astype(np.float32),
            },
            handle,
        )


def build_input_from_images(config: ImageOnlyInputConfig) -> Path:
    image_dir = _resolve_image_dir(config.image_root)
    sample_id = config.sample_id or (image_dir.parent.name if image_dir.name == "image_select" else image_dir.name)
    sample_root = config.output_root / sample_id
    if sample_root.exists() and not config.overwrite:
        return sample_root

    _ensure_dir(sample_root)
    cache_root = sample_root / "cache"
    reproject_root = sample_root / "reproject"
    _reset_dir(cache_root)
    _reset_dir(reproject_root / "image")
    _reset_dir(reproject_root / "uv")
    _reset_dir(reproject_root / "mask")

    device = _device(config.device)
    camera_ids, image_paths, image_rgbs, detections, images, crop_records = _load_images(image_dir, config.views, config.image_size, device)
    (cache_root / "crop_metadata.json").write_text(
        json.dumps(crop_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    masks = _predict_masks(image_rgbs, detections, camera_ids, crop_records, config.image_size, device)
    uvs = _predict_uvs(images, masks, config.pixel3dmm_root, device)
    point_maps = _predict_point_maps(images, config.vggtface2_root, device, config.image_size)

    pkl_path = cache_root / "vggtface2_input.pkl"
    mesh_path = sample_root / "vggtface2_mesh.obj"
    _write_internal_pkl(pkl_path, images, uvs, masks, point_maps)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    infer_vggtface2_mesh(
        VGGTFace2MeshConfig(
            pkl_path=pkl_path,
            output_mesh_path=mesh_path,
            vggtface2_root=config.vggtface2_root,
            uv_size=config.image_size,
            num_views=images.shape[0],
            device=config.device,
            overwrite=True,
        )
    )

    for camera_id, image, uv, mask in zip(camera_ids, images, uvs, masks):
        _save_reprojection_view(image, uv, mask > 0.5, reproject_root, f"cam{camera_id}")

    _build_from_uv_buffers(
        sample_root=sample_root,
        image_dir=reproject_root / "image",
        uv_dir=reproject_root / "uv",
        mask_dir=reproject_root / "mask",
        texture_size=config.texture_size,
        preserve_reproject=True,
    )
    quality_path = sample_root / "input" / "partial_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality.update(
        {
            "sampling_mode": "image_only_predicted_uv",
            "source_images": str(image_dir),
            "internal_vggtface2_input": str(pkl_path),
            "vggtface2_mesh": str(mesh_path),
            "views_requested": int(config.views),
            "views_used": int(images.shape[0]),
            "image_size": int(config.image_size),
        }
    )
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sample_root
