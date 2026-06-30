from __future__ import annotations

"""Dataset for self-supervised texture training with image reprojection."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from quickdelight.input.paths import normalize_camera_id
from quickdelight.utils.uv_mask import load_fixed_uv_mask


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float() / 255.0


def _load_mask_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return (pil_to_tensor(image.convert("L")).float() / 255.0 > 0).float()


def _load_uv_tensor(path: Path) -> torch.Tensor:
    import numpy as np

    uv = np.load(path).astype("float32")
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError(f"expected UV buffer with shape (H,W,2), got {uv.shape}: {path}")
    return torch.from_numpy(uv).permute(2, 0, 1)


def _camera_sort_key(camera_id: str) -> tuple[int, str]:
    return (int(camera_id), camera_id) if camera_id.isdigit() else (10**12, camera_id)


def _discover_by_camera(root: Path, suffix: str) -> dict[str, Path]:
    return {
        normalize_camera_id(path.name): path
        for path in root.glob(f"cam*.{suffix}")
        if path.is_file()
    }


def _stack_images(paths: tuple[Path, ...]) -> torch.Tensor:
    return torch.stack([_load_rgb_tensor(path) for path in paths], dim=0)


def _stack_masks(paths: tuple[Path, ...]) -> torch.Tensor:
    return torch.stack([_load_mask_tensor(path) for path in paths], dim=0)


def _fixed_uv_mask(texture_size: int) -> torch.Tensor:
    mask = load_fixed_uv_mask(texture_size)
    return torch.from_numpy(mask.astype("float32")).unsqueeze(0)


def _compute_crop_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    mask_2d = mask.squeeze(0) > 0
    coords = torch.nonzero(mask_2d, as_tuple=False)
    if coords.numel() == 0:
        height, width = mask_2d.shape
        return 0, height, 0, width
    top = int(coords[:, 0].min().item())
    bottom = int(coords[:, 0].max().item()) + 1
    left = int(coords[:, 1].min().item())
    right = int(coords[:, 1].max().item()) + 1
    return top, bottom, left, right


def _crop_tensor(image: torch.Tensor, crop_box: tuple[int, int, int, int]) -> torch.Tensor:
    top, bottom, left, right = crop_box
    return image[..., top:bottom, left:right]


@dataclass(frozen=True)
class SelfSupSampleRecord:
    sample_id: str
    sample_dir: Path
    input_paths: tuple[Path, ...]
    input_mask_paths: tuple[Path, ...]
    image_paths: tuple[Path, ...]
    uv_paths: tuple[Path, ...]
    reproj_mask_paths: tuple[Path, ...]


def _record_if_complete(sample_dir: Path) -> SelfSupSampleRecord | None:
    inputs = _discover_by_camera(sample_dir / "input" / "partial", "png")
    input_masks = _discover_by_camera(sample_dir / "input" / "mask", "png")
    images = _discover_by_camera(sample_dir / "reproject" / "image", "png")
    uvs = _discover_by_camera(sample_dir / "reproject" / "uv", "npy")
    reproj_masks = _discover_by_camera(sample_dir / "reproject" / "mask", "png")
    camera_ids = set(inputs)
    if not camera_ids or any(set(paths) != camera_ids for paths in (input_masks, images, uvs, reproj_masks)):
        return None
    ordered = tuple(sorted(camera_ids, key=_camera_sort_key))
    return SelfSupSampleRecord(
        sample_dir.name,
        sample_dir,
        tuple(inputs[camera_id] for camera_id in ordered),
        tuple(input_masks[camera_id] for camera_id in ordered),
        tuple(images[camera_id] for camera_id in ordered),
        tuple(uvs[camera_id] for camera_id in ordered),
        tuple(reproj_masks[camera_id] for camera_id in ordered),
    )


def discover_selfsup_records(dataset_root: Path) -> tuple[SelfSupSampleRecord, ...]:
    return tuple(
        record
        for sample_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir())
        if (record := _record_if_complete(sample_dir)) is not None
    )


class SelfSupervisedTextureDataset(Dataset):
    def __init__(self, records: tuple[SelfSupSampleRecord, ...], crop_to_uv_mask: bool = True):
        self.records = records
        self.crop_to_uv_mask = crop_to_uv_mask
        self.gt_mask, self.crop_box = self._prepare_fixed_mask(records)

    @staticmethod
    def _prepare_fixed_mask(records: tuple[SelfSupSampleRecord, ...]) -> tuple[torch.Tensor | None, tuple[int, int, int, int] | None]:
        if not records:
            return None, None
        first_input = records[0].input_paths[0]
        with Image.open(first_input) as image:
            width, height = image.size
        if width != height:
            raise ValueError(f"expected square UV input, got {width}x{height}: {first_input}")
        gt_mask = _fixed_uv_mask(width)
        return gt_mask, _compute_crop_box(gt_mask)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self.gt_mask is None or self.crop_box is None:
            raise RuntimeError("dataset was created without valid records")
        record = self.records[index]
        gt_mask = self.gt_mask.clone()
        crop_box = self.crop_box
        inputs = _stack_images(record.input_paths)
        masks = _stack_masks(record.input_mask_paths) * gt_mask.unsqueeze(0)
        texture_size = int(gt_mask.shape[-1])
        images = torch.stack([_load_rgb_tensor(path) for path in record.image_paths], dim=0)
        uv = torch.stack([_load_uv_tensor(path) for path in record.uv_paths], dim=0)
        reproj_masks = torch.stack([_load_mask_tensor(path) for path in record.reproj_mask_paths], dim=0)
        if self.crop_to_uv_mask:
            inputs = _crop_tensor(inputs, crop_box)
            masks = _crop_tensor(masks, crop_box)
            gt_mask = _crop_tensor(gt_mask, crop_box)
        return {
            "sample_id": record.sample_id,
            "inputs": inputs,
            "masks": masks,
            "gt_mask": gt_mask,
            "reproject_images": images,
            "reproject_uv": uv,
            "reproject_masks": reproj_masks,
            "crop_box": torch.tensor(crop_box, dtype=torch.long),
            "texture_size": torch.tensor(texture_size, dtype=torch.long),
        }


def collate_selfsup_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    max_views = max(item["inputs"].shape[0] for item in batch)
    inputs = []
    masks = []
    sample_ids = []
    gt_masks = []
    images = []
    uv = []
    reproj_masks = []
    crop_boxes = []
    texture_sizes = []
    for item in batch:
        view_padding = max_views - item["inputs"].shape[0]
        if view_padding:
            inputs.append(pad(item["inputs"], (0, 0, 0, 0, 0, 0, 0, view_padding)))
            masks.append(pad(item["masks"], (0, 0, 0, 0, 0, 0, 0, view_padding)))
            images.append(pad(item["reproject_images"], (0, 0, 0, 0, 0, 0, 0, view_padding)))
            uv.append(pad(item["reproject_uv"], (0, 0, 0, 0, 0, 0, 0, view_padding)))
            reproj_masks.append(pad(item["reproject_masks"], (0, 0, 0, 0, 0, 0, 0, view_padding)))
        else:
            inputs.append(item["inputs"])
            masks.append(item["masks"])
            images.append(item["reproject_images"])
            uv.append(item["reproject_uv"])
            reproj_masks.append(item["reproject_masks"])
        sample_ids.append(item["sample_id"])
        gt_masks.append(item["gt_mask"])
        crop_boxes.append(item["crop_box"])
        texture_sizes.append(item["texture_size"])
    return {
        "sample_id": sample_ids,
        "inputs": torch.stack(inputs, dim=0),
        "masks": torch.stack(masks, dim=0),
        "gt_mask": torch.stack(gt_masks, dim=0),
        "reproject_images": torch.stack(images, dim=0),
        "reproject_uv": torch.stack(uv, dim=0),
        "reproject_masks": torch.stack(reproj_masks, dim=0),
        "crop_box": torch.stack(crop_boxes, dim=0),
        "texture_size": torch.stack(texture_sizes, dim=0),
    }
