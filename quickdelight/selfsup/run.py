from __future__ import annotations

"""Entry point for self-supervised texture training."""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quickdelight.texture_model import TextureCompletionNet
from quickdelight.utils.device import choose_device, data_loader_kwargs, seed_everything

from .dataset import SelfSupervisedTextureDataset, collate_selfsup_batch, discover_selfsup_records
from .trainer import SelfSupTrainer, SelfSupTrainerConfig


@dataclass(frozen=True)
class SelfSupervisedTrainingConfig:
    dataset_root: Path
    save_root: Path
    device: str = "cuda:0"
    epochs: int = 10
    batch_size: int = 1
    num_workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    base_channels: int = 32
    val_ratio: float = 0.1
    seed: int = 42
    max_samples: int | None = None
    crop_to_uv_mask: bool = True
    use_mask: bool = True
    use_amp: bool = True
    preview_every: int = 1
    reprojection_weight: float = 1.0
    reprojection_loss_type: str = "l1"
    mask_erode_pixels: int = 0
    color_normalize: bool = False
    gradient_weight: float = 0.0
    charbonnier_eps: float = 1e-3
    grad_clip_norm: float = 1.0
    save_every: int = 1
    use_scheduler: bool = True


def _split(records, val_ratio: float, seed: int):
    if len(records) <= 1:
        return records, records
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(len(records) * val_ratio)))
    val_indices = set(indices[:val_count])
    train = tuple(records[index] for index in indices if index not in val_indices)
    val = tuple(records[index] for index in indices if index in val_indices)
    return train or val, val


def run_self_supervised_training(config: SelfSupervisedTrainingConfig) -> None:
    seed_everything(config.seed)
    device = choose_device(config.device)
    records = discover_selfsup_records(config.dataset_root)
    if config.max_samples is not None:
        records = records[: config.max_samples]
    if not records:
        raise RuntimeError(f"no valid self-supervised samples found under {config.dataset_root}")
    train_records, val_records = _split(records, config.val_ratio, config.seed)
    loader_kwargs = data_loader_kwargs(device, config.num_workers)
    train_loader = DataLoader(
        SelfSupervisedTextureDataset(train_records, crop_to_uv_mask=config.crop_to_uv_mask),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_selfsup_batch,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        SelfSupervisedTextureDataset(val_records, crop_to_uv_mask=config.crop_to_uv_mask),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_selfsup_batch,
        **loader_kwargs,
    )
    model = TextureCompletionNet(base_channels=config.base_channels, use_mask=config.use_mask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs)) if config.use_scheduler else None
    trainer = SelfSupTrainer(
        model,
        optimizer,
        device,
        SelfSupTrainerConfig(
            save_root=config.save_root,
            epochs=config.epochs,
            preview_every=config.preview_every,
            use_amp=config.use_amp,
            reprojection_weight=config.reprojection_weight,
            reprojection_loss_type=config.reprojection_loss_type,
            mask_erode_pixels=config.mask_erode_pixels,
            color_normalize=config.color_normalize,
            gradient_weight=config.gradient_weight,
            charbonnier_eps=config.charbonnier_eps,
            grad_clip_norm=config.grad_clip_norm,
            save_every=config.save_every,
        ),
        scheduler=scheduler,
    )
    trainer.fit(train_loader, val_loader)
