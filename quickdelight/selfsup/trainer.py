from __future__ import annotations

"""Training loop for self-supervised reprojection."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quickdelight.preview import save_preview
from quickdelight.texture_model import TextureCompletionNet

from .loss import masked_reprojection_l1


@dataclass(frozen=True)
class SelfSupTrainerConfig:
    save_root: Path
    epochs: int
    preview_every: int = 1
    use_amp: bool = True
    reprojection_weight: float = 1.0


class SelfSupTrainer:
    def __init__(self, model: TextureCompletionNet, optimizer, device: torch.device, config: SelfSupTrainerConfig):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        self.config.save_root.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self.evaluate(val_loader) if val_loader is not None else {}
            metrics = {f"train_{key}": value for key, value in train_metrics.items()} | val_metrics
            print("epoch=" + str(epoch) + " " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()), flush=True)
            torch.save(
                {"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "metrics": metrics},
                self.config.save_root / "latest.pth",
            )
            if val_loader is not None and epoch % self.config.preview_every == 0:
                self._save_preview(val_loader, epoch)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        sums: dict[str, float] = {}
        count = 0
        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            masks = batch["masks"].to(self.device)
            target_images = batch["reproject_images"].to(self.device)
            uv = batch["reproject_uv"].to(self.device)
            reproj_masks = batch["reproject_masks"].to(self.device)
            crop_box = batch["crop_box"].to(self.device)
            texture_size = int(batch["texture_size"][0].item())
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                prediction = self.model(inputs, masks)
                reproj_l1, reproj_ratio = masked_reprojection_l1(
                    prediction,
                    target_images,
                    uv,
                    reproj_masks,
                    crop_box=crop_box,
                    full_size=texture_size,
                )
                loss = self.config.reprojection_weight * reproj_l1
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            count += 1
            sums["total"] = sums.get("total", 0.0) + float(loss.detach().cpu())
            sums["reprojection_l1"] = sums.get("reprojection_l1", 0.0) + float(reproj_l1.detach().cpu())
            sums["reprojection_ratio"] = sums.get("reprojection_ratio", 0.0) + float(reproj_ratio.detach().cpu())
        return {key: value / max(1, count) for key, value in sums.items()}

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader | None) -> dict[str, float]:
        if loader is None:
            return {}
        metrics = self._run_epoch(loader, training=False)
        return {f"val_{key}": value for key, value in metrics.items()}

    @torch.inference_mode()
    def _save_preview(self, loader: DataLoader, epoch: int) -> None:
        batch = next(iter(loader))
        inputs = batch["inputs"].to(self.device)
        masks = batch["masks"].to(self.device)
        crop_box = batch["crop_box"].to(self.device)
        texture_size = int(batch["texture_size"][0].item())
        prediction = self.model(inputs, masks)
        save_preview(
            self.config.save_root / "previews" / f"epoch_{epoch:04d}",
            inputs,
            prediction,
            crop_box,
            full_size=texture_size,
        )
