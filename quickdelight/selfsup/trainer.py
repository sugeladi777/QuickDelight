from __future__ import annotations

"""Training loop for the self-supervised texture baseline."""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quickdelight.preview import save_preview
from quickdelight.texture_model import TextureCompletionNet

from .loss import masked_reprojection_loss


@dataclass(frozen=True)
class SelfSupTrainerConfig:
    save_root: Path
    epochs: int
    preview_every: int = 1
    use_amp: bool = True
    reprojection_weight: float = 1.0
    reprojection_loss_type: str = "l1"
    mask_erode_pixels: int = 0
    color_normalize: bool = False
    gradient_weight: float = 0.0
    charbonnier_eps: float = 1e-3
    grad_clip_norm: float = 1.0
    save_every: int = 1


class SelfSupTrainer:
    def __init__(self, model: TextureCompletionNet, optimizer, device: torch.device, config: SelfSupTrainerConfig, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler = self._make_grad_scaler()
        self.best_metric = float("inf")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        self.config.save_root.mkdir(parents=True, exist_ok=True)
        self._write_config()
        metrics_path = self.config.save_root / "metrics.jsonl"
        for epoch in range(1, self.config.epochs + 1):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self.evaluate(val_loader) if val_loader is not None else {}
            metrics = {"epoch": epoch, "lr": learning_rate}
            metrics |= {f"train_{key}": value for key, value in train_metrics.items()} | val_metrics
            self._append_metrics(metrics_path, metrics)
            print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()), flush=True)
            is_best = metrics.get("val_total", metrics["train_total"]) < self.best_metric
            if is_best:
                self.best_metric = metrics.get("val_total", metrics["train_total"])
            if self.config.save_every <= 0 or epoch % self.config.save_every == 0 or epoch == self.config.epochs:
                self._save_checkpoint("latest.pth", epoch, metrics)
            if is_best:
                self._save_checkpoint("best.pth", epoch, metrics)
            if val_loader is not None and self.config.preview_every > 0 and epoch % self.config.preview_every == 0:
                self._save_preview(val_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

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
            with self._autocast():
                prediction = self.model(inputs, masks)
                reproj = masked_reprojection_loss(
                    prediction,
                    target_images,
                    uv,
                    reproj_masks,
                    crop_box=crop_box,
                    full_size=texture_size,
                    loss_type=self.config.reprojection_loss_type,
                    mask_erode_pixels=self.config.mask_erode_pixels,
                    color_normalize=self.config.color_normalize,
                    gradient_weight=self.config.gradient_weight,
                    charbonnier_eps=self.config.charbonnier_eps,
                )
                loss = self.config.reprojection_weight * reproj["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss detected: {float(loss.detach().cpu())}")
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            count += 1
            sums["total"] = sums.get("total", 0.0) + float(loss.detach().cpu())
            sums["reprojection"] = sums.get("reprojection", 0.0) + float(reproj["total"].detach().cpu())
            sums["photometric"] = sums.get("photometric", 0.0) + float(reproj["photometric"].detach().cpu())
            sums["gradient"] = sums.get("gradient", 0.0) + float(reproj["gradient"].detach().cpu())
            sums["reprojection_ratio"] = sums.get("reprojection_ratio", 0.0) + float(reproj["valid_ratio"].detach().cpu())
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
        target_images = batch["reproject_images"].to(self.device)
        uv = batch["reproject_uv"].to(self.device)
        reproject_masks = batch["reproject_masks"].to(self.device)
        prediction = self.model(inputs, masks)
        save_preview(
            self.config.save_root / "previews" / f"epoch_{epoch:04d}",
            inputs,
            masks,
            prediction,
            crop_box,
            full_size=texture_size,
            target_images=target_images,
            uv=uv,
            reproject_masks=reproject_masks,
        )

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "metrics": metrics,
                "config": self._serializable_config(),
            },
            self.config.save_root / name,
        )

    def _write_config(self) -> None:
        with (self.config.save_root / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(self._serializable_config(), handle, ensure_ascii=False, indent=2)

    def _append_metrics(self, path: Path, metrics: dict[str, float]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def _serializable_config(self) -> dict[str, object]:
        config = asdict(self.config)
        config["save_root"] = str(config["save_root"])
        return config

    def _make_grad_scaler(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=self.use_amp)
        return torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)
