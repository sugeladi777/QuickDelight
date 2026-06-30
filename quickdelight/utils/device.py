from __future__ import annotations

"""Runtime helpers for devices, randomness, and DataLoader options."""

import random

import numpy as np
import torch


def normalize_torch_device(device: str | None = None) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    value = str(device).strip()
    if value.isdigit():
        return f"cuda:{value}"
    return value


def choose_device(device: str | None = None) -> torch.device:
    return torch.device(normalize_torch_device(device))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_loader_kwargs(device: torch.device, num_workers: int) -> dict[str, object]:
    return {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
