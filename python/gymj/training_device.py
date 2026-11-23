from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import torch


DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_training_device(name: str) -> torch.device:
    if name not in DEVICE_CHOICES:
        raise ValueError(
            f"device must be one of {', '.join(DEVICE_CHOICES)}"
        )
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but the installed PyTorch runtime cannot use it"
        )
    return device


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {
        name: value.to(device, non_blocking=non_blocking)
        for name, value in batch.items()
    }


def checkpoint_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {
            key: checkpoint_to_cpu(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(checkpoint_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [checkpoint_to_cpu(item) for item in value]
    return value


def atomic_torch_save_cpu(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_to_cpu(value), temporary)
    os.replace(temporary, path)


__all__ = [
    "DEVICE_CHOICES",
    "atomic_torch_save_cpu",
    "checkpoint_to_cpu",
    "move_batch_to_device",
    "resolve_training_device",
]
