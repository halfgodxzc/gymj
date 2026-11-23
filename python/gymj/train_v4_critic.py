from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .behavior_cloning import split_by_episode
from .training_device import DEVICE_CHOICES, resolve_training_device
from .v4_agent import load_v4_actor_checkpoint
from .v4_critic import (
    V4PrivilegedCriticConfig,
    V4PrivilegedGuiyangCritic,
    load_v4_privileged_critic_checkpoint,
    save_v4_privileged_critic_checkpoint,
)
from .v4_critic_dataset import (
    V4_CRITIC_DATASET_FORMAT_VERSION,
    V4CriticBatch,
    V4CriticDataset,
    make_v4_critic_data_loader,
)


@dataclass(frozen=True)
class V4CriticTrainingConfig:
    epochs: int = 8
    warmup_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    public_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    shanten_loss_weight: float = 0.25
    gradient_clip: float = 1.0
    patience: int = 3
    seed: int = 20260930
    num_workers: int = 0

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("warmup epochs must be in range 0..epochs")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.learning_rate <= 0.0 or self.public_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay cannot be negative")
        if self.shanten_loss_weight < 0.0:
            raise ValueError("shanten loss weight cannot be negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient clip must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.num_workers < 0:
            raise ValueError("num workers cannot be negative")


def train_v4_critic(
    dataset_path: Path | str,
    actor_checkpoint: Path | str,
    output_path: Path | str,
    *,
    training_config: V4CriticTrainingConfig = V4CriticTrainingConfig(),
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    training_config.validate()
    device = torch.device(device)
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)

    dataset = V4CriticDataset(dataset_path)
    split = split_by_episode(
        dataset.episode_seeds,
        seed=training_config.seed,
    )
    loader_options = {
        "batch_size": training_config.batch_size,
        "seed": training_config.seed,
        "num_workers": training_config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        "train": make_v4_critic_data_loader(
            dataset,
            split.train_indices,
            shuffle=True,
            **loader_options,
        ),
        "train_evaluation": make_v4_critic_data_loader(
            dataset,
            split.train_indices,
            shuffle=False,
            **loader_options,
        ),
        "validation": make_v4_critic_data_loader(
            dataset,
            split.validation_indices,
            shuffle=False,
            **loader_options,
        ),
        "test": make_v4_critic_data_loader(
            dataset,
            split.test_indices,
            shuffle=False,
            **loader_options,
        ),
    }

    actor, actor_data = load_v4_actor_checkpoint(
        actor_checkpoint,
        device="cpu",
    )
    critic_config = V4PrivilegedCriticConfig(
        public_config=actor.config,
    )
    critic = V4PrivilegedGuiyangCritic(critic_config)
    critic.initialize_public_encoder(actor)
    critic.to(device)

    public_parameters = list(critic.public_encoder.parameters())
    public_ids = {id(parameter) for parameter in public_parameters}
    privileged_parameters = [
        parameter
        for parameter in critic.parameters()
        if id(parameter) not in public_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": public_parameters,
                "lr": training_config.public_learning_rate,
            },
            {
                "params": privileged_parameters,
                "lr": training_config.learning_rate,
            },
        ],
        weight_decay=training_config.weight_decay,
    )
    target_means = _target_means(loaders["train_evaluation"])

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_v4_critic.pt"
    history = []
    best_validation_loss = math.inf
    stale_epochs = 0
    started = time.perf_counter()

    print(
        f"device={device} samples={len(dataset)} "
        f"train={len(split.train_indices)} "
        f"validation={len(split.validation_indices)} "
        f"test={len(split.test_indices)}",
        flush=True,
    )
    for epoch in range(1, training_config.epochs + 1):
        public_trainable = epoch > training_config.warmup_epochs
        for parameter in public_parameters:
            parameter.requires_grad_(public_trainable)

        train_metrics = _run_epoch(
            critic,
            loaders["train"],
            device,
            target_means,
            training_config,
            optimizer=optimizer,
        )
        validation_metrics = _run_epoch(
            critic,
            loaders["validation"],
            device,
            target_means,
            training_config,
        )
        record = {
            "epoch": epoch,
            "public_encoder_trainable": public_trainable,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)

        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stale_epochs = 0
            save_v4_privileged_critic_checkpoint(
                checkpoint_path,
                critic,
                metadata={
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "training_config": asdict(training_config),
                    "v4_critic_dataset_format_version": (
                        V4_CRITIC_DATASET_FORMAT_VERSION
                    ),
                    "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
                    "actor_metadata": dict(actor_data.get("metadata", {})),
                    "target_means": target_means,
                    "split_seeds": {
                        "train": split.train_seeds,
                        "validation": split.validation_seeds,
                        "test": split.test_seeds,
                    },
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                print(f"early_stop epoch={epoch}", flush=True)
                break

    critic, checkpoint = load_v4_privileged_critic_checkpoint(
        checkpoint_path,
        device=device,
    )
    validation_metrics = _run_epoch(
        critic,
        loaders["validation"],
        device,
        target_means,
        training_config,
    )
    test_metrics = _run_epoch(
        critic,
        loaders["test"],
        device,
        target_means,
        training_config,
    )
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": int(checkpoint["metadata"]["epoch"]),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "sample_count": len(dataset),
        "parameter_count": sum(
            parameter.numel() for parameter in critic.parameters()
        ),
        "target_means": target_means,
        "split": {
            "train_episodes": len(split.train_seeds),
            "validation_episodes": len(split.validation_seeds),
            "test_episodes": len(split.test_seeds),
            "train_samples": len(split.train_indices),
            "validation_samples": len(split.validation_indices),
            "test_samples": len(split.test_indices),
        },
        "training_config": asdict(training_config),
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (output / "training_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return result


def _target_means(loader: Iterable[V4CriticBatch]) -> dict[str, float]:
    count = 0
    score_sum = 0.0
    shanten_sum = 0.0
    for batch in loader:
        count += len(batch)
        score_sum += float(batch.score_return.double().sum())
        shanten_sum += float(batch.shanten_return.double().sum())
    if count == 0:
        raise ValueError("V4 Critic training split is empty")
    return {
        "score": score_sum / count,
        "shanten": shanten_sum / count,
    }


def _run_epoch(
    critic: V4PrivilegedGuiyangCritic,
    loader: Iterable[V4CriticBatch],
    device: torch.device,
    target_means: dict[str, float],
    config: V4CriticTrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    critic.train(training)
    totals = {
        "count": 0.0,
        "loss": 0.0,
        "score_squared_error": 0.0,
        "score_absolute_error": 0.0,
        "score_baseline_squared_error": 0.0,
        "shanten_squared_error": 0.0,
        "shanten_absolute_error": 0.0,
        "shanten_baseline_squared_error": 0.0,
        "score_prediction_sum": 0.0,
        "score_target_sum": 0.0,
        "score_prediction_square_sum": 0.0,
        "score_target_square_sum": 0.0,
        "score_cross_sum": 0.0,
    }
    mse = nn.MSELoss()

    for cpu_batch in loader:
        batch = cpu_batch.to(
            device,
            non_blocking=device.type == "cuda",
        )
        with torch.set_grad_enabled(training):
            values = critic(batch.public, batch.privileged)
            score_loss = mse(values.score_value, batch.score_return)
            shanten_loss = mse(
                values.shanten_value,
                batch.shanten_return,
            )
            loss = score_loss + config.shanten_loss_weight * shanten_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(),
                    config.gradient_clip,
                )
                optimizer.step()

        score_prediction = values.score_value.detach().double()
        score_target = batch.score_return.detach().double()
        shanten_prediction = values.shanten_value.detach().double()
        shanten_target = batch.shanten_return.detach().double()
        score_error = score_prediction - score_target
        shanten_error = shanten_prediction - shanten_target
        count = len(batch)
        totals["count"] += count
        totals["loss"] += float(loss.detach()) * count
        totals["score_squared_error"] += float(score_error.square().sum())
        totals["score_absolute_error"] += float(score_error.abs().sum())
        totals["score_baseline_squared_error"] += float(
            (score_target - target_means["score"]).square().sum()
        )
        totals["shanten_squared_error"] += float(
            shanten_error.square().sum()
        )
        totals["shanten_absolute_error"] += float(
            shanten_error.abs().sum()
        )
        totals["shanten_baseline_squared_error"] += float(
            (shanten_target - target_means["shanten"]).square().sum()
        )
        totals["score_prediction_sum"] += float(score_prediction.sum())
        totals["score_target_sum"] += float(score_target.sum())
        totals["score_prediction_square_sum"] += float(
            score_prediction.square().sum()
        )
        totals["score_target_square_sum"] += float(
            score_target.square().sum()
        )
        totals["score_cross_sum"] += float(
            (score_prediction * score_target).sum()
        )

    count = totals["count"]
    if count == 0:
        raise ValueError("V4 Critic data loader is empty")
    score_mse = totals["score_squared_error"] / count
    score_baseline_mse = totals["score_baseline_squared_error"] / count
    shanten_mse = totals["shanten_squared_error"] / count
    shanten_baseline_mse = (
        totals["shanten_baseline_squared_error"] / count
    )
    return {
        "loss": totals["loss"] / count,
        "score_mse": score_mse,
        "score_rmse": math.sqrt(score_mse),
        "score_mae": totals["score_absolute_error"] / count,
        "score_baseline_mse": score_baseline_mse,
        "score_mse_improvement": _relative_improvement(
            score_mse,
            score_baseline_mse,
        ),
        "score_correlation": _correlation(totals),
        "shanten_mse": shanten_mse,
        "shanten_rmse": math.sqrt(shanten_mse),
        "shanten_mae": totals["shanten_absolute_error"] / count,
        "shanten_baseline_mse": shanten_baseline_mse,
        "shanten_mse_improvement": _relative_improvement(
            shanten_mse,
            shanten_baseline_mse,
        ),
    }


def _relative_improvement(error: float, baseline: float) -> float:
    if baseline <= 1.0e-12:
        return 0.0
    return 1.0 - error / baseline


def _correlation(totals: dict[str, float]) -> float:
    count = totals["count"]
    covariance = (
        totals["score_cross_sum"]
        - totals["score_prediction_sum"] * totals["score_target_sum"] / count
    )
    prediction_variance = (
        totals["score_prediction_square_sum"]
        - totals["score_prediction_sum"] ** 2 / count
    )
    target_variance = (
        totals["score_target_square_sum"]
        - totals["score_target_sum"] ** 2 / count
    )
    denominator = math.sqrt(
        max(0.0, prediction_variance) * max(0.0, target_variance)
    )
    return covariance / denominator if denominator > 0.0 else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the structured privileged V4 Critic."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--public-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--shanten-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=14)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    torch.set_num_threads(args.torch_threads)
    train_v4_critic(
        args.dataset,
        args.actor,
        args.output,
        training_config=V4CriticTrainingConfig(
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            public_learning_rate=args.public_learning_rate,
            weight_decay=args.weight_decay,
            shanten_loss_weight=args.shanten_loss_weight,
            gradient_clip=args.gradient_clip,
            patience=args.patience,
            seed=args.seed,
            num_workers=args.num_workers,
        ),
        device=resolve_training_device(args.device),
    )


if __name__ == "__main__":
    main()
