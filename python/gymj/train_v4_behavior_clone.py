from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Iterable, Mapping

import numpy as np
import torch

from .behavior_cloning import set_valued_nll, split_by_episode
from .teacher import PASS_ACTION_ID, PON_ACTION_ID
from .training_device import DEVICE_CHOICES, resolve_training_device
from .v4_actor import V4Actor, V4ActorConfig
from .v4_agent import load_v4_actor_checkpoint, save_v4_actor_checkpoint
from .v4_demonstration_dataset import (
    V4_CATEGORY_NAMES,
    V4_DISCARD_CATEGORY,
    V4_HU_CATEGORY,
    V4_PON_PASS_CATEGORY,
    V4_READY_CATEGORY,
    V4DemonstrationBatch,
    V4DemonstrationDataset,
    make_v4_data_loader,
)


V4_DECISION_GROUP_NAMES = ("discard", "pon", "pass", "hu", "ready")


@dataclass(frozen=True)
class V4BehaviorCloneTrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    min_learning_rate: float = 1.0e-6
    lr_factor: float = 0.5
    lr_patience: int = 2
    patience: int = 6
    balance_exponent: float = 0.5
    gradient_clip: float = 5.0
    split_seed: int = 20260929
    seed: int = 20260929
    device: str = "auto"
    num_workers: int = 0
    cpu_threads: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay cannot be negative")
        if self.min_learning_rate < 0.0:
            raise ValueError("minimum learning rate cannot be negative")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("learning-rate factor must be between zero and one")
        if self.lr_patience < 0 or self.patience <= 0:
            raise ValueError("training patience values are invalid")
        if not 0.0 <= self.balance_exponent <= 1.0:
            raise ValueError("balance exponent must be in range 0..1")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient clip must be positive")
        if self.device not in DEVICE_CHOICES:
            raise ValueError("unknown training device")
        if self.num_workers < 0 or self.cpu_threads < 0:
            raise ValueError("worker and CPU thread counts cannot be negative")


def train_v4_behavior_clone(
    dataset_path: Path | str,
    output_path: Path | str,
    *,
    actor_config: V4ActorConfig = V4ActorConfig(),
    training_config: V4BehaviorCloneTrainingConfig = (
        V4BehaviorCloneTrainingConfig()
    ),
    initial_checkpoint_path: Path | str | None = None,
) -> dict[str, object]:
    _set_random_seed(training_config.seed)
    if training_config.cpu_threads > 0:
        torch.set_num_threads(training_config.cpu_threads)
    device = resolve_training_device(training_config.device)
    dataset = V4DemonstrationDataset(dataset_path)
    split = split_by_episode(
        dataset.episode_seeds,
        seed=training_config.split_seed,
    )
    loaders = {
        "train": make_v4_data_loader(
            dataset,
            split.train_indices,
            batch_size=training_config.batch_size,
            shuffle=True,
            seed=training_config.seed,
            num_workers=training_config.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "validation": make_v4_data_loader(
            dataset,
            split.validation_indices,
            batch_size=training_config.batch_size,
            shuffle=False,
            seed=training_config.seed,
            num_workers=training_config.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "test": make_v4_data_loader(
            dataset,
            split.test_indices,
            batch_size=training_config.batch_size,
            shuffle=False,
            seed=training_config.seed,
            num_workers=training_config.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }

    if initial_checkpoint_path is None:
        actor = V4Actor(actor_config).to(device)
        initial_checkpoint = None
    else:
        actor, loaded = load_v4_actor_checkpoint(
            initial_checkpoint_path,
            device=device,
        )
        initial_path = Path(initial_checkpoint_path).resolve()
        initial_checkpoint = {
            "path": str(initial_path),
            "best_epoch": loaded.get("metadata", {}).get("epoch"),
        }
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_config.lr_factor,
        patience=training_config.lr_patience,
        min_lr=training_config.min_learning_rate,
    )
    group_weights = _decision_group_weights(
        dataset,
        split.train_indices,
        training_config.balance_exponent,
        device,
    )

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_v4_actor.pt"
    history = []
    best_balanced_accuracy = -1.0
    best_validation_loss = float("inf")
    stale_epochs = 0
    started = time.perf_counter()

    print(f"device={device}")
    print(
        f"samples={len(dataset)} "
        f"train={len(split.train_indices)} "
        f"validation={len(split.validation_indices)} "
        f"test={len(split.test_indices)}"
    )
    print(
        "group_weights="
        + ",".join(
            f"{name}:{group_weights[index].item():.3f}"
            for index, name in enumerate(V4_DECISION_GROUP_NAMES)
        )
    )

    for epoch in range(1, training_config.epochs + 1):
        actor.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for batch in loaders["train"]:
            batch = batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            logits = actor(batch.features)
            losses = set_valued_nll(
                logits,
                batch.supervision_mask,
                batch.target_mask,
            )
            groups = _decision_groups(batch.category, batch.target_mask)
            weights = group_weights[groups]
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                training_config.gradient_clip,
            )
            optimizer.step()
            train_loss_sum += float(losses.detach().sum().item())
            train_sample_count += len(batch)

        validation = evaluate_v4_behavior_clone(
            actor,
            loaders["validation"],
            device,
        )
        train_loss = train_loss_sum / max(1, train_sample_count)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "validation": validation,
        })
        print(
            _epoch_line(epoch, train_loss, learning_rate, validation),
            flush=True,
        )

        balanced = float(validation["decision_balanced_accuracy"])
        validation_loss = float(validation["loss"])
        improved = (
            balanced > best_balanced_accuracy
            or (
                balanced == best_balanced_accuracy
                and validation_loss < best_validation_loss
            )
        )
        if improved:
            best_balanced_accuracy = balanced
            best_validation_loss = validation_loss
            stale_epochs = 0
            save_v4_actor_checkpoint(
                checkpoint_path,
                actor,
                metadata={
                    "epoch": epoch,
                    "validation": validation,
                    "training_config": asdict(training_config),
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
                print(f"early_stop epoch={epoch}")
                break
        scheduler.step(balanced)

    actor, checkpoint = load_v4_actor_checkpoint(
        checkpoint_path,
        device=device,
    )
    validation = evaluate_v4_behavior_clone(
        actor,
        loaders["validation"],
        device,
    )
    test = evaluate_v4_behavior_clone(actor, loaders["test"], device)
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": int(checkpoint["metadata"]["epoch"]),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "actor_parameters": actor.parameter_count(),
        "initial_checkpoint": initial_checkpoint,
        "split": {
            "train_episodes": len(split.train_seeds),
            "validation_episodes": len(split.validation_seeds),
            "test_episodes": len(split.test_seeds),
            "train_samples": len(split.train_indices),
            "validation_samples": len(split.validation_indices),
            "test_samples": len(split.test_indices),
        },
        "group_weights": {
            name: float(group_weights[index].item())
            for index, name in enumerate(V4_DECISION_GROUP_NAMES)
        },
        "validation": validation,
        "test": test,
        "history": history,
    }
    (output / "training_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    print("validation " + _report_line(validation))
    print("test       " + _report_line(test))
    print(f"checkpoint={checkpoint_path.resolve()}")
    print(f"elapsed_seconds={elapsed:.2f}")
    return result


def evaluate_v4_behavior_clone(
    actor: V4Actor,
    loader: Iterable[V4DemonstrationBatch],
    device: torch.device,
) -> dict[str, object]:
    actor.eval()
    aggregate = _empty_evaluation_counts()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = actor(batch.features)
            losses = set_valued_nll(
                logits,
                batch.supervision_mask,
                batch.target_mask,
            )
            predictions = logits.masked_fill(
                ~batch.supervision_mask,
                -torch.inf,
            ).argmax(dim=1)
            correct = batch.target_mask.gather(
                1,
                predictions.unsqueeze(1),
            ).squeeze(1)
            aggregate["sample_count"] += len(batch)
            aggregate["loss_sum"] += float(losses.sum().item())
            aggregate["correct_count"] += int(correct.sum().item())
            for category_id, name in enumerate(V4_CATEGORY_NAMES):
                selected = batch.category == category_id
                aggregate["categories"][name]["sample_count"] += int(
                    selected.sum().item()
                )
                aggregate["categories"][name]["correct_count"] += int(
                    correct[selected].sum().item()
                )
            groups = _decision_groups(batch.category, batch.target_mask)
            for group_id, name in enumerate(V4_DECISION_GROUP_NAMES):
                selected = groups == group_id
                aggregate["decision_groups"][name]["sample_count"] += int(
                    selected.sum().item()
                )
                aggregate["decision_groups"][name]["correct_count"] += int(
                    correct[selected].sum().item()
                )
            legal_predictions = logits.masked_fill(
                ~batch.features.selectable_actions,
                -torch.inf,
            ).argmax(dim=1)
            in_supervision = batch.supervision_mask.gather(
                1,
                legal_predictions.unsqueeze(1),
            ).squeeze(1)
            aggregate["outside_supervision_count"] += int(
                (~in_supervision).sum().item()
            )
    return _finalize_evaluation(aggregate)


def _decision_group_weights(
    dataset: V4DemonstrationDataset,
    indices: np.ndarray,
    exponent: float,
    device: torch.device,
) -> torch.Tensor:
    categories = torch.from_numpy(dataset.categories[indices]).long()
    targets = torch.from_numpy(dataset.target_masks[indices]).bool()
    groups = _decision_groups(categories, targets)
    counts = torch.bincount(
        groups,
        minlength=len(V4_DECISION_GROUP_NAMES),
    ).float()
    active = counts > 0
    weights = torch.zeros_like(counts)
    weights[active] = (
        counts[active].sum() / (active.sum() * counts[active])
    ).pow(exponent)
    return weights.to(device)


def _decision_groups(
    categories: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    groups = torch.full(
        categories.shape,
        -1,
        dtype=torch.long,
        device=categories.device,
    )
    groups[categories == V4_DISCARD_CATEGORY] = 0
    pon_pass = categories == V4_PON_PASS_CATEGORY
    groups[pon_pass & targets[:, PON_ACTION_ID]] = 1
    groups[pon_pass & targets[:, PASS_ACTION_ID]] = 2
    groups[categories == V4_HU_CATEGORY] = 3
    groups[categories == V4_READY_CATEGORY] = 4
    if torch.any(groups < 0):
        raise ValueError("some V4 samples have no decision group")
    return groups


def _empty_evaluation_counts() -> dict[str, object]:
    return {
        "sample_count": 0,
        "loss_sum": 0.0,
        "correct_count": 0,
        "outside_supervision_count": 0,
        "categories": {
            name: {"sample_count": 0, "correct_count": 0}
            for name in V4_CATEGORY_NAMES
        },
        "decision_groups": {
            name: {"sample_count": 0, "correct_count": 0}
            for name in V4_DECISION_GROUP_NAMES
        },
    }


def _finalize_evaluation(counts: Mapping[str, object]) -> dict[str, object]:
    sample_count = int(counts["sample_count"])
    categories = {
        name: _finalize_count_pair(counts["categories"][name])
        for name in V4_CATEGORY_NAMES
    }
    groups = {
        name: _finalize_count_pair(counts["decision_groups"][name])
        for name in V4_DECISION_GROUP_NAMES
    }
    active_group_accuracies = [
        group["accuracy"]
        for group in groups.values()
        if group["sample_count"] > 0
    ]
    return {
        "sample_count": sample_count,
        "loss": (
            float(counts["loss_sum"]) / sample_count
            if sample_count else 0.0
        ),
        "accuracy": _ratio(int(counts["correct_count"]), sample_count),
        "decision_balanced_accuracy": (
            sum(active_group_accuracies) / len(active_group_accuracies)
            if active_group_accuracies else 0.0
        ),
        "outside_supervision_rate": _ratio(
            int(counts["outside_supervision_count"]),
            sample_count,
        ),
        "categories": categories,
        "decision_groups": groups,
    }


def _finalize_count_pair(counts: Mapping[str, int]) -> dict[str, object]:
    count = int(counts["sample_count"])
    correct = int(counts["correct_count"])
    return {
        "sample_count": count,
        "correct_count": correct,
        "accuracy": _ratio(correct, count),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _epoch_line(
    epoch: int,
    train_loss: float,
    learning_rate: float,
    report: Mapping[str, object],
) -> str:
    groups = report["decision_groups"]
    return (
        f"epoch={epoch:02d} train_loss={train_loss:.4f} "
        f"val_loss={report['loss']:.4f} lr={learning_rate:.2e} "
        f"overall={report['accuracy']:.3f} "
        f"balanced={report['decision_balanced_accuracy']:.3f} "
        + " ".join(
            f"{name}={groups[name]['accuracy']:.3f}"
            for name in V4_DECISION_GROUP_NAMES
        )
    )


def _report_line(report: Mapping[str, object]) -> str:
    return (
        f"loss={report['loss']:.4f} "
        f"accuracy={report['accuracy']:.3f} "
        f"balanced={report['decision_balanced_accuracy']:.3f} "
        f"outside_supervision={report['outside_supervision_rate']:.3f}"
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V4 Actor from sharded demonstrations."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--balance-exponent", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--split-seed", type=int, default=20260929)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--initial-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_v4_behavior_clone(
        args.dataset,
        args.output,
        training_config=V4BehaviorCloneTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            min_learning_rate=args.min_learning_rate,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            patience=args.patience,
            balance_exponent=args.balance_exponent,
            gradient_clip=args.gradient_clip,
            split_seed=args.split_seed,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            cpu_threads=args.cpu_threads,
        ),
        initial_checkpoint_path=args.initial_checkpoint,
    )


if __name__ == "__main__":
    main()
