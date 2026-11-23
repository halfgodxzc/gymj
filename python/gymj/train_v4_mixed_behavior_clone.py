from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch

from .behavior_cloning import set_valued_nll, split_by_episode
from .prepare_v4_big_hand_training_view import (
    load_v4_big_hand_training_view,
)
from .training_device import DEVICE_CHOICES, resolve_training_device
from .train_v4_behavior_clone import (
    V4_DECISION_GROUP_NAMES,
    _decision_group_weights,
    _decision_groups,
    _report_line,
    _set_random_seed,
    evaluate_v4_behavior_clone,
)
from .v4_agent import (
    load_v4_actor_checkpoint,
    save_v4_actor_checkpoint,
)
from .v4_demonstration_dataset import (
    V4DemonstrationBatch,
    V4DemonstrationDataset,
    make_v4_data_loader,
)


BIG_HAND_ROUTE_NAMES = ("seven_pairs", "all_triplets", "flush")


@dataclass(frozen=True)
class V4MixedBehaviorCloneTrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    teacher_fraction: float = 0.7
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    min_learning_rate: float = 1.0e-6
    lr_factor: float = 0.5
    lr_patience: int = 2
    patience: int = 5
    balance_exponent: float = 0.5
    gradient_clip: float = 5.0
    max_teacher_balanced_drop: float = 0.01
    big_hand_route_shares: tuple[float, float, float] = (
        0.40,
        0.45,
        0.15,
    )
    big_hand_teacher_agreement_weight: float = 0.5
    big_hand_teacher_disagreement_weight: float = 2.0
    max_train_episodes: int = 0
    split_seed: int = 20260929
    seed: int = 20260929
    device: str = "auto"
    num_workers: int = 0
    cpu_threads: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size < 2:
            raise ValueError("epochs must be positive and batch size at least two")
        if not 0.0 < self.teacher_fraction < 1.0:
            raise ValueError("teacher fraction must be between zero and one")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer configuration is invalid")
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
        if not 0.0 <= self.max_teacher_balanced_drop <= 1.0:
            raise ValueError("maximum Teacher drop must be in range 0..1")
        if len(self.big_hand_route_shares) != len(BIG_HAND_ROUTE_NAMES):
            raise ValueError("BigHand route shares must contain three values")
        if any(value < 0.0 for value in self.big_hand_route_shares):
            raise ValueError("BigHand route shares cannot be negative")
        if not np.isclose(sum(self.big_hand_route_shares), 1.0):
            raise ValueError("BigHand route shares must sum to one")
        if (
            self.big_hand_teacher_agreement_weight <= 0.0
            or self.big_hand_teacher_disagreement_weight <= 0.0
        ):
            raise ValueError("BigHand agreement weights must be positive")
        if self.device not in DEVICE_CHOICES:
            raise ValueError("unknown training device")
        if self.num_workers < 0 or self.cpu_threads < 0:
            raise ValueError("worker and CPU thread counts cannot be negative")
        if self.max_train_episodes < 0 or self.max_train_episodes == 1:
            raise ValueError(
                "maximum training episodes must be zero or at least two"
            )


def train_v4_mixed_behavior_clone(
    teacher_dataset_path: Path | str,
    big_hand_dataset_path: Path | str,
    big_hand_view_path: Path | str,
    initial_checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    training_config: V4MixedBehaviorCloneTrainingConfig = (
        V4MixedBehaviorCloneTrainingConfig()
    ),
) -> dict[str, object]:
    _set_random_seed(training_config.seed)
    if training_config.cpu_threads > 0:
        torch.set_num_threads(training_config.cpu_threads)
    device = resolve_training_device(training_config.device)
    teacher_dataset = V4DemonstrationDataset(teacher_dataset_path)
    big_hand_dataset = V4DemonstrationDataset(big_hand_dataset_path)
    big_hand_view = load_v4_big_hand_training_view(
        big_hand_view_path,
        big_hand_dataset,
    )
    teacher_split = split_by_episode(
        teacher_dataset.episode_seeds,
        seed=training_config.split_seed,
    )
    teacher_indices = {
        "train": teacher_split.train_indices,
        "validation": teacher_split.validation_indices,
        "test": teacher_split.test_indices,
    }
    big_hand_indices = {
        name: np.asarray(big_hand_view[f"{name}_indices"], dtype=np.int64)
        for name in ("train", "validation", "test")
    }
    if training_config.max_train_episodes:
        teacher_episode_limit = min(
            training_config.max_train_episodes - 1,
            max(
                1,
                round(
                    training_config.max_train_episodes
                    * training_config.teacher_fraction
                ),
            ),
        )
        big_hand_episode_limit = (
            training_config.max_train_episodes - teacher_episode_limit
        )
        teacher_indices["train"] = _limit_indices_by_episode_count(
            teacher_dataset,
            teacher_indices["train"],
            teacher_episode_limit,
            training_config.seed,
        )
        big_hand_indices["train"] = _limit_indices_by_episode_count(
            big_hand_dataset,
            big_hand_indices["train"],
            big_hand_episode_limit,
            training_config.seed + 1,
        )
    teacher_batch_size, big_hand_batch_size = _source_batch_sizes(
        training_config.batch_size,
        training_config.teacher_fraction,
    )
    teacher_loaders = _source_loaders(
        teacher_dataset,
        teacher_indices,
        teacher_batch_size,
        training_config,
        seed_offset=0,
        pin_memory=device.type == "cuda",
    )
    big_hand_loaders = _source_loaders(
        big_hand_dataset,
        big_hand_indices,
        big_hand_batch_size,
        training_config,
        seed_offset=1000,
        pin_memory=device.type == "cuda",
    )
    actor, initial_checkpoint = load_v4_actor_checkpoint(
        initial_checkpoint_path,
        device=device,
    )
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
    teacher_group_weights = _decision_group_weights(
        teacher_dataset,
        teacher_indices["train"],
        training_config.balance_exponent,
        device,
    )
    big_hand_group_weights = _decision_group_weights(
        big_hand_dataset,
        big_hand_indices["train"],
        training_config.balance_exponent,
        device,
    )
    strategy_lookup, route_scales = _big_hand_strategy_lookup(
        big_hand_dataset,
        big_hand_view,
        big_hand_indices["train"],
        training_config,
    )
    evaluation_lookup, _ = _big_hand_strategy_lookup(
        big_hand_dataset,
        big_hand_view,
        np.asarray(big_hand_view["eligible_indices"], dtype=np.int64),
        training_config,
    )

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_v4_actor.pt"
    initial_teacher = evaluate_v4_behavior_clone(
        actor,
        teacher_loaders["validation"],
        device,
    )
    initial_big_hand = evaluate_v4_behavior_clone(
        actor,
        big_hand_loaders["validation"],
        device,
    )
    initial_big_hand_strategy = evaluate_big_hand_strategy(
        actor,
        big_hand_loaders["validation"],
        device,
        evaluation_lookup,
    )
    teacher_floor = max(
        0.0,
        float(initial_teacher["decision_balanced_accuracy"])
        - training_config.max_teacher_balanced_drop,
    )
    best_score = _mixed_validation_score(
        initial_teacher,
        initial_big_hand,
        training_config.teacher_fraction,
    )
    best_loss = _mixed_validation_loss(
        initial_teacher,
        initial_big_hand,
        training_config.teacher_fraction,
    )
    best_epoch = 0
    save_v4_actor_checkpoint(
        checkpoint_path,
        actor,
        metadata={
            "epoch": 0,
            "selection_score": best_score,
            "teacher_validation": initial_teacher,
            "big_hand_validation": initial_big_hand,
            "source_checkpoint": str(Path(initial_checkpoint_path).resolve()),
        },
    )

    history = []
    stale_epochs = 0
    started = time.perf_counter()
    print(f"device={device}")
    print(
        f"batch=teacher:{teacher_batch_size},big_hand:{big_hand_batch_size} "
        f"loss_mix={training_config.teacher_fraction:.2f}/"
        f"{1.0 - training_config.teacher_fraction:.2f}"
    )
    print(
        f"teacher_train={len(teacher_indices['train'])} "
        f"big_hand_train={len(big_hand_indices['train'])} "
        f"teacher_floor={teacher_floor:.4f}"
    )
    print(
        "route_scales="
        + ",".join(
            f"{name}:{route_scales[index]:.3f}"
            for index, name in enumerate(BIG_HAND_ROUTE_NAMES)
        )
    )
    print(
        _mixed_epoch_line(
            0,
            0.0,
            0.0,
            float(optimizer.param_groups[0]["lr"]),
            initial_teacher,
            initial_big_hand,
            best_score,
            True,
        )
    )

    for epoch in range(1, training_config.epochs + 1):
        train_report = _train_mixed_epoch(
            actor=actor,
            optimizer=optimizer,
            teacher_loader=teacher_loaders["train"],
            big_hand_loader=big_hand_loaders["train"],
            teacher_group_weights=teacher_group_weights,
            big_hand_group_weights=big_hand_group_weights,
            big_hand_strategy_lookup=strategy_lookup,
            teacher_fraction=training_config.teacher_fraction,
            gradient_clip=training_config.gradient_clip,
            device=device,
        )
        teacher_validation = evaluate_v4_behavior_clone(
            actor,
            teacher_loaders["validation"],
            device,
        )
        big_hand_validation = evaluate_v4_behavior_clone(
            actor,
            big_hand_loaders["validation"],
            device,
        )
        big_hand_strategy = evaluate_big_hand_strategy(
            actor,
            big_hand_loaders["validation"],
            device,
            evaluation_lookup,
        )
        score = _mixed_validation_score(
            teacher_validation,
            big_hand_validation,
            training_config.teacher_fraction,
        )
        validation_loss = _mixed_validation_loss(
            teacher_validation,
            big_hand_validation,
            training_config.teacher_fraction,
        )
        teacher_retained = (
            float(teacher_validation["decision_balanced_accuracy"])
            >= teacher_floor
        )
        improved = teacher_retained and (
            score > best_score
            or (score == best_score and validation_loss < best_loss)
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_report,
            "teacher_validation": teacher_validation,
            "big_hand_validation": big_hand_validation,
            "big_hand_strategy": big_hand_strategy,
            "selection_score": score,
            "teacher_retained": teacher_retained,
        })
        print(
            _mixed_epoch_line(
                epoch,
                train_report["teacher_loss"],
                train_report["big_hand_loss"],
                learning_rate,
                teacher_validation,
                big_hand_validation,
                score,
                teacher_retained,
            ),
            flush=True,
        )
        if improved:
            best_score = score
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            save_v4_actor_checkpoint(
                checkpoint_path,
                actor,
                metadata={
                    "epoch": epoch,
                    "selection_score": score,
                    "teacher_validation": teacher_validation,
                    "big_hand_validation": big_hand_validation,
                    "big_hand_strategy": big_hand_strategy,
                    "source_checkpoint": str(
                        Path(initial_checkpoint_path).resolve()
                    ),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                print(f"early_stop epoch={epoch}")
                break
        scheduler.step(score if teacher_retained else 0.0)

    actor, checkpoint = load_v4_actor_checkpoint(checkpoint_path, device=device)
    final_teacher_validation = evaluate_v4_behavior_clone(
        actor,
        teacher_loaders["validation"],
        device,
    )
    final_big_hand_validation = evaluate_v4_behavior_clone(
        actor,
        big_hand_loaders["validation"],
        device,
    )
    final_big_hand_strategy = evaluate_big_hand_strategy(
        actor,
        big_hand_loaders["validation"],
        device,
        evaluation_lookup,
    )
    teacher_test = evaluate_v4_behavior_clone(
        actor,
        teacher_loaders["test"],
        device,
    )
    big_hand_test = evaluate_v4_behavior_clone(
        actor,
        big_hand_loaders["test"],
        device,
    )
    big_hand_test_strategy = evaluate_big_hand_strategy(
        actor,
        big_hand_loaders["test"],
        device,
        evaluation_lookup,
    )
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": int(checkpoint["metadata"]["epoch"]),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "training_config": asdict(training_config),
        "initial_checkpoint": str(Path(initial_checkpoint_path).resolve()),
        "initial_checkpoint_epoch": initial_checkpoint.get("metadata", {}).get(
            "epoch"
        ),
        "teacher_dataset": str(Path(teacher_dataset_path).resolve()),
        "big_hand_dataset": str(Path(big_hand_dataset_path).resolve()),
        "big_hand_view": str(Path(big_hand_view_path).resolve()),
        "batch_sizes": {
            "teacher": teacher_batch_size,
            "big_hand": big_hand_batch_size,
        },
        "training_episodes": {
            "teacher": int(np.unique(
                teacher_dataset.episode_seeds[teacher_indices["train"]]
            ).size),
            "big_hand": int(np.unique(
                big_hand_dataset.episode_seeds[big_hand_indices["train"]]
            ).size),
        },
        "route_scales": {
            name: route_scales[index]
            for index, name in enumerate(BIG_HAND_ROUTE_NAMES)
        },
        "teacher_floor": teacher_floor,
        "initial": {
            "teacher_validation": initial_teacher,
            "big_hand_validation": initial_big_hand,
            "big_hand_strategy": initial_big_hand_strategy,
        },
        "validation": {
            "teacher": final_teacher_validation,
            "big_hand": final_big_hand_validation,
            "big_hand_strategy": final_big_hand_strategy,
        },
        "test": {
            "teacher": teacher_test,
            "big_hand": big_hand_test,
            "big_hand_strategy": big_hand_test_strategy,
        },
        "history": history,
        "selection": {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "teacher_balanced_accuracy_floor": teacher_floor,
        },
    }
    (output / "training_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    print("teacher validation " + _report_line(final_teacher_validation))
    print("big hand validation " + _report_line(final_big_hand_validation))
    print("teacher test       " + _report_line(teacher_test))
    print("big hand test      " + _report_line(big_hand_test))
    print(f"checkpoint={checkpoint_path.resolve()}")
    return result


def evaluate_big_hand_strategy(
    actor,
    loader,
    device: torch.device,
    metadata_lookup: Mapping[tuple[int, int, int], tuple[int, bool, float]],
) -> dict[str, object]:
    actor.eval()
    route_counts = {
        name: {"sample_count": 0, "correct_count": 0}
        for name in BIG_HAND_ROUTE_NAMES
    }
    disagreement_counts = {
        name: {"sample_count": 0, "correct_count": 0}
        for name in ("teacher_agreement", "teacher_disagreement")
    }
    with torch.inference_mode():
        for batch in loader:
            metadata = _batch_strategy_metadata(batch, metadata_lookup)
            route_ids = metadata[0]
            disagreements = metadata[1]
            moved = batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = actor(moved.features)
            predictions = logits.masked_fill(
                ~moved.supervision_mask,
                -torch.inf,
            ).argmax(dim=1)
            correct = moved.target_mask.gather(
                1,
                predictions.unsqueeze(1),
            ).squeeze(1).cpu()
            for route_id, name in enumerate(BIG_HAND_ROUTE_NAMES):
                selected = route_ids == route_id
                route_counts[name]["sample_count"] += int(selected.sum())
                route_counts[name]["correct_count"] += int(
                    correct[selected].sum()
                )
            for disagreement, name in (
                (False, "teacher_agreement"),
                (True, "teacher_disagreement"),
            ):
                selected = disagreements == disagreement
                disagreement_counts[name]["sample_count"] += int(
                    selected.sum()
                )
                disagreement_counts[name]["correct_count"] += int(
                    correct[selected].sum()
                )
    return {
        "routes": {
            name: _finalize_accuracy(counts)
            for name, counts in route_counts.items()
        },
        "teacher_relationship": {
            name: _finalize_accuracy(counts)
            for name, counts in disagreement_counts.items()
        },
    }


def _train_mixed_epoch(
    *,
    actor,
    optimizer,
    teacher_loader,
    big_hand_loader,
    teacher_group_weights: torch.Tensor,
    big_hand_group_weights: torch.Tensor,
    big_hand_strategy_lookup: Mapping[
        tuple[int, int, int], tuple[int, bool, float]
    ],
    teacher_fraction: float,
    gradient_clip: float,
    device: torch.device,
) -> dict[str, float]:
    actor.train()
    big_hand_iterator = iter(big_hand_loader)
    teacher_loss_sum = 0.0
    big_hand_loss_sum = 0.0
    teacher_samples = 0
    big_hand_samples = 0
    steps = 0
    for teacher_batch in teacher_loader:
        try:
            big_hand_batch = next(big_hand_iterator)
        except StopIteration:
            big_hand_iterator = iter(big_hand_loader)
            big_hand_batch = next(big_hand_iterator)
        strategy_weights = _batch_strategy_metadata(
            big_hand_batch,
            big_hand_strategy_lookup,
        )[2].to(device)
        teacher_batch = teacher_batch.to(
            device,
            non_blocking=device.type == "cuda",
        )
        big_hand_batch = big_hand_batch.to(
            device,
            non_blocking=device.type == "cuda",
        )
        optimizer.zero_grad(set_to_none=True)
        teacher_losses = set_valued_nll(
            actor(teacher_batch.features),
            teacher_batch.supervision_mask,
            teacher_batch.target_mask,
        )
        big_hand_losses = set_valued_nll(
            actor(big_hand_batch.features),
            big_hand_batch.supervision_mask,
            big_hand_batch.target_mask,
        )
        teacher_groups = _decision_groups(
            teacher_batch.category,
            teacher_batch.target_mask,
        )
        big_hand_groups = _decision_groups(
            big_hand_batch.category,
            big_hand_batch.target_mask,
        )
        teacher_weights = teacher_group_weights[teacher_groups]
        big_hand_weights = (
            big_hand_group_weights[big_hand_groups] * strategy_weights
        )
        teacher_loss = (
            teacher_losses * teacher_weights
        ).sum() / teacher_weights.sum()
        big_hand_loss = (
            big_hand_losses * big_hand_weights
        ).sum() / big_hand_weights.sum()
        loss = (
            teacher_fraction * teacher_loss
            + (1.0 - teacher_fraction) * big_hand_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
        optimizer.step()
        teacher_loss_sum += float(teacher_losses.detach().sum().item())
        big_hand_loss_sum += float(big_hand_losses.detach().sum().item())
        teacher_samples += len(teacher_batch)
        big_hand_samples += len(big_hand_batch)
        steps += 1
    return {
        "teacher_loss": teacher_loss_sum / max(1, teacher_samples),
        "big_hand_loss": big_hand_loss_sum / max(1, big_hand_samples),
        "teacher_samples": teacher_samples,
        "big_hand_samples": big_hand_samples,
        "steps": steps,
    }


def _source_loaders(
    dataset: V4DemonstrationDataset,
    indices: Mapping[str, np.ndarray],
    batch_size: int,
    config: V4MixedBehaviorCloneTrainingConfig,
    *,
    seed_offset: int,
    pin_memory: bool,
) -> dict[str, object]:
    return {
        name: make_v4_data_loader(
            dataset,
            split_indices,
            batch_size=batch_size,
            shuffle=name == "train",
            seed=config.seed + seed_offset,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        for name, split_indices in indices.items()
    }


def _big_hand_strategy_lookup(
    dataset: V4DemonstrationDataset,
    view: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
    config: V4MixedBehaviorCloneTrainingConfig,
) -> tuple[dict[tuple[int, int, int], tuple[int, bool, float]], list[float]]:
    eligible = np.asarray(view["eligible_indices"], dtype=np.int64)
    route_ids = np.asarray(view["eligible_route_ids"], dtype=np.uint8)
    disagreements = np.asarray(
        view["eligible_teacher_target_disagreement"],
        dtype=np.bool_,
    )
    positions = np.full(len(dataset), -1, dtype=np.int64)
    positions[eligible] = np.arange(len(eligible), dtype=np.int64)
    selected_positions = positions[selected_indices]
    if np.any(selected_positions < 0):
        raise ValueError("BigHand selected indices are outside the training view")
    selected_routes = route_ids[selected_positions]
    counts = np.bincount(
        selected_routes,
        minlength=len(BIG_HAND_ROUTE_NAMES),
    ).astype(np.float64)
    if np.any((counts == 0) & (np.asarray(config.big_hand_route_shares) > 0)):
        raise ValueError("BigHand split is missing a requested route")
    observed = counts / counts.sum()
    route_scales = [
        float(target / actual) if actual > 0.0 else 0.0
        for target, actual in zip(config.big_hand_route_shares, observed)
    ]
    lookup = {}
    for sample_index in selected_indices:
        position = positions[int(sample_index)]
        route_id = int(route_ids[position])
        disagreement = bool(disagreements[position])
        relation_weight = (
            config.big_hand_teacher_disagreement_weight
            if disagreement
            else config.big_hand_teacher_agreement_weight
        )
        key = (
            int(dataset.episode_seeds[sample_index]),
            int(dataset.player_ids[sample_index]),
            int(dataset.decision_indices[sample_index]),
        )
        lookup[key] = (
            route_id,
            disagreement,
            route_scales[route_id] * relation_weight,
        )
    if len(lookup) != len(selected_indices):
        raise ValueError("BigHand strategy lookup contains duplicate samples")
    return lookup, route_scales


def _batch_strategy_metadata(
    batch: V4DemonstrationBatch,
    lookup: Mapping[tuple[int, int, int], tuple[int, bool, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = []
    for seed, player, decision in zip(
        batch.episode_seed.tolist(),
        batch.player_id.tolist(),
        batch.decision_index.tolist(),
    ):
        key = (int(seed), int(player), int(decision))
        try:
            values.append(lookup[key])
        except KeyError as error:
            raise ValueError(f"BigHand batch sample is absent from view: {key}") from error
    return (
        torch.tensor([value[0] for value in values], dtype=torch.long),
        torch.tensor([value[1] for value in values], dtype=torch.bool),
        torch.tensor([value[2] for value in values], dtype=torch.float32),
    )


def _source_batch_sizes(
    batch_size: int,
    teacher_fraction: float,
) -> tuple[int, int]:
    if batch_size < 2:
        raise ValueError("mixed batch size must be at least two")
    if not 0.0 < teacher_fraction < 1.0:
        raise ValueError("teacher fraction must be between zero and one")
    teacher = min(batch_size - 1, max(1, round(batch_size * teacher_fraction)))
    return teacher, batch_size - teacher


def _limit_indices_by_episode_count(
    dataset: V4DemonstrationDataset,
    indices: np.ndarray,
    episode_count: int,
    seed: int,
) -> np.ndarray:
    if episode_count <= 0:
        raise ValueError("episode limit must be positive")
    indices = np.asarray(indices, dtype=np.int64)
    episode_seeds = np.unique(dataset.episode_seeds[indices])
    if episode_count >= len(episode_seeds):
        return indices.copy()
    selected_seeds = np.random.default_rng(seed).choice(
        episode_seeds,
        size=episode_count,
        replace=False,
    )
    return indices[np.isin(dataset.episode_seeds[indices], selected_seeds)]


def _mixed_validation_score(
    teacher: Mapping[str, object],
    big_hand: Mapping[str, object],
    teacher_fraction: float,
) -> float:
    return (
        teacher_fraction * float(teacher["decision_balanced_accuracy"])
        + (1.0 - teacher_fraction)
        * float(big_hand["decision_balanced_accuracy"])
    )


def _mixed_validation_loss(
    teacher: Mapping[str, object],
    big_hand: Mapping[str, object],
    teacher_fraction: float,
) -> float:
    return (
        teacher_fraction * float(teacher["loss"])
        + (1.0 - teacher_fraction) * float(big_hand["loss"])
    )


def _finalize_accuracy(counts: Mapping[str, int]) -> dict[str, object]:
    sample_count = int(counts["sample_count"])
    correct_count = int(counts["correct_count"])
    return {
        "sample_count": sample_count,
        "correct_count": correct_count,
        "accuracy": correct_count / sample_count if sample_count else 0.0,
    }


def _mixed_epoch_line(
    epoch: int,
    teacher_loss: float,
    big_hand_loss: float,
    learning_rate: float,
    teacher_validation: Mapping[str, object],
    big_hand_validation: Mapping[str, object],
    score: float,
    teacher_retained: bool,
) -> str:
    return (
        f"epoch={epoch:02d} teacher_loss={teacher_loss:.4f} "
        f"big_hand_loss={big_hand_loss:.4f} lr={learning_rate:.2e} "
        f"teacher_balanced="
        f"{teacher_validation['decision_balanced_accuracy']:.3f} "
        f"big_hand_balanced="
        f"{big_hand_validation['decision_balanced_accuracy']:.3f} "
        f"selection={score:.3f} retained={int(teacher_retained)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune V4 from Teacher and filtered BigHand data."
    )
    parser.add_argument("teacher_dataset", type=Path)
    parser.add_argument("big_hand_dataset", type=Path)
    parser.add_argument("big_hand_view", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-fraction", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--balance-exponent", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-teacher-drop", type=float, default=0.01)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260929)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_v4_mixed_behavior_clone(
        args.teacher_dataset,
        args.big_hand_dataset,
        args.big_hand_view,
        args.initial_checkpoint,
        args.output,
        training_config=V4MixedBehaviorCloneTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            teacher_fraction=args.teacher_fraction,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            min_learning_rate=args.min_learning_rate,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            patience=args.patience,
            balance_exponent=args.balance_exponent,
            gradient_clip=args.gradient_clip,
            max_teacher_balanced_drop=args.max_teacher_drop,
            max_train_episodes=args.max_train_episodes,
            split_seed=args.split_seed,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            cpu_threads=args.cpu_threads,
        ),
    )


if __name__ == "__main__":
    main()
