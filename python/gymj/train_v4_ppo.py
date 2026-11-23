from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .ppo import clipped_policy_objective
from .gae import complete_episode_gae
from .training_device import DEVICE_CHOICES, resolve_training_device
from .v4_agent import (
    load_v4_actor_checkpoint,
    save_v4_actor_checkpoint,
)
from .v4_critic import (
    V4FourSeatCritic,
    load_v4_privileged_critic_checkpoint,
    save_v4_privileged_critic_checkpoint,
    v4_critic_public_from_actor_batch,
)
from .v4_ppo_dataset import (
    V4PPODataset,
    collate_v4_ppo_samples,
    make_v4_ppo_data_loader,
)


V4_PPO_TRAINING_VERSION = 1


@dataclass(frozen=True)
class V4PPOTrainingConfig:
    ppo_epochs: int = 2
    batch_size: int = 128
    actor_learning_rate: float = 3.0e-5
    actor_encoder_learning_rate: float = 3.0e-6
    actor_scorer_learning_rate: float = 1.0e-5
    kan_adapter_learning_rate: float = 3.0e-5
    critic_learning_rate: float = 3.0e-5
    final_learning_rate_fraction: float = 0.1
    weight_decay: float = 1.0e-4
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 1.0
    shanten_weight: float = 0.25
    shanten_value_loss_weight: float = 0.25
    target_kl: float = 0.015
    kl_early_stopping: bool = True
    gradient_clip: float = 1.0
    temperature: float = 1.0
    seed: int = 20260930
    num_workers: int = 0
    old_log_probability_tolerance: float = 0.05
    actor_train_scope: str = "all"
    kl_check_interval: int = 0
    kl_check_samples: int = 4096
    save_epoch_checkpoints: bool = False
    advantage_method: str = "mc"
    gae_lambda: float = 0.95
    chicken_attribution_weight: float = 0.0
    call_replay_directory: str | None = None
    call_replay_weight: float = 0.0
    call_replay_batch_size: int = 16
    call_replay_bonus_cap: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.call_replay_weight) or self.call_replay_weight < 0:
            raise ValueError("call replay weight must be finite and nonnegative")
        if self.call_replay_batch_size < 1 or not math.isfinite(self.call_replay_bonus_cap) or self.call_replay_bonus_cap <= 0:
            raise ValueError("invalid call replay batch size or cap")
        if self.call_replay_weight and (not self.call_replay_directory or self.actor_train_scope != "all" or not self.kl_check_interval):
            raise ValueError("call replay requires a directory, all-Actor scope and periodic KL checks")
        if not math.isfinite(self.chicken_attribution_weight) or self.chicken_attribution_weight < 0:
            raise ValueError("chicken attribution weight must be finite and nonnegative")
        if self.advantage_method not in {"mc", "gae"}:
            raise ValueError("advantage method must be mc or gae")
        if not math.isfinite(self.gae_lambda) or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae lambda must be finite and in [0, 1]")
        positive = (
            "ppo_epochs",
            "batch_size",
            "actor_learning_rate",
            "actor_encoder_learning_rate",
            "actor_scorer_learning_rate",
            "kan_adapter_learning_rate",
            "critic_learning_rate",
            "clip_range",
            "value_loss_coefficient",
            "target_kl",
            "gradient_clip",
            "temperature",
            "old_log_probability_tolerance",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip range must be between zero and one")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy coefficient cannot be negative")
        if self.shanten_weight < 0.0:
            raise ValueError("shanten weight cannot be negative")
        if self.shanten_value_loss_weight < 0.0:
            raise ValueError("shanten value loss weight cannot be negative")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay cannot be negative")
        if not 0.0 <= self.final_learning_rate_fraction <= 1.0:
            raise ValueError(
                "final learning-rate fraction must be between zero and one"
            )
        if self.num_workers < 0:
            raise ValueError("num workers cannot be negative")
        if self.kl_check_interval < 0 or self.kl_check_samples <= 0:
            raise ValueError("KL interval must be nonnegative and sample count positive")
        if self.actor_train_scope not in {
            "all",
            "kan_adapter",
            "joint_grouped",
        }:
            raise ValueError(
                "actor train scope must be all, kan_adapter, or joint_grouped"
            )


def train_v4_ppo_iteration(
    dataset_directory: Path | str,
    actor_checkpoint: Path | str,
    critic_checkpoint: Path | str,
    output_directory: Path | str,
    *,
    config: V4PPOTrainingConfig = V4PPOTrainingConfig(),
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    config.validate()
    call_auxiliary = None
    if config.call_replay_weight:
        from .call_replay_training import CallReplayAuxiliary
        call_auxiliary = CallReplayAuxiliary(config.call_replay_directory, seed=config.seed)
    device = torch.device(device)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "training_result.json"
    if result_path.exists():
        raise FileExistsError(f"completed PPO output already exists: {output}")

    dataset_directory = Path(dataset_directory)
    manifest_path = dataset_directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("V4 PPO dataset has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if config.chicken_attribution_weight:
        if manifest.get("chicken_attribution_version") != 1:
            raise ValueError("chicken bonus requires a dataset with attribution version 1")
        if float(manifest.get("score_divisor", math.nan)) != 30.0:
            raise ValueError("chicken bonus currently requires score_divisor=30")
    if config.advantage_method == "gae" and manifest.get("trajectory_contract") != "complete_candidate_player_decisions_v1":
        raise ValueError("GAE requires complete candidate-player trajectories; truncated or unverified datasets are unsupported")
    if manifest.get("shanten_reward_enabled") is False and (
        config.shanten_weight != 0.0
        or config.shanten_value_loss_weight != 0.0
    ):
        raise ValueError(
            "score-only rollouts require shanten_weight=0 and "
            "shanten_value_loss_weight=0"
        )
    actor_checkpoint = Path(actor_checkpoint).resolve()
    critic_checkpoint = Path(critic_checkpoint).resolve()
    if manifest.get("actor_sha256") != _sha256(actor_checkpoint):
        raise ValueError("PPO rollout actor does not match the supplied checkpoint")
    if float(manifest.get("temperature", math.nan)) != config.temperature:
        raise ValueError("PPO rollout temperature does not match training")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dataset = V4PPODataset(dataset_directory)
    if config.advantage_method == "gae" and manifest.get("sample_count") != len(dataset):
        raise ValueError("GAE dataset sample count does not match complete rollout manifest")
    indices = np.arange(len(dataset), dtype=np.int64)
    actor, _ = load_v4_actor_checkpoint(
        actor_checkpoint,
        device=device,
    )
    critic, _ = load_v4_privileged_critic_checkpoint(
        critic_checkpoint,
        device=device,
    )
    if isinstance(critic, V4FourSeatCritic) and (config.shanten_weight or config.shanten_value_loss_weight):
        raise ValueError("four-seat Critic requires score-only training")
    if _public_encoder_config(actor.config) != _public_encoder_config(
        critic.config.public_config
    ):
        raise ValueError("V4 Actor and Critic public encoder configs differ")
    actor_optimizer_groups, actor_parameters = _configure_actor_train_scope(
        actor,
        config,
    )
    actor.eval()
    critic.eval()

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    audit_loader = make_v4_ppo_data_loader(
        dataset,
        indices,
        shuffle=False,
        seed=config.seed,
        **loader_options,
    )
    updates_per_epoch = len(audit_loader)
    total_optimizer_steps = config.ppo_epochs * updates_per_epoch
    started = time.perf_counter()
    audit = _prepare_advantages(
        actor,
        critic,
        audit_loader,
        len(dataset),
        device,
        config,
    )
    if audit["old_log_probability_max_error"] > (
        config.old_log_probability_tolerance
    ):
        raise ValueError(
            "stored PPO behavior probabilities do not match the source Actor: "
            f"max error {audit['old_log_probability_max_error']:.6f}"
        )
    trainable = audit["actor_trainable"]
    if int(trainable.sum()) == 0:
        raise ValueError("PPO dataset contains no trainable Actor decisions")
    raw_advantage = audit["raw_advantage"]
    advantage_mean = float(raw_advantage[trainable].mean())
    advantage_std = float(raw_advantage[trainable].std(unbiased=False))
    normalized_advantage = torch.zeros_like(raw_advantage)
    normalized_advantage[trainable] = (
        raw_advantage[trainable] - advantage_mean
    ) / max(advantage_std, 1.0e-8)
    normalized_advantage, chicken_diagnostics = _add_chicken_action_bonus(
        normalized_advantage, audit["chicken_attribution_points"], trainable,
        config.chicken_attribution_weight,
    )
    kl_loader = None
    if config.kl_check_interval:
        probe_indices = np.flatnonzero(trainable.numpy())
        rng = np.random.default_rng(config.seed)
        probe_indices = np.sort(rng.choice(
            probe_indices,
            size=min(config.kl_check_samples, len(probe_indices)),
            replace=False,
        ))
        # Cache only the bounded CPU probe, not the full dataset or GPU batches.
        kl_loader = list(torch.utils.data.DataLoader(
            dataset, sampler=probe_indices.tolist(),
            batch_size=config.batch_size, num_workers=0,
            collate_fn=collate_v4_ppo_samples,
            pin_memory=device.type == "cuda",
        ))

    initial_actor = {
        name: value.detach().cpu().clone()
        for name, value in actor.state_dict().items()
    }
    actor_optimizer = torch.optim.AdamW(
        actor_optimizer_groups,
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        weight_decay=config.weight_decay,
    )
    mse = nn.MSELoss()
    history = []
    early_stopped = False
    optimizer_step = 0
    kl_checks = []

    for epoch in range(1, config.ppo_epochs + 1):
        loader = make_v4_ppo_data_loader(
            dataset,
            indices,
            shuffle=True,
            seed=config.seed + epoch,
            **loader_options,
        )
        totals = {
            "call_replay_loss": 0.0,
            "call_replay_batches": 0,
            "actor_count": 0,
            "critic_count": 0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
        }
        actor_learning_rate_start = None
        actor_group_learning_rates_start = None
        critic_learning_rate_start = None
        actor_learning_rate_end = None
        actor_group_learning_rates_end = None
        critic_learning_rate_end = None
        _set_actor_optimization_mode(actor)
        critic.train()
        for cpu_batch in loader:
            learning_rate_factor = _linear_learning_rate_factor(
                optimizer_step,
                total_optimizer_steps,
                config.final_learning_rate_fraction,
            )
            actor_group_learning_rates = (
                _set_grouped_optimizer_learning_rates(
                    actor_optimizer,
                    learning_rate_factor,
                )
            )
            actor_learning_rate = max(actor_group_learning_rates.values())
            critic_learning_rate = (
                config.critic_learning_rate * learning_rate_factor
            )
            _set_optimizer_learning_rate(
                critic_optimizer,
                critic_learning_rate,
            )
            if actor_learning_rate_start is None:
                actor_learning_rate_start = actor_learning_rate
                actor_group_learning_rates_start = dict(
                    actor_group_learning_rates
                )
                critic_learning_rate_start = critic_learning_rate
            actor_learning_rate_end = actor_learning_rate
            actor_group_learning_rates_end = dict(actor_group_learning_rates)
            critic_learning_rate_end = critic_learning_rate

            batch = cpu_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            sample_indices = batch.sample_index.cpu()
            advantages = normalized_advantage.index_select(
                0,
                sample_indices,
            ).to(device)
            old_log_probability = audit["recomputed_old_log_probability"].index_select(
                0,
                sample_indices,
            ).to(device)

            logits = actor(batch.features) / config.temperature
            log_probability, entropy = _selected_policy_statistics(
                logits,
                batch.action_id,
            )
            mask = _actor_trainable_mask(batch, config.actor_train_scope)
            if mask.any():
                objective = clipped_policy_objective(
                    log_probability[mask],
                    old_log_probability[mask],
                    advantages[mask],
                    entropy[mask],
                    clip_range=config.clip_range,
                    entropy_coefficient=config.entropy_coefficient,
                )
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss = objective.loss
                if call_auxiliary is not None:
                    call_loss = call_auxiliary.loss(
                        actor, config.call_replay_batch_size, device,
                        temperature=config.temperature, cap=config.call_replay_bonus_cap,
                    )
                    actor_loss = actor_loss + config.call_replay_weight * call_loss
                    totals["call_replay_loss"] += float(call_loss.detach())
                    totals["call_replay_batches"] += 1
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor_parameters,
                    config.gradient_clip,
                )
                actor_optimizer.step()
                count = int(mask.sum())
                totals["actor_count"] += count
                totals["actor_loss"] += float(objective.loss.detach()) * count
                totals["entropy"] += float(objective.entropy.detach()) * count
                totals["approximate_kl"] += (
                    float(objective.approximate_kl.detach()) * count
                )
                totals["clip_fraction"] += (
                    float(objective.clip_fraction.detach()) * count
                )

            public = v4_critic_public_from_actor_batch(batch.features)
            values = critic(public, batch.privileged)
            value_targets = audit["value_targets"].index_select(0, sample_indices).to(device)
            if isinstance(critic, V4FourSeatCritic):
                score_loss = mse(values.score_values_all, value_targets)
                shanten_loss = score_loss.new_zeros(())
            else:
                score_loss = mse(values.score_value, value_targets[:, 0])
                shanten_loss = mse(values.shanten_value, value_targets[:, 1])
            critic_loss = score_loss + (
                config.shanten_value_loss_weight * shanten_loss
            )
            critic_optimizer.zero_grad(set_to_none=True)
            (config.value_loss_coefficient * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                config.gradient_clip,
            )
            critic_optimizer.step()
            totals["critic_count"] += len(batch)
            totals["critic_loss"] += float(critic_loss.detach()) * len(batch)
            optimizer_step += 1
            if kl_loader is not None and optimizer_step % config.kl_check_interval == 0:
                check = _policy_drift_metrics(
                    actor, kl_loader, audit["recomputed_old_log_probability"],
                    device, config.temperature, config.clip_range,
                )
                check.update(epoch=epoch, optimizer_step=optimizer_step)
                kl_checks.append(check)
                print(json.dumps({"kl_check": check}), flush=True)
                _set_actor_optimization_mode(actor)
                if config.kl_early_stopping and check["approximate_kl"] > config.target_kl:
                    early_stopped = True
                    break

        if kl_loader is not None and (
            not kl_checks or kl_checks[-1]["optimizer_step"] != optimizer_step
        ):
            check = _policy_drift_metrics(
                actor, kl_loader, audit["recomputed_old_log_probability"],
                device, config.temperature, config.clip_range,
            )
            check.update(epoch=epoch, optimizer_step=optimizer_step)
            kl_checks.append(check)
            early_stopped = config.kl_early_stopping and check["approximate_kl"] > config.target_kl

        actor_count = totals["actor_count"]
        critic_count = totals["critic_count"]
        metrics = {
            "call_replay_loss": totals["call_replay_loss"] / max(1, totals["call_replay_batches"]),
            "call_replay_batches": totals["call_replay_batches"],
            "call_replay": call_auxiliary.metadata if call_auxiliary is not None else None,
            "epoch": epoch,
            "actor_loss": totals["actor_loss"] / actor_count,
            "critic_loss": totals["critic_loss"] / critic_count,
            "entropy": totals["entropy"] / actor_count,
            "approximate_kl": totals["approximate_kl"] / actor_count,
            "clip_fraction": totals["clip_fraction"] / actor_count,
            "actor_samples": actor_count,
            "critic_samples": critic_count,
            "actor_learning_rate_start": actor_learning_rate_start,
            "actor_learning_rate_end": actor_learning_rate_end,
            "actor_group_learning_rates_start": (
                actor_group_learning_rates_start
            ),
            "actor_group_learning_rates_end": actor_group_learning_rates_end,
            "critic_learning_rate_start": critic_learning_rate_start,
            "critic_learning_rate_end": critic_learning_rate_end,
            "epoch_complete": totals["critic_count"] == len(dataset),
            "policy_probe": kl_checks[-1] if kl_loader is not None else None,
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)
        if config.save_epoch_checkpoints:
            epoch_metadata = {
                "source_actor": str(actor_checkpoint),
                "source_critic": str(critic_checkpoint),
                "rollout_dataset": str(dataset_directory.resolve()),
                "training_config": asdict(config),
                "epoch_metrics": metrics,
            }
            save_v4_actor_checkpoint(
                output / f"actor_epoch_{epoch:02d}.pt", actor,
                metadata=epoch_metadata,
            )
            save_v4_privileged_critic_checkpoint(
                output / f"critic_epoch_{epoch:02d}.pt", critic,
                metadata=epoch_metadata,
            )
        if early_stopped or (
            config.kl_early_stopping and kl_loader is None and metrics["approximate_kl"] > config.target_kl
        ):
            early_stopped = True
            break

    evaluation_loader = make_v4_ppo_data_loader(
        dataset,
        indices,
        shuffle=False,
        seed=config.seed,
        **loader_options,
    )
    final_critic = _critic_metrics(
        critic,
        evaluation_loader,
        device,
        config.shanten_value_loss_weight,
    )
    actor_delta = math.sqrt(sum(
        float((value.detach().cpu() - initial_actor[name]).square().sum())
        for name, value in actor.state_dict().items()
    ))
    base_actor_delta = math.sqrt(sum(
        float((value.detach().cpu() - initial_actor[name]).square().sum())
        for name, value in actor.state_dict().items()
        if not name.startswith("kan_adapter.")
    ))
    kan_adapter_delta = math.sqrt(sum(
        float((value.detach().cpu() - initial_actor[name]).square().sum())
        for name, value in actor.state_dict().items()
        if name.startswith("kan_adapter.")
    ))
    named_parameter_groups = _named_actor_parameter_groups(actor)
    actor_group_deltas = {
        group_name: math.sqrt(sum(
            float(
                (parameter.detach().cpu() - initial_actor[name])
                .square()
                .sum()
            )
            for name, parameter in parameters
        ))
        for group_name, parameters in named_parameter_groups.items()
    }

    actor_path = output / "actor_after.pt"
    critic_path = output / "critic_after.pt"
    metadata = {
        "v4_ppo_training_version": V4_PPO_TRAINING_VERSION,
        "source_actor": str(actor_checkpoint),
        "source_critic": str(critic_checkpoint),
        "rollout_dataset": str(dataset_directory.resolve()),
        "rollout_rounds": int(manifest["round_count"]),
        "gamma": 1.0,
        "gae_lambda": config.gae_lambda if config.advantage_method == "gae" else 1.0,
        "advantage_method": config.advantage_method,
        "training_config": asdict(config),
    }
    save_v4_actor_checkpoint(actor_path, actor, metadata=metadata)
    save_v4_privileged_critic_checkpoint(
        critic_path,
        critic,
        metadata=metadata,
    )
    torch.save({
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
    }, output / "optimizer_state.pt")

    result: dict[str, object] = {
        "v4_ppo_training_version": V4_PPO_TRAINING_VERSION,
        "device": str(device),
        "actor_checkpoint": str(actor_path.resolve()),
        "critic_checkpoint": str(critic_path.resolve()),
        "source_actor": str(actor_checkpoint),
        "source_critic": str(critic_checkpoint),
        "dataset": str(dataset_directory.resolve()),
        "sample_count": len(dataset),
        "episode_count": int(len(np.unique(dataset.episode_seeds))),
        "actor_trainable_count": int(trainable.sum()),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "chicken_attribution": chicken_diagnostics,
        "call_replay": call_auxiliary.metadata if call_auxiliary is not None else None,
        "old_log_probability_max_error": audit[
            "old_log_probability_max_error"
        ],
        "old_log_probability_mean_error": audit[
            "old_log_probability_mean_error"
        ],
        "initial_critic": audit["critic_metrics"],
        "final_critic": final_critic,
        "actor_parameter_delta_l2": actor_delta,
        "base_actor_parameter_delta_l2": base_actor_delta,
        "kan_adapter_parameter_delta_l2": kan_adapter_delta,
        "actor_group_parameter_delta_l2": actor_group_deltas,
        "early_stopped_for_kl": early_stopped,
        "history": history,
        "training_config": asdict(config),
        "gamma": 1.0,
        "gae_lambda": config.gae_lambda if config.advantage_method == "gae" else 1.0,
        "advantage_method": config.advantage_method,
        "value_target": "lambda_return" if config.advantage_method == "gae" else "monte_carlo",
        "elapsed_seconds": time.perf_counter() - started,
        "planned_optimizer_steps": total_optimizer_steps,
        "completed_optimizer_steps": optimizer_step,
        "kl_checks": kl_checks,
        "source_actor_sha256": _sha256(actor_checkpoint),
        "source_critic_sha256": _sha256(critic_checkpoint),
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    return result


def _policy_drift_metrics(actor, loader, old_log_probabilities, device,
                          temperature, clip_range):
    """Measure one current policy on a fixed, uniformly sampled decision probe."""
    count, kl_sum, clipped = 0, 0.0, 0
    actor.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
            current, _ = _selected_policy_statistics(
                actor(batch.features) / temperature, batch.action_id,
            )
            old = old_log_probabilities.index_select(
                0, batch.sample_index.cpu(),
            ).to(device)
            log_ratio = current - old
            ratio = log_ratio.exp()
            kl = (ratio - 1) - log_ratio
            if not torch.isfinite(kl).all():
                raise ValueError("nonfinite policy drift")
            count += len(batch)
            kl_sum += float(kl.sum())
            clipped += int(((ratio - 1).abs() > clip_range).sum())
    if not count:
        raise ValueError("empty policy drift probe")
    return {"sample_count": count, "approximate_kl": kl_sum / count,
            "clip_fraction": clipped / count}


def _prepare_advantages(
    actor,
    critic,
    loader,
    sample_count: int,
    device: torch.device,
    config: V4PPOTrainingConfig,
) -> dict[str, object]:
    raw_advantage = torch.empty(sample_count, dtype=torch.float32)
    four_seat = isinstance(critic, V4FourSeatCritic)
    mc_targets = torch.empty((sample_count, 4 if four_seat else 2), dtype=torch.float32)
    old_values = torch.empty_like(mc_targets)
    episode_ids = torch.empty(sample_count, dtype=torch.int64)
    player_ids = torch.empty_like(episode_ids)
    decision_indices = torch.empty_like(episode_ids)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    recomputed_old = torch.empty(sample_count, dtype=torch.float32)
    actor_trainable = torch.zeros(sample_count, dtype=torch.bool)
    chicken_points = torch.full((sample_count,), float("nan"), dtype=torch.float32)
    probability_error_sum = 0.0
    probability_error_max = 0.0
    metrics = _empty_critic_totals()
    actor.eval()
    critic.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = actor(batch.features) / config.temperature
            log_probability, _ = _selected_policy_statistics(
                logits,
                batch.action_id,
            )
            public = v4_critic_public_from_actor_batch(batch.features)
            values = critic(public, batch.privileged)
            returns = batch.score_return + (
                config.shanten_weight * batch.shanten_return
            )
            predicted = values.score_value + (
                config.shanten_weight * values.shanten_value
            )
            advantage = returns - predicted
            indices = batch.sample_index.cpu()
            if len(indices.unique()) != len(indices) or seen[indices].any():
                raise ValueError("duplicate sample in advantage audit")
            seen[indices] = True
            if four_seat:
                labels = batch.score_returns_all
                if (not torch.isfinite(labels).all() or labels.shape != (len(batch), 4)
                        or not torch.allclose(labels[:, 0], batch.score_return, atol=1e-6)
                        or labels.sum(dim=1).abs().max() > 1e-5):
                    raise ValueError("invalid four-seat relative score labels")
                mc_targets[indices] = labels.cpu()
                old_values[indices] = values.score_values_all.cpu()
            else:
                mc_targets[indices] = torch.stack((batch.score_return, batch.shanten_return), dim=1).cpu()
                old_values[indices] = torch.stack((values.score_value, values.shanten_value), dim=1).cpu()
            episode_ids[indices] = batch.episode_seed.cpu()
            player_ids[indices] = batch.player_id.cpu()
            decision_indices[indices] = batch.decision_index.cpu()
            raw_advantage[indices] = advantage.cpu()
            chicken_points[indices] = batch.chicken_attribution_points.cpu()
            recomputed_old[indices] = log_probability.cpu()
            actor_trainable[indices] = _actor_trainable_mask(
                batch,
                config.actor_train_scope,
            ).cpu()
            errors = (
                log_probability - batch.old_log_probability
            ).abs().cpu()
            probability_error_sum += float(errors.sum())
            probability_error_max = max(
                probability_error_max,
                float(errors.max()),
            )
            _accumulate_critic_totals(metrics, values, batch)
    if not seen.all():
        raise ValueError("missing samples in advantage audit")
    value_targets = mc_targets
    if config.advantage_method == "gae":
        head_advantages, value_targets = complete_episode_gae(
            mc_targets, old_values, episode_ids, player_ids, decision_indices,
            gae_lambda=config.gae_lambda,
        )
        raw_advantage = head_advantages[:, 0] if four_seat else head_advantages[:, 0] + config.shanten_weight * head_advantages[:, 1]
    return {
        "raw_advantage": raw_advantage,
        "chicken_attribution_points": chicken_points,
        "value_targets": value_targets,
        "recomputed_old_log_probability": recomputed_old,
        "actor_trainable": actor_trainable,
        "old_log_probability_mean_error": probability_error_sum / sample_count,
        "old_log_probability_max_error": probability_error_max,
        "critic_metrics": _finish_critic_totals(
            metrics,
            shanten_weight=config.shanten_value_loss_weight,
        ),
    }


def _add_chicken_action_bonus(normalized_advantage, points, trainable, weight):
    """Add a local, non-propagating signal AFTER base advantage normalization.

    Critic targets never enter this function. Do not recenter the bonus: doing so
    would also change unrelated decisions and make its weight batch-dependent.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("chicken attribution weight must be finite and nonnegative")
    if points.shape != normalized_advantage.shape or trainable.shape != points.shape:
        raise ValueError("chicken bonus arrays must have matching shapes")
    known = torch.isfinite(points)
    if weight and not known.all():
        raise ValueError("chicken bonus requires finite attribution for every sample; legacy data is not zero")
    bonus = torch.zeros_like(normalized_advantage)
    if weight:
        bonus[trainable] = weight * points[trainable] / 30.0
    updated = normalized_advantage + bonus if weight else normalized_advantage
    if not torch.isfinite(updated).all():
        raise ValueError("nonfinite Actor advantage after chicken bonus")
    selected = bonus[trainable]
    known_trainable = known & trainable
    return updated, {
        "weight": weight,
        "application": "after_base_normalization_no_propagation_no_renormalization",
        "score_divisor": 30.0,
        "missing_sample_count": int((~known).sum()),
        "attributed_trainable_count": int(((points != 0) & known_trainable).sum()),
        "positive_trainable_count": int(((points > 0) & known_trainable).sum()),
        "negative_trainable_count": int(((points < 0) & known_trainable).sum()),
        "ignored_forced_nonzero_count": int(((points != 0) & known & ~trainable).sum()),
        "bonus_mean": float(selected.mean()) if selected.numel() else 0.0,
        "bonus_std": float(selected.std(unbiased=False)) if selected.numel() else 0.0,
        "bonus_max_abs": float(selected.abs().max()) if selected.numel() else 0.0,
        "advantage_sign_changed_count": int(((updated.sign() != normalized_advantage.sign()) & trainable).sum()),
    }


def _selected_policy_statistics(
    logits: torch.Tensor,
    action_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probability = torch.log_softmax(logits, dim=1)
    selected = log_probability.gather(1, action_id[:, None]).squeeze(1)
    selectable = torch.isfinite(logits)
    probabilities = torch.softmax(logits, dim=1)
    safe_log_probability = log_probability.masked_fill(~selectable, 0.0)
    entropy = -(probabilities * safe_log_probability).sum(dim=1)
    return selected, entropy


def _set_actor_optimization_mode(actor: nn.Module) -> None:
    """Enable recurrent backpropagation while keeping policy dropout off."""

    actor.train()
    for module in actor.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def _public_encoder_config(config) -> dict[str, object]:
    return {
        name: value
        for name, value in asdict(config).items()
        if not name.startswith("kan_adapter_")
    }


def _configure_actor_train_scope(
    actor,
    config: V4PPOTrainingConfig,
) -> tuple[list[dict[str, object]], tuple[nn.Parameter, ...]]:
    scope = config.actor_train_scope
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    if scope == "all":
        selected = {
            "all": (
                config.actor_learning_rate,
                tuple(actor.named_parameters()),
            ),
        }
    elif scope == "kan_adapter":
        if actor.kan_adapter is None:
            raise ValueError(
                "kan-adapter actor train scope requires an enabled adapter"
            )
        selected = {
            "kan_adapter": (
                config.actor_learning_rate,
                tuple(
                    (name, parameter)
                    for name, parameter in actor.named_parameters()
                    if name.startswith("kan_adapter.")
                ),
            ),
        }
    elif scope == "joint_grouped":
        if actor.kan_adapter is None:
            raise ValueError(
                "joint-grouped actor train scope requires an enabled adapter"
            )
        named_groups = _named_actor_parameter_groups(actor)
        selected = {
            "encoder": (
                config.actor_encoder_learning_rate,
                named_groups["encoder"],
            ),
            "scorer": (
                config.actor_scorer_learning_rate,
                named_groups["scorer"],
            ),
            "kan_adapter": (
                config.kan_adapter_learning_rate,
                named_groups["kan_adapter"],
            ),
        }
    else:
        raise ValueError(f"unsupported actor train scope: {scope}")

    optimizer_groups: list[dict[str, object]] = []
    parameters: list[nn.Parameter] = []
    for group_name, (learning_rate, named_parameters) in selected.items():
        group_parameters = tuple(
            parameter for _, parameter in named_parameters
        )
        if not group_parameters:
            raise ValueError(
                f"actor train scope selected empty parameter group: {group_name}"
            )
        for parameter in group_parameters:
            parameter.requires_grad_(True)
        parameters.extend(group_parameters)
        optimizer_groups.append({
            "params": group_parameters,
            "lr": learning_rate,
            "base_lr": learning_rate,
            "group_name": group_name,
        })
    if not parameters:
        raise ValueError("actor train scope selected no parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("actor parameter groups overlap")
    return optimizer_groups, tuple(parameters)


_ACTOR_SCORER_PREFIXES = (
    "action_type_embedding.",
    "action_scalar_encoder.",
    "action_encoder.",
    "state_for_action.",
    "pon_follow_up_pool.",
    "family_scorers.",
)


def _named_actor_parameter_groups(
    actor,
) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "encoder": [],
        "scorer": [],
        "kan_adapter": [],
    }
    for name, parameter in actor.named_parameters():
        if name.startswith("kan_adapter."):
            group_name = "kan_adapter"
        elif name.startswith(_ACTOR_SCORER_PREFIXES):
            group_name = "scorer"
        else:
            group_name = "encoder"
        groups[group_name].append((name, parameter))
    return {
        group_name: tuple(named_parameters)
        for group_name, named_parameters in groups.items()
    }


def _actor_trainable_mask(batch, scope: str) -> torch.Tensor:
    mask = batch.actor_trainable
    if scope in {"all", "joint_grouped"}:
        return mask
    if scope != "kan_adapter":
        raise ValueError(f"unsupported actor train scope: {scope}")
    features = batch.features.kan_adapter_features
    if features is None:
        raise ValueError("kan-adapter PPO requires adapter features")
    known_kan = features[..., 0] > 0.5
    kan_opportunity = (
        known_kan
        & batch.features.action_legal
    ).any(dim=1)
    return mask & kan_opportunity


def _linear_learning_rate_factor(
    step: int,
    total_steps: int,
    final_fraction: float,
) -> float:
    if total_steps <= 1:
        return 1.0
    progress = step / (total_steps - 1)
    return 1.0 + (final_fraction - 1.0) * progress


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _set_grouped_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    factor: float,
) -> dict[str, float]:
    learning_rates = {}
    for index, parameter_group in enumerate(optimizer.param_groups):
        group_name = str(parameter_group.get("group_name", index))
        learning_rate = float(parameter_group["base_lr"]) * factor
        parameter_group["lr"] = learning_rate
        learning_rates[group_name] = learning_rate
    return learning_rates


def _critic_metrics(
    critic,
    loader,
    device: torch.device,
    shanten_weight: float,
) -> dict[str, float]:
    totals = _empty_critic_totals()
    critic.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            values = critic(
                v4_critic_public_from_actor_batch(batch.features),
                batch.privileged,
            )
            _accumulate_critic_totals(totals, values, batch)
    return _finish_critic_totals(totals, shanten_weight=shanten_weight)


def _empty_critic_totals() -> dict[str, float]:
    return {
        "count": 0.0,
        "score_squared_error": 0.0,
        "score_absolute_error": 0.0,
        "shanten_squared_error": 0.0,
        "shanten_absolute_error": 0.0,
    }


def _accumulate_critic_totals(totals, values, batch) -> None:
    score_error = values.score_value - batch.score_return
    shanten_error = values.shanten_value - batch.shanten_return
    totals["count"] += len(batch)
    totals["score_squared_error"] += float(score_error.square().sum())
    totals["score_absolute_error"] += float(score_error.abs().sum())
    totals["shanten_squared_error"] += float(shanten_error.square().sum())
    totals["shanten_absolute_error"] += float(shanten_error.abs().sum())


def _finish_critic_totals(
    totals,
    *,
    shanten_weight: float = 0.25,
) -> dict[str, float]:
    count = totals["count"]
    score_mse = totals["score_squared_error"] / count
    shanten_mse = totals["shanten_squared_error"] / count
    return {
        "loss": score_mse + shanten_weight * shanten_mse,
        "score_mse": score_mse,
        "score_rmse_native": math.sqrt(score_mse) * 30.0,
        "score_mae_native": totals["score_absolute_error"] / count * 30.0,
        "shanten_mse": shanten_mse,
        "shanten_rmse_native": math.sqrt(shanten_mse) * 10.0,
        "shanten_mae_native": (
            totals["shanten_absolute_error"] / count * 10.0
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one on-policy V4 PPO update with MC or per-player GAE."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--advantage-method", choices=("mc", "gae"), default="mc")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--chicken-attribution-weight", type=float, default=0.0)
    parser.add_argument("--call-replay-directory")
    parser.add_argument("--call-replay-weight", type=float, default=0.0)
    parser.add_argument("--call-replay-batch-size", type=int, default=16)
    parser.add_argument("--call-replay-bonus-cap", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--actor-encoder-learning-rate",
        type=float,
        default=3e-6,
    )
    parser.add_argument(
        "--actor-scorer-learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--kan-adapter-learning-rate",
        type=float,
        default=3e-5,
    )
    parser.add_argument("--critic-learning-rate", type=float, default=3e-5)
    parser.add_argument("--final-lr-fraction", type=float, default=0.1)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--disable-kl-early-stopping", action="store_true")
    parser.add_argument("--shanten-weight", type=float, default=0.25)
    parser.add_argument("--shanten-value-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--old-log-probability-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--kl-check-interval", type=int, default=0)
    parser.add_argument("--kl-check-samples", type=int, default=4096)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument(
        "--actor-train-scope",
        choices=("all", "kan_adapter", "joint_grouped"),
        default="all",
    )
    args = parser.parse_args()
    result = train_v4_ppo_iteration(
        args.dataset,
        args.actor,
        args.critic,
        args.output,
        config=V4PPOTrainingConfig(
            advantage_method=args.advantage_method,
            gae_lambda=args.gae_lambda,
            chicken_attribution_weight=args.chicken_attribution_weight,
            call_replay_directory=args.call_replay_directory,
            call_replay_weight=args.call_replay_weight,
            call_replay_batch_size=args.call_replay_batch_size,
            call_replay_bonus_cap=args.call_replay_bonus_cap,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            actor_learning_rate=args.actor_learning_rate,
            actor_encoder_learning_rate=args.actor_encoder_learning_rate,
            actor_scorer_learning_rate=args.actor_scorer_learning_rate,
            kan_adapter_learning_rate=args.kan_adapter_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            final_learning_rate_fraction=args.final_lr_fraction,
            entropy_coefficient=args.entropy_coefficient,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            kl_early_stopping=not args.disable_kl_early_stopping,
            shanten_weight=args.shanten_weight,
            shanten_value_loss_weight=args.shanten_value_loss_weight,
            old_log_probability_tolerance=(
                args.old_log_probability_tolerance
            ),
            seed=args.seed,
            num_workers=args.num_workers,
            actor_train_scope=args.actor_train_scope,
            kl_check_interval=args.kl_check_interval,
            kl_check_samples=args.kl_check_samples,
            save_epoch_checkpoints=args.save_epoch_checkpoints,
        ),
        device=resolve_training_device(args.device),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
