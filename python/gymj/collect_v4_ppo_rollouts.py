from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import time

import torch

from .training_device import DEVICE_CHOICES, resolve_training_device
from .v4_agent import V4Agent
from .v4_ppo_dataset import (
    V4_PPO_DATASET_FORMAT_VERSION,
    V4_PPO_DATASET_PREFIX,
    V4PPOWriter,
)
from .v4_ppo_rollout import run_v4_ppo_episode


def collect_v4_ppo_rollouts(
    output: Path | str,
    actor_checkpoint: Path | str,
    *,
    round_count: int,
    seed_start: int,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    policy_seed: int = 20260930,
    samples_per_shard: int = 128,
    compressed: bool = True,
    progress_interval: int = 20,
    shanten_reward_enabled: bool = True,
    actor_float32: bool = False,
) -> dict[str, object]:
    if round_count <= 0:
        raise ValueError("round count must be positive")
    if progress_interval < 0:
        raise ValueError("progress interval cannot be negative")
    destination = Path(output)
    if tuple(destination.glob(f"{V4_PPO_DATASET_PREFIX}_*.npz")):
        raise FileExistsError(f"V4 PPO shards already exist in {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    checkpoint_path = Path(actor_checkpoint).resolve()
    agent = V4Agent.from_checkpoint(
        checkpoint_path,
        device=device,
        greedy=False,
        temperature=temperature,
        seed=policy_seed,
    )

    end_reasons: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    score_totals = [0, 0, 0, 0]
    sample_count = 0
    trainable_count = 0
    hu_round_count = 0
    started = time.perf_counter()
    with gzip.open(destination / "episodes.jsonl.gz", "xt", encoding="ascii") as episodes, V4PPOWriter(
        destination,
        samples_per_shard=samples_per_shard,
        compressed=compressed,
        actor_float32=actor_float32,
    ) as writer:
        for round_index in range(round_count):
            episode = run_v4_ppo_episode(
                agent,
                seed_start + round_index,
                shanten_reward_enabled=shanten_reward_enabled,
            )
            writer.append_many(episode.samples)
            episodes.write(json.dumps({
                "episode_seed": episode.seed,
                "dealer": 0,
                "step_count": episode.step_count,
                "score_deltas": episode.score_deltas,
                "hu_players": episode.hu_players,
                "end_reason": str(episode.end_reason),
                "chicken_attribution": episode.chicken_attribution_audit,
            }) + "\n")
            sample_count += len(episode.samples)
            trainable_count += sum(
                sample.actor_trainable for sample in episode.samples
            )
            end_reasons[str(episode.end_reason)] += 1
            hu_round_count += int(bool(episode.hu_players))
            for seat, score in enumerate(episode.score_deltas):
                score_totals[seat] += score
            action_counts.update(sample.action_id for sample in episode.samples)
            completed = round_index + 1
            if progress_interval and completed % progress_interval == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rounds={completed}/{round_count} "
                    f"samples={sample_count} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "chicken_attribution_version": 1,
        "score_divisor": 30.0,
        "v4_ppo_dataset_format_version": V4_PPO_DATASET_FORMAT_VERSION,
        "trajectory_contract": "complete_candidate_player_decisions_v1",
        "actor_checkpoint": str(checkpoint_path),
        "actor_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "greedy": False,
        "shanten_reward_enabled": shanten_reward_enabled,
        "reward_mode": "score_and_shanten" if shanten_reward_enabled else "score_only",
        "actor_float32": actor_float32,
        "temperature": temperature,
        "policy_seed": policy_seed,
        "round_count": round_count,
        "seed_start": seed_start,
        "seed_end": seed_start + round_count - 1,
        "sample_count": sample_count,
        "actor_trainable_count": trainable_count,
        "actor_trainable_fraction": trainable_count / sample_count,
        "samples_per_round": sample_count / round_count,
        "samples_per_shard": samples_per_shard,
        "compressed": compressed,
        "hu_round_count": hu_round_count,
        "end_reason_counts": dict(sorted(end_reasons.items())),
        "score_totals": score_totals,
        "action_counts": {
            str(action_id): count
            for action_id, count in sorted(action_counts.items())
        },
        "elapsed_seconds": elapsed,
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    return report


def _collect_v4_ppo_worker(arguments: tuple[object, ...]) -> dict[str, object]:
    (
        output,
        actor_checkpoint,
        round_count,
        seed_start,
        temperature,
        policy_seed,
        samples_per_shard,
        compressed,
        progress_interval,
        torch_threads,
        shanten_reward_enabled,
        actor_float32,
    ) = arguments
    torch.set_num_threads(int(torch_threads))
    return collect_v4_ppo_rollouts(
        Path(str(output)),
        Path(str(actor_checkpoint)),
        round_count=int(round_count),
        seed_start=int(seed_start),
        device="cpu",
        temperature=float(temperature),
        policy_seed=int(policy_seed),
        samples_per_shard=int(samples_per_shard),
        compressed=bool(compressed),
        progress_interval=int(progress_interval),
        shanten_reward_enabled=bool(shanten_reward_enabled),
        actor_float32=bool(actor_float32),
    )


def collect_v4_ppo_rollouts_parallel(
    output: Path | str,
    actor_checkpoint: Path | str,
    *,
    round_count: int,
    seed_start: int,
    worker_count: int,
    temperature: float = 1.0,
    policy_seed: int = 20260930,
    samples_per_shard: int = 128,
    compressed: bool = True,
    progress_interval: int = 20,
    torch_threads: int = 1,
    shanten_reward_enabled: bool = True,
    actor_float32: bool = False,
) -> dict[str, object]:
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    if torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    if worker_count == 1:
        torch.set_num_threads(torch_threads)
        return collect_v4_ppo_rollouts(
            output,
            actor_checkpoint,
            round_count=round_count,
            seed_start=seed_start,
            device="cpu",
            temperature=temperature,
            policy_seed=policy_seed,
            samples_per_shard=samples_per_shard,
            compressed=compressed,
            progress_interval=progress_interval,
            shanten_reward_enabled=shanten_reward_enabled,
            actor_float32=actor_float32,
        )
    if round_count < worker_count:
        raise ValueError("round count must be at least the worker count")

    destination = Path(output)
    if (
        tuple(destination.glob(f"{V4_PPO_DATASET_PREFIX}_*.npz"))
        or (destination / "manifest.json").exists()
    ):
        raise FileExistsError(f"V4 PPO output already exists in {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    worker_root = destination / "_workers"
    if worker_root.exists():
        raise FileExistsError(f"worker directory already exists: {worker_root}")
    worker_root.mkdir()

    checkpoint_path = Path(actor_checkpoint).resolve()
    base_rounds, extra_rounds = divmod(round_count, worker_count)
    next_seed = seed_start
    worker_arguments = []
    for worker_index in range(worker_count):
        worker_rounds = base_rounds + (1 if worker_index < extra_rounds else 0)
        worker_arguments.append((
            str(worker_root / f"worker_{worker_index:03d}"),
            str(checkpoint_path),
            worker_rounds,
            next_seed,
            temperature,
            policy_seed,
            samples_per_shard,
            compressed,
            progress_interval,
            torch_threads,
            shanten_reward_enabled,
            actor_float32,
        ))
        next_seed += worker_rounds

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        worker_reports = list(executor.map(
            _collect_v4_ppo_worker,
            worker_arguments,
        ))

    shard_index = 0
    episode_directory = destination / "episodes"
    episode_directory.mkdir()
    for worker_index in range(worker_count):
        worker_directory = worker_root / f"worker_{worker_index:03d}"
        for source in sorted(worker_directory.glob(
            f"{V4_PPO_DATASET_PREFIX}_*.npz"
        )):
            target = destination / (
                f"{V4_PPO_DATASET_PREFIX}_{shard_index:06d}.npz"
            )
            source.replace(target)
            shard_index += 1
        (worker_directory / "episodes.jsonl.gz").replace(
            episode_directory / f"worker_{worker_index:03d}.jsonl.gz"
        )
    shutil.rmtree(worker_root)

    end_reasons: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    score_totals = [0, 0, 0, 0]
    for report in worker_reports:
        end_reasons.update({
            str(name): int(count)
            for name, count in report["end_reason_counts"].items()
        })
        action_counts.update({
            int(action_id): int(count)
            for action_id, count in report["action_counts"].items()
        })
        for seat, value in enumerate(report["score_totals"]):
            score_totals[seat] += int(value)

    sample_count = sum(int(report["sample_count"]) for report in worker_reports)
    trainable_count = sum(
        int(report["actor_trainable_count"])
        for report in worker_reports
    )
    elapsed = time.perf_counter() - started
    merged: dict[str, object] = {
        "chicken_attribution_version": 1,
        "score_divisor": 30.0,
        "v4_ppo_dataset_format_version": V4_PPO_DATASET_FORMAT_VERSION,
        "trajectory_contract": "complete_candidate_player_decisions_v1",
        "actor_checkpoint": str(checkpoint_path),
        "actor_sha256": _sha256(checkpoint_path),
        "device": "cpu",
        "greedy": False,
        "shanten_reward_enabled": shanten_reward_enabled,
        "reward_mode": "score_and_shanten" if shanten_reward_enabled else "score_only",
        "actor_float32": actor_float32,
        "temperature": temperature,
        "policy_seed": policy_seed,
        "round_count": round_count,
        "seed_start": seed_start,
        "seed_end": seed_start + round_count - 1,
        "worker_count": worker_count,
        "torch_threads_per_worker": torch_threads,
        "sample_count": sample_count,
        "actor_trainable_count": trainable_count,
        "actor_trainable_fraction": trainable_count / sample_count,
        "samples_per_round": sample_count / round_count,
        "samples_per_shard": samples_per_shard,
        "compressed": compressed,
        "shard_count": shard_index,
        "hu_round_count": sum(
            int(report["hu_round_count"]) for report in worker_reports
        ),
        "end_reason_counts": dict(sorted(end_reasons.items())),
        "score_totals": score_totals,
        "action_counts": {
            str(action_id): count
            for action_id, count in sorted(action_counts.items())
        },
        "elapsed_seconds": elapsed,
    }
    (destination / "manifest.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect on-policy V4 PPO rollouts."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-seed", type=int, default=20260930)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=14)
    parser.add_argument("--no-shanten-reward", action="store_true")
    parser.add_argument("--actor-float32", action="store_true")
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.workers > 1:
        if args.device != "cpu":
            parser.error("parallel rollout collection currently requires --device cpu")
        result = collect_v4_ppo_rollouts_parallel(
            args.output,
            args.actor,
            round_count=args.rounds,
            seed_start=args.seed_start,
            worker_count=args.workers,
            temperature=args.temperature,
            policy_seed=args.policy_seed,
            samples_per_shard=args.samples_per_shard,
            compressed=not args.no_compression,
            progress_interval=args.progress_interval,
            torch_threads=args.torch_threads,
            shanten_reward_enabled=not args.no_shanten_reward,
            actor_float32=args.actor_float32,
        )
    else:
        torch.set_num_threads(args.torch_threads)
        result = collect_v4_ppo_rollouts(
            args.output,
            args.actor,
            round_count=args.rounds,
            seed_start=args.seed_start,
            device=resolve_training_device(args.device),
            temperature=args.temperature,
            policy_seed=args.policy_seed,
            samples_per_shard=args.samples_per_shard,
            compressed=not args.no_compression,
            progress_interval=args.progress_interval,
            shanten_reward_enabled=not args.no_shanten_reward,
            actor_float32=args.actor_float32,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
