from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from .training_device import DEVICE_CHOICES, resolve_training_device
from .v4_agent import V4Agent
from .v4_critic_dataset import (
    V4_CRITIC_DATASET_FORMAT_VERSION,
    V4_CRITIC_DATASET_PREFIX,
    V4CriticWriter,
)
from .v4_critic_rollout import run_v4_critic_episode


def collect_v4_critic_rollouts(
    output: Path | str,
    actor_checkpoint: Path | str,
    *,
    round_count: int,
    seed_start: int,
    device: torch.device | str = "cpu",
    greedy: bool = False,
    temperature: float = 1.0,
    policy_seed: int = 20260930,
    samples_per_shard: int = 256,
    compressed: bool = True,
    progress_interval: int = 100,
) -> dict[str, object]:
    if round_count <= 0:
        raise ValueError("round count must be positive")
    if progress_interval < 0:
        raise ValueError("progress interval cannot be negative")
    destination = Path(output)
    if tuple(destination.glob(f"{V4_CRITIC_DATASET_PREFIX}_*.npz")):
        raise FileExistsError(
            f"V4 Critic shards already exist in {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    checkpoint_path = Path(actor_checkpoint).resolve()
    agent = V4Agent.from_checkpoint(
        checkpoint_path,
        device=device,
        greedy=greedy,
        temperature=temperature,
        seed=policy_seed,
    )

    end_reasons: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    score_totals = [0, 0, 0, 0]
    sample_count = 0
    hu_round_count = 0
    started = time.perf_counter()
    with V4CriticWriter(
        destination,
        samples_per_shard=samples_per_shard,
        compressed=compressed,
    ) as writer:
        for round_index in range(round_count):
            episode = run_v4_critic_episode(
                agent,
                seed_start + round_index,
            )
            writer.append_many(episode.samples)
            sample_count += len(episode.samples)
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
        "v4_critic_dataset_format_version": (
            V4_CRITIC_DATASET_FORMAT_VERSION
        ),
        "actor_checkpoint": str(checkpoint_path),
        "actor_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "greedy": greedy,
        "temperature": temperature,
        "policy_seed": policy_seed,
        "round_count": round_count,
        "seed_start": seed_start,
        "seed_end": seed_start + round_count - 1,
        "sample_count": sample_count,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect structured V4 privileged Critic rollouts."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-seed", type=int, default=20260930)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    torch.set_num_threads(args.torch_threads)
    report = collect_v4_critic_rollouts(
        args.output,
        args.actor,
        round_count=args.rounds,
        seed_start=args.seed_start,
        device=resolve_training_device(args.device),
        greedy=args.greedy,
        temperature=args.temperature,
        policy_seed=args.policy_seed,
        samples_per_shard=args.samples_per_shard,
        compressed=not args.no_compression,
        progress_interval=args.progress_interval,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
