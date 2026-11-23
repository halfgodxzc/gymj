from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

from ._gymj import (
    ACTION_ENCODING_VERSION,
    DECISION_FEATURE_VERSION,
    ACTION_SPACE_SIZE,
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    ObservationEncoder,
)
from .env import EnvState
from .teacher import ShantenTeacher
from .v4_demonstration_dataset import (
    V4_CATEGORY_NAMES,
    V4_DEMONSTRATION_FORMAT_VERSION,
    V4_DEMONSTRATION_PREFIX,
    V4DemonstrationSample,
    V4DemonstrationWriter,
)
from .v4_features import V4_TENSOR_VERSION


def collect_v4_demonstrations(
    output: Path | str,
    *,
    round_count: int,
    seed_start: int,
    teacher_seed: int = 20260929,
    samples_per_shard: int = 256,
    compressed: bool = True,
    progress_interval: int = 100,
    ready_policy: str = "skip",
) -> dict[str, object]:
    if round_count <= 0:
        raise ValueError("round count must be positive")
    if samples_per_shard <= 0:
        raise ValueError("samples per shard must be positive")
    if progress_interval < 0:
        raise ValueError("progress interval cannot be negative")
    destination = Path(output)
    existing = tuple(destination.glob(f"{V4_DEMONSTRATION_PREFIX}_*.npz"))
    if existing:
        raise FileExistsError(
            f"V4 demonstration shards already exist in {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    teacher = ShantenTeacher(
        seed=teacher_seed,
        kan_probability=0.0,
        ready_policy=ready_policy,
    )
    category_counts: Counter[str] = Counter()
    end_reason_counts: Counter[str] = Counter()
    decision_count = 0
    skipped_decision_count = 0
    sample_count = 0
    started = time.perf_counter()

    with V4DemonstrationWriter(
        destination,
        samples_per_shard=samples_per_shard,
        compressed=compressed,
    ) as writer:
        for round_index in range(round_count):
            episode_seed = seed_start + round_index
            arena = Arena(seed=episode_seed)
            arena.start_round()
            round_decision_index = 0

            while not arena.finished:
                current = arena.current_decision()
                encoded = ActionEncoder.encode_legal_actions(
                    current.legal_actions
                )
                legal_mask = tuple(
                    int(encoded.is_legal(action_id))
                    for action_id in range(ACTION_SPACE_SIZE)
                )
                observation = ObservationEncoder.encode(
                    arena,
                    current.player,
                )
                teacher_decision = teacher.decide(EnvState(
                    player_id=current.player,
                    observation=observation,
                    legal_mask=legal_mask,
                ))

                if teacher_decision.trainable:
                    features = DecisionFeatureEncoder.encode(
                        arena,
                        current.player,
                    )
                    sample = V4DemonstrationSample.from_teacher_decision(
                        episode_seed=episode_seed,
                        player_id=current.player,
                        decision_index=round_decision_index,
                        features=features,
                        decision=teacher_decision,
                    )
                    writer.append(sample)
                    category_counts[V4_CATEGORY_NAMES[sample.category]] += 1
                    sample_count += 1
                else:
                    skipped_decision_count += 1

                selected = teacher_decision.selected_action_id
                if not encoded.is_legal(selected):
                    raise RuntimeError("teacher selected an illegal action")
                arena.submit_action(encoded.action_at(selected))
                decision_count += 1
                round_decision_index += 1

            end_reason_counts[str(arena.result.reason)] += 1
            completed = round_index + 1
            if progress_interval and completed % progress_interval == 0:
                print(
                    f"rounds={completed}/{round_count} "
                    f"samples={sample_count} decisions={decision_count}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "v4_demonstration_format_version": V4_DEMONSTRATION_FORMAT_VERSION,
        "v4_tensor_version": V4_TENSOR_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "round_count": round_count,
        "seed_start": seed_start,
        "seed_end": seed_start + round_count - 1,
        "teacher_seed": teacher_seed,
        "teacher_kan_probability": 0.0,
        "teacher_ready_policy": ready_policy,
        "samples_per_shard": samples_per_shard,
        "compressed": compressed,
        "decision_count": decision_count,
        "skipped_decision_count": skipped_decision_count,
        "sample_count": sample_count,
        "category_counts": {
            name: int(category_counts[name])
            for name in V4_CATEGORY_NAMES
        },
        "end_reason_counts": dict(sorted(end_reason_counts.items())),
        "elapsed_seconds": elapsed,
    }
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    return report


def _collect_worker(arguments: tuple[object, ...]) -> dict[str, object]:
    (
        output,
        round_count,
        seed_start,
        teacher_seed,
        samples_per_shard,
        compressed,
        ready_policy,
    ) = arguments
    return collect_v4_demonstrations(
        Path(output),
        round_count=int(round_count),
        seed_start=int(seed_start),
        teacher_seed=int(teacher_seed),
        samples_per_shard=int(samples_per_shard),
        compressed=bool(compressed),
        progress_interval=0,
        ready_policy=str(ready_policy),
    )


def collect_v4_demonstrations_parallel(
    output: Path | str,
    *,
    round_count: int,
    seed_start: int,
    worker_count: int,
    teacher_seed: int = 20260929,
    samples_per_shard: int = 256,
    compressed: bool = True,
    ready_policy: str = "skip",
) -> dict[str, object]:
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    if worker_count == 1:
        return collect_v4_demonstrations(
            output,
            round_count=round_count,
            seed_start=seed_start,
            teacher_seed=teacher_seed,
            samples_per_shard=samples_per_shard,
            compressed=compressed,
            ready_policy=ready_policy,
        )
    if round_count < worker_count:
        raise ValueError("round count must be at least the worker count")

    destination = Path(output)
    existing = tuple(destination.glob(f"{V4_DEMONSTRATION_PREFIX}_*.npz"))
    if existing or (destination / "manifest.json").exists():
        raise FileExistsError(
            f"V4 demonstration output already exists in {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    worker_root = destination / "_workers"
    if worker_root.exists():
        raise FileExistsError(f"worker directory already exists: {worker_root}")
    worker_root.mkdir()

    base_rounds, extra_rounds = divmod(round_count, worker_count)
    next_seed = seed_start
    worker_arguments: list[tuple[object, ...]] = []
    for worker_index in range(worker_count):
        worker_rounds = base_rounds + (1 if worker_index < extra_rounds else 0)
        worker_arguments.append((
            str(worker_root / f"worker_{worker_index:03d}"),
            worker_rounds,
            next_seed,
            teacher_seed + worker_index,
            samples_per_shard,
            compressed,
            ready_policy,
        ))
        next_seed += worker_rounds

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        worker_reports = list(executor.map(_collect_worker, worker_arguments))

    shard_index = 0
    for worker_index in range(worker_count):
        worker_directory = worker_root / f"worker_{worker_index:03d}"
        for source in sorted(worker_directory.glob(
            f"{V4_DEMONSTRATION_PREFIX}_*.npz"
        )):
            target = destination / (
                f"{V4_DEMONSTRATION_PREFIX}_{shard_index:06d}.npz"
            )
            source.replace(target)
            shard_index += 1

    category_counts = {
        name: sum(
            int(report["category_counts"][name])
            for report in worker_reports
        )
        for name in V4_CATEGORY_NAMES
    }
    end_reason_counts: Counter[str] = Counter()
    for report in worker_reports:
        end_reason_counts.update({
            str(name): int(count)
            for name, count in report["end_reason_counts"].items()
        })

    report = {
        "v4_demonstration_format_version": V4_DEMONSTRATION_FORMAT_VERSION,
        "v4_tensor_version": V4_TENSOR_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "round_count": round_count,
        "seed_start": seed_start,
        "seed_end": seed_start + round_count - 1,
        "teacher_seed": teacher_seed,
        "worker_teacher_seeds": [
            teacher_seed + index for index in range(worker_count)
        ],
        "teacher_kan_probability": 0.0,
        "teacher_ready_policy": ready_policy,
        "worker_count": worker_count,
        "samples_per_shard": samples_per_shard,
        "compressed": compressed,
        "shard_count": shard_index,
        "decision_count": sum(
            int(report["decision_count"]) for report in worker_reports
        ),
        "skipped_decision_count": sum(
            int(report["skipped_decision_count"])
            for report in worker_reports
        ),
        "sample_count": sum(
            int(report["sample_count"]) for report in worker_reports
        ),
        "category_counts": category_counts,
        "end_reason_counts": dict(sorted(end_reason_counts.items())),
        "elapsed_seconds": time.perf_counter() - started,
        "worker_elapsed_seconds": [
            float(report["elapsed_seconds"]) for report in worker_reports
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    shutil.rmtree(worker_root)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sharded V4 behavior-cloning demonstrations."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--teacher-seed", type=int, default=20260929)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ready-policy",
        choices=("skip", "declare", "pass"),
        default="skip",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers == 1:
        report = collect_v4_demonstrations(
            args.output,
            round_count=args.rounds,
            seed_start=args.seed_start,
            teacher_seed=args.teacher_seed,
            samples_per_shard=args.samples_per_shard,
            compressed=not args.no_compression,
            progress_interval=args.progress_interval,
            ready_policy=args.ready_policy,
        )
    else:
        report = collect_v4_demonstrations_parallel(
            args.output,
            round_count=args.rounds,
            seed_start=args.seed_start,
            worker_count=args.workers,
            teacher_seed=args.teacher_seed,
            samples_per_shard=args.samples_per_shard,
            compressed=not args.no_compression,
            ready_policy=args.ready_policy,
        )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
