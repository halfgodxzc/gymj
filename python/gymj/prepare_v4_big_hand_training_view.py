from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .collect_v4_big_hand_demonstrations import BIG_HAND_SIDECAR_NAME
from .v4_demonstration_dataset import (
    V4_CATEGORY_NAMES,
    V4DemonstrationDataset,
)


BIG_HAND_TRAINING_VIEW_FORMAT_VERSION = 1
BIG_HAND_TRAINING_VIEW_INDICES = "indices.npz"
BIG_HAND_TRAINING_VIEW_MANIFEST = "manifest.json"
BIG_HAND_TRAINING_VIEW_AUDIT = "audit_cn.md"

_ROUTE_IDS = {
    "seven_pairs": 0,
    "all_triplets": 1,
    "flush": 2,
}
_ROUTE_NAMES = tuple(_ROUTE_IDS)
_REASON_IDS = {
    "non_flush": 0,
    "strong_flush_start": 1,
    "recovered_flush": 2,
}
_REASON_NAMES = tuple(_REASON_IDS)


def prepare_v4_big_hand_training_view(
    dataset_path: Path | str,
    output_path: Path | str,
    *,
    split_seed: int = 20260929,
    flush_max_initial_shanten: int = 3,
    flush_recovery_shanten: int = 2,
) -> dict[str, object]:
    if flush_max_initial_shanten < -1:
        raise ValueError("flush initial shanten threshold cannot be below -1")
    if flush_recovery_shanten < -1:
        raise ValueError("flush recovery shanten threshold cannot be below -1")

    source = Path(dataset_path)
    output = Path(output_path)
    _ensure_empty_output(output)
    manifest_path = source / "manifest.json"
    sidecar_path = source / BIG_HAND_SIDECAR_NAME
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise ValueError("BigHand dataset is missing its manifest or sidecar")
    source_manifest = json.loads(manifest_path.read_text("ascii"))
    if source_manifest.get("collector") != "v4_big_hand":
        raise ValueError("source is not a V4 BigHand dataset")

    dataset = V4DemonstrationDataset(source)
    if len(dataset) != int(source_manifest["sample_count"]):
        raise ValueError("source dataset length does not match its manifest")
    sample_lookup = _sample_lookup(dataset)
    seen = np.zeros(len(dataset), dtype=np.bool_)
    eligible_records: list[tuple[int, int, int, bool]] = []
    excluded_indices: list[int] = []
    route_raw_counts: Counter[str] = Counter()
    route_eligible_counts: Counter[str] = Counter()
    route_excluded_counts: Counter[str] = Counter()
    category_eligible_counts: Counter[str] = Counter()
    category_excluded_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    episode_seeds = []

    for episode in _read_sidecar(sidecar_path):
        episode_seed = int(episode["episode_seed"])
        episode_seeds.append(episode_seed)
        for player in episode["players"]:
            player_id = int(player["player_id"])
            route = player.get("audit_route")
            if route not in _ROUTE_IDS:
                if any(
                    decision.get("sample_written")
                    for decision in player["decisions"]
                ):
                    raise ValueError("supervised BigHand player has no audit route")
                continue
            initial_shanten = player.get("initial_observed_route_shanten")
            for decision in player["decisions"]:
                if not decision.get("sample_written"):
                    continue
                key = (
                    episode_seed,
                    player_id,
                    int(decision["decision_index"]),
                )
                try:
                    sample_index = sample_lookup[key]
                except KeyError as error:
                    raise ValueError(
                        f"sidecar sample is missing from V4 shards: {key}"
                    ) from error
                if seen[sample_index]:
                    raise ValueError(f"duplicate sidecar sample: {key}")
                seen[sample_index] = True
                category = V4_CATEGORY_NAMES[
                    int(dataset.categories[sample_index])
                ]
                route_raw_counts[route] += 1
                reason = _eligibility_reason(
                    route=route,
                    initial_shanten=initial_shanten,
                    current_shanten=decision.get("route_shanten_before"),
                    flush_max_initial_shanten=flush_max_initial_shanten,
                    flush_recovery_shanten=flush_recovery_shanten,
                )
                if reason is None:
                    excluded_indices.append(sample_index)
                    route_excluded_counts[route] += 1
                    category_excluded_counts[category] += 1
                    continue
                eligible_records.append((
                    sample_index,
                    _ROUTE_IDS[route],
                    _REASON_IDS[reason],
                    bool(decision.get("teacher_target_disagreement")),
                ))
                route_eligible_counts[route] += 1
                category_eligible_counts[category] += 1
                reason_counts[reason] += 1

    if len(episode_seeds) != int(source_manifest["round_count"]):
        raise ValueError("sidecar episode count does not match source manifest")
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("sidecar contains duplicate episode seeds")
    if not np.all(seen):
        raise ValueError(
            f"sidecar omitted {int((~seen).sum())} V4 demonstration samples"
        )

    eligible_records.sort(key=lambda item: item[0])
    excluded_indices.sort()
    eligible_indices = np.asarray(
        [item[0] for item in eligible_records],
        dtype=np.int64,
    )
    eligible_route_ids = np.asarray(
        [item[1] for item in eligible_records],
        dtype=np.uint8,
    )
    eligible_reason_ids = np.asarray(
        [item[2] for item in eligible_records],
        dtype=np.uint8,
    )
    eligible_teacher_disagreement = np.asarray(
        [item[3] for item in eligible_records],
        dtype=np.bool_,
    )
    excluded = np.asarray(excluded_indices, dtype=np.int64)

    split_seeds = _split_episode_seeds(episode_seeds, split_seed)
    split_indices = {
        name: eligible_indices[np.isin(
            dataset.episode_seeds[eligible_indices],
            seeds,
        )]
        for name, seeds in split_seeds.items()
    }

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / BIG_HAND_TRAINING_VIEW_INDICES,
        format_version=np.asarray(
            BIG_HAND_TRAINING_VIEW_FORMAT_VERSION,
            dtype=np.uint32,
        ),
        source_sample_count=np.asarray(len(dataset), dtype=np.uint64),
        eligible_indices=eligible_indices,
        eligible_route_ids=eligible_route_ids,
        eligible_reason_ids=eligible_reason_ids,
        eligible_teacher_target_disagreement=(
            eligible_teacher_disagreement
        ),
        excluded_indices=excluded,
        train_indices=split_indices["train"],
        validation_indices=split_indices["validation"],
        test_indices=split_indices["test"],
        train_seeds=split_seeds["train"],
        validation_seeds=split_seeds["validation"],
        test_seeds=split_seeds["test"],
    )

    report: dict[str, object] = {
        "format_version": BIG_HAND_TRAINING_VIEW_FORMAT_VERSION,
        "source_dataset": str(source.resolve()),
        "source_round_count": int(source_manifest["round_count"]),
        "source_sample_count": len(dataset),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_sidecar_sha256": _sha256(sidecar_path),
        "filter": {
            "flush_max_initial_shanten": flush_max_initial_shanten,
            "flush_recovery_shanten": flush_recovery_shanten,
            "terminal_result_used": False,
        },
        "route_ids": dict(_ROUTE_IDS),
        "eligibility_reason_ids": dict(_REASON_IDS),
        "split_seed": split_seed,
        "eligible_sample_count": len(eligible_indices),
        "excluded_sample_count": len(excluded),
        "eligible_rate": len(eligible_indices) / len(dataset),
        "route_counts": _count_report(
            route_raw_counts,
            route_eligible_counts,
            route_excluded_counts,
            _ROUTE_NAMES,
        ),
        "category_counts": _count_report(
            Counter({
                name: (
                    category_eligible_counts[name]
                    + category_excluded_counts[name]
                )
                for name in V4_CATEGORY_NAMES
            }),
            category_eligible_counts,
            category_excluded_counts,
            V4_CATEGORY_NAMES,
        ),
        "eligibility_reason_counts": {
            name: int(reason_counts[name])
            for name in _REASON_NAMES
        },
        "eligible_teacher_target_disagreement_count": int(
            eligible_teacher_disagreement.sum()
        ),
        "eligible_teacher_target_disagreement_rate": float(
            eligible_teacher_disagreement.mean()
            if len(eligible_teacher_disagreement) else 0.0
        ),
        "splits": {
            name: {
                "episode_count": len(split_seeds[name]),
                "sample_count": len(split_indices[name]),
            }
            for name in ("train", "validation", "test")
        },
        "indices_file": BIG_HAND_TRAINING_VIEW_INDICES,
    }
    (output / BIG_HAND_TRAINING_VIEW_MANIFEST).write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    (output / BIG_HAND_TRAINING_VIEW_AUDIT).write_text(
        _audit_markdown(report),
        encoding="utf-8",
    )
    load_v4_big_hand_training_view(output, dataset)
    return report


def load_v4_big_hand_training_view(
    view_path: Path | str,
    dataset: V4DemonstrationDataset,
) -> dict[str, np.ndarray]:
    root = Path(view_path)
    manifest = json.loads(
        (root / BIG_HAND_TRAINING_VIEW_MANIFEST).read_text("ascii")
    )
    if manifest.get("format_version") != BIG_HAND_TRAINING_VIEW_FORMAT_VERSION:
        raise ValueError("incompatible BigHand training view version")
    if int(manifest["source_sample_count"]) != len(dataset):
        raise ValueError("BigHand training view belongs to another dataset")
    source_manifest = dataset.directory / "manifest.json"
    source_sidecar = dataset.directory / BIG_HAND_SIDECAR_NAME
    if (
        not source_manifest.is_file()
        or _sha256(source_manifest) != manifest["source_manifest_sha256"]
    ):
        raise ValueError("BigHand source manifest fingerprint changed")
    if (
        not source_sidecar.is_file()
        or _sha256(source_sidecar) != manifest["source_sidecar_sha256"]
    ):
        raise ValueError("BigHand source sidecar fingerprint changed")
    with np.load(
        root / BIG_HAND_TRAINING_VIEW_INDICES,
        allow_pickle=False,
    ) as source:
        if int(source["format_version"]) != BIG_HAND_TRAINING_VIEW_FORMAT_VERSION:
            raise ValueError("incompatible BigHand index version")
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    eligible = arrays["eligible_indices"].astype(np.int64, copy=False)
    excluded = arrays["excluded_indices"].astype(np.int64, copy=False)
    for name in (
        "eligible_route_ids",
        "eligible_reason_ids",
        "eligible_teacher_target_disagreement",
    ):
        if arrays[name].shape != eligible.shape:
            raise ValueError(f"{name} does not align with eligible indices")
    if np.any((eligible < 0) | (eligible >= len(dataset))):
        raise ValueError("eligible index is outside the source dataset")
    if np.any((excluded < 0) | (excluded >= len(dataset))):
        raise ValueError("excluded index is outside the source dataset")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible indices contain duplicates")
    if len(np.unique(excluded)) != len(excluded):
        raise ValueError("excluded indices contain duplicates")
    if np.intersect1d(eligible, excluded).size:
        raise ValueError("eligible and excluded BigHand samples overlap")
    if len(eligible) + len(excluded) != len(dataset):
        raise ValueError("BigHand training view does not cover the source")
    split_parts = [
        arrays[f"{name}_indices"].astype(np.int64, copy=False)
        for name in ("train", "validation", "test")
    ]
    combined = np.concatenate(split_parts)
    if not np.array_equal(np.sort(combined), eligible):
        raise ValueError("BigHand split indices do not partition eligibility")
    return arrays


def _sample_lookup(
    dataset: V4DemonstrationDataset,
) -> dict[tuple[int, int, int], int]:
    lookup = {
        (int(seed), int(player), int(decision)): index
        for index, (seed, player, decision) in enumerate(zip(
            dataset.episode_seeds,
            dataset.player_ids,
            dataset.decision_indices,
        ))
    }
    if len(lookup) != len(dataset):
        raise ValueError("V4 dataset contains duplicate sample identities")
    return lookup


def _eligibility_reason(
    *,
    route: str,
    initial_shanten: Optional[int],
    current_shanten: Optional[int],
    flush_max_initial_shanten: int,
    flush_recovery_shanten: int,
) -> Optional[str]:
    if route != "flush":
        return "non_flush"
    if initial_shanten is None:
        raise ValueError("flush route has no initial shanten")
    if int(initial_shanten) <= flush_max_initial_shanten:
        return "strong_flush_start"
    if (
        current_shanten is not None
        and int(current_shanten) <= flush_recovery_shanten
    ):
        return "recovered_flush"
    return None


def _split_episode_seeds(
    episode_seeds: list[int],
    split_seed: int,
) -> dict[str, np.ndarray]:
    unique = np.asarray(sorted(set(episode_seeds)), dtype=np.int64)
    if len(unique) < 3:
        raise ValueError("at least three episodes are required for a split")
    shuffled = np.random.default_rng(split_seed).permutation(unique)
    validation_count = max(1, int(round(len(unique) * 0.1)))
    test_count = max(1, int(round(len(unique) * 0.1)))
    train_count = len(unique) - validation_count - test_count
    if train_count <= 0:
        raise ValueError("episode split leaves no training episodes")
    return {
        "train": np.sort(shuffled[:train_count]),
        "validation": np.sort(
            shuffled[train_count:train_count + validation_count]
        ),
        "test": np.sort(shuffled[train_count + validation_count:]),
    }


def _count_report(
    raw: Mapping[str, int],
    eligible: Mapping[str, int],
    excluded: Mapping[str, int],
    names: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    return {
        name: {
            "raw": int(raw[name]),
            "eligible": int(eligible[name]),
            "excluded": int(excluded[name]),
        }
        for name in names
    }


def _read_sidecar(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid sidecar JSON at line {line_number}"
                ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _ensure_empty_output(output: Path) -> None:
    if not output.exists():
        return
    if any(output.iterdir()):
        raise FileExistsError(f"BigHand training view already exists: {output}")


def _audit_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# V4 BigHand Actor 训练视图",
        "",
        f"- 原始样本：{report['source_sample_count']}",
        f"- 合格样本：{report['eligible_sample_count']}",
        f"- 排除样本：{report['excluded_sample_count']}",
        f"- 保留率：{report['eligible_rate']:.3%}",
        "- 终局结果参与过滤：否",
        "",
        "## 路线",
        "",
        "| 路线 | 原始 | 合格 | 排除 |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "seven_pairs": "七对",
        "all_triplets": "大对子",
        "flush": "清一色",
    }
    for name, counts in report["route_counts"].items():
        lines.append(
            f"| {labels[name]} | {counts['raw']} | "
            f"{counts['eligible']} | {counts['excluded']} |"
        )
    lines.extend([
        "",
        "## 清一色过滤",
        "",
        f"- 强开局：{report['eligibility_reason_counts']['strong_flush_start']}",
        f"- 中途恢复：{report['eligibility_reason_counts']['recovered_flush']}",
        f"- Teacher目标外合格样本："
        f"{report['eligible_teacher_target_disagreement_count']} "
        f"({report['eligible_teacher_target_disagreement_rate']:.3%})",
        "",
        "## 数据划分",
        "",
        "| 划分 | 牌局 | 样本 |",
        "| --- | ---: | ---: |",
    ])
    for name, values in report["splits"].items():
        lines.append(
            f"| {name} | {values['episode_count']} | "
            f"{values['sample_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a filtered V4 BigHand Actor training view."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260929)
    parser.add_argument("--flush-max-initial-shanten", type=int, default=3)
    parser.add_argument("--flush-recovery-shanten", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = prepare_v4_big_hand_training_view(
        args.dataset,
        args.output,
        split_seed=args.split_seed,
        flush_max_initial_shanten=args.flush_max_initial_shanten,
        flush_recovery_shanten=args.flush_recovery_shanten,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
