from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import shutil
import time
from typing import Iterable, Mapping, Optional, Sequence

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    DECISION_FEATURE_VERSION,
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    HuPattern,
    ObservationEncoder,
)
from .big_hand_agent import (
    ANKAN_ACTION_START,
    DAIMINKAN_ACTION_ID,
    DECLARE_READY_ACTION_ID,
    KAKAN_ACTION_START,
    PASS_ACTION_ID,
    PON_ACTION_ID,
    RON_ACTION_ID,
    TSUMO_ACTION_ID,
    BigHandAgent,
    BigHandPlan,
    BigHandRoute,
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


BIG_HAND_SIDECAR_FORMAT_VERSION = 1
BIG_HAND_SIDECAR_NAME = "big_hand_episodes.jsonl.gz"
BIG_HAND_AUDIT_JSON = "data_audit.json"
BIG_HAND_AUDIT_MARKDOWN = "data_audit_cn.md"

_NORMAL_ROUTE_INDEX = 0
_SEVEN_PAIRS_ROUTE_INDEX = 1
_ALL_TRIPLETS_ROUTE_INDEX = 2
_FLUSH_SUIT_NAMES = ("wan", "tong", "tiao")
_TARGET_ROUTES = (
    BigHandRoute.SEVEN_PAIRS,
    BigHandRoute.ALL_TRIPLETS,
    BigHandRoute.FLUSH,
)
_SEVEN_PAIRS_HU_PATTERNS = {
    HuPattern.SEVEN_PAIRS,
    HuPattern.DRAGON_SEVEN_PAIRS,
    HuPattern.DI_LONG,
}


def collect_v4_big_hand_demonstrations(
    output: Path | str,
    *,
    round_count: int,
    seed_start: int,
    agent_seed: int = 20260928,
    teacher_seed: int = 20260929,
    samples_per_shard: int = 256,
    compressed: bool = True,
    progress_interval: int = 100,
    kan_probability: float = 0.5,
    seven_pairs_max_time_penalty: float = 1.0,
    dadui_switch_margin: float = 0.0,
) -> dict[str, object]:
    _validate_collection_arguments(
        round_count,
        samples_per_shard,
        progress_interval,
    )
    destination = Path(output)
    _ensure_empty_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)

    agent = BigHandAgent(
        seed=agent_seed,
        kan_probability=kan_probability,
        seven_pairs_max_time_penalty=seven_pairs_max_time_penalty,
        dadui_switch_margin=dadui_switch_margin,
    )
    category_counts: Counter[str] = Counter()
    end_reason_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    decision_count = 0
    skipped_decision_count = 0
    filtered_kan_decision_count = 0
    sample_count = 0
    started = time.perf_counter()
    sidecar_path = destination / BIG_HAND_SIDECAR_NAME

    with (
        V4DemonstrationWriter(
            destination,
            samples_per_shard=samples_per_shard,
            compressed=compressed,
        ) as writer,
        gzip.open(sidecar_path, "wt", encoding="utf-8", newline="\n") as sidecar,
    ):
        for round_index in range(round_count):
            episode_seed = seed_start + round_index
            agent.reset_episode(episode_seed)
            teachers = _shadow_teachers(episode_seed, teacher_seed)
            arena = Arena(seed=episode_seed)
            arena.start_round()
            episode = _new_episode_record(episode_seed)
            round_decision_index = 0

            while not arena.finished:
                current = arena.current_decision()
                player_id = int(current.player)
                encoded = ActionEncoder.encode_legal_actions(
                    current.legal_actions
                )
                legal_mask = tuple(
                    int(encoded.is_legal(action_id))
                    for action_id in range(ACTION_SPACE_SIZE)
                )
                observation = ObservationEncoder.encode(arena, player_id)
                state = EnvState(
                    player_id=player_id,
                    observation=observation,
                    legal_mask=legal_mask,
                )
                transition_count = len(agent.transitions)
                decision = agent.decide(state)
                plan = agent.plan_for_observation(observation)
                teacher_decision = teachers[player_id].decide(state)
                selected = int(decision.selected_action_id)
                if not encoded.is_legal(selected):
                    raise RuntimeError("big-hand agent selected an illegal action")

                new_transitions = agent.transitions[transition_count:]
                transition_records = [
                    {
                        "player_id": player_id,
                        "decision_index": round_decision_index,
                        "from_route": transition.from_route.value,
                        "to_route": transition.to_route.value,
                        "reason": transition.reason,
                        "wall_remaining": int(transition.wall_remaining),
                    }
                    for transition in new_transitions
                ]
                episode["route_transitions"].extend(transition_records)
                features = None
                if sum(legal_mask) == 1:
                    decision_record = _forced_decision_record(
                        player_id=player_id,
                        decision_index=round_decision_index,
                        player_record=episode["players"][player_id],
                        observation=observation,
                        plan=plan,
                        decision=decision,
                        teacher_decision=teacher_decision,
                    )
                else:
                    features = DecisionFeatureEncoder.encode(arena, player_id)
                    decision_record = _decision_record(
                        player_id=player_id,
                        decision_index=round_decision_index,
                        player_record=episode["players"][player_id],
                        observation=observation,
                        features=features,
                        plan=plan,
                        decision=decision,
                        teacher_decision=teacher_decision,
                        transition_records=transition_records,
                    )
                episode["players"][player_id]["decisions"].append(
                    decision_record
                )
                if plan is not None:
                    _update_player_plan_record(
                        episode["players"][player_id],
                        plan,
                        decision_record,
                    )

                is_filtered_kan = _is_kan_action_id(selected)
                sample_written = False
                if decision.trainable and not is_filtered_kan:
                    if features is None:
                        raise RuntimeError(
                            "forced action unexpectedly requested supervision"
                        )
                    sample = V4DemonstrationSample.from_teacher_decision(
                        episode_seed=episode_seed,
                        player_id=player_id,
                        decision_index=round_decision_index,
                        features=features,
                        decision=decision,
                    )
                    writer.append(sample)
                    category_counts[V4_CATEGORY_NAMES[sample.category]] += 1
                    sample_count += 1
                    sample_written = True
                elif decision.trainable and is_filtered_kan:
                    filtered_kan_decision_count += 1
                else:
                    skipped_decision_count += 1
                decision_record["sample_written"] = sample_written
                decision_record["kan_filtered"] = bool(
                    decision.trainable and is_filtered_kan
                )

                action_counts[_action_category(selected)] += 1
                arena.submit_action(encoded.action_at(selected))
                decision_count += 1
                round_decision_index += 1

            _finish_episode_record(episode, arena)
            sidecar.write(json.dumps(
                episode,
                ensure_ascii=False,
                separators=(",", ":"),
            ))
            sidecar.write("\n")
            end_reason_counts[episode["end_reason"]] += 1
            completed = round_index + 1
            if progress_interval and completed % progress_interval == 0:
                print(
                    f"rounds={completed}/{round_count} "
                    f"samples={sample_count} decisions={decision_count}",
                    flush=True,
                )

    report = _collection_report(
        round_count=round_count,
        seed_start=seed_start,
        agent_seed=agent_seed,
        teacher_seed=teacher_seed,
        samples_per_shard=samples_per_shard,
        compressed=compressed,
        kan_probability=kan_probability,
        seven_pairs_max_time_penalty=seven_pairs_max_time_penalty,
        dadui_switch_margin=dadui_switch_margin,
        decision_count=decision_count,
        skipped_decision_count=skipped_decision_count,
        filtered_kan_decision_count=filtered_kan_decision_count,
        sample_count=sample_count,
        category_counts=category_counts,
        action_counts=action_counts,
        end_reason_counts=end_reason_counts,
        elapsed_seconds=time.perf_counter() - started,
    )
    report["shard_count"] = len(tuple(destination.glob(
        f"{V4_DEMONSTRATION_PREFIX}_*.npz"
    )))
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    audit_v4_big_hand_dataset(destination)
    return report


def collect_v4_big_hand_demonstrations_parallel(
    output: Path | str,
    *,
    round_count: int,
    seed_start: int,
    worker_count: int,
    agent_seed: int = 20260928,
    teacher_seed: int = 20260929,
    samples_per_shard: int = 256,
    compressed: bool = True,
    kan_probability: float = 0.5,
    seven_pairs_max_time_penalty: float = 1.0,
    dadui_switch_margin: float = 0.0,
) -> dict[str, object]:
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    if worker_count == 1:
        return collect_v4_big_hand_demonstrations(
            output,
            round_count=round_count,
            seed_start=seed_start,
            agent_seed=agent_seed,
            teacher_seed=teacher_seed,
            samples_per_shard=samples_per_shard,
            compressed=compressed,
            kan_probability=kan_probability,
            seven_pairs_max_time_penalty=seven_pairs_max_time_penalty,
            dadui_switch_margin=dadui_switch_margin,
        )
    if round_count < worker_count:
        raise ValueError("round count must be at least the worker count")
    _validate_collection_arguments(round_count, samples_per_shard, 0)

    destination = Path(output)
    _ensure_empty_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    worker_root = destination / "_workers"
    worker_root.mkdir()

    base_rounds, extra_rounds = divmod(round_count, worker_count)
    next_seed = seed_start
    arguments = []
    for worker_index in range(worker_count):
        worker_rounds = base_rounds + (worker_index < extra_rounds)
        arguments.append((
            str(worker_root / f"worker_{worker_index:03d}"),
            int(worker_rounds),
            next_seed,
            agent_seed,
            teacher_seed,
            samples_per_shard,
            compressed,
            kan_probability,
            seven_pairs_max_time_penalty,
            dadui_switch_margin,
        ))
        next_seed += int(worker_rounds)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        worker_reports = list(executor.map(_collect_worker, arguments))

    shard_index = 0
    final_sidecar = destination / BIG_HAND_SIDECAR_NAME
    with gzip.open(final_sidecar, "wt", encoding="utf-8", newline="\n") as out:
        for worker_index in range(worker_count):
            worker = worker_root / f"worker_{worker_index:03d}"
            for source in sorted(worker.glob(
                f"{V4_DEMONSTRATION_PREFIX}_*.npz"
            )):
                target = destination / (
                    f"{V4_DEMONSTRATION_PREFIX}_{shard_index:06d}.npz"
                )
                source.replace(target)
                shard_index += 1
            with gzip.open(
                worker / BIG_HAND_SIDECAR_NAME,
                "rt",
                encoding="utf-8",
            ) as source_sidecar:
                shutil.copyfileobj(source_sidecar, out)

    category_counts = _sum_nested_counts(
        worker_reports,
        "category_counts",
        V4_CATEGORY_NAMES,
    )
    action_names = sorted({
        name
        for report in worker_reports
        for name in report["action_counts"]
    })
    action_counts = _sum_nested_counts(
        worker_reports,
        "action_counts",
        action_names,
    )
    end_names = sorted({
        name
        for report in worker_reports
        for name in report["end_reason_counts"]
    })
    end_reason_counts = _sum_nested_counts(
        worker_reports,
        "end_reason_counts",
        end_names,
    )
    report = _collection_report(
        round_count=round_count,
        seed_start=seed_start,
        agent_seed=agent_seed,
        teacher_seed=teacher_seed,
        samples_per_shard=samples_per_shard,
        compressed=compressed,
        kan_probability=kan_probability,
        seven_pairs_max_time_penalty=seven_pairs_max_time_penalty,
        dadui_switch_margin=dadui_switch_margin,
        decision_count=sum(int(item["decision_count"]) for item in worker_reports),
        skipped_decision_count=sum(
            int(item["skipped_decision_count"]) for item in worker_reports
        ),
        filtered_kan_decision_count=sum(
            int(item["filtered_kan_decision_count"])
            for item in worker_reports
        ),
        sample_count=sum(int(item["sample_count"]) for item in worker_reports),
        category_counts=Counter(category_counts),
        action_counts=Counter(action_counts),
        end_reason_counts=Counter(end_reason_counts),
        elapsed_seconds=time.perf_counter() - started,
    )
    report.update({
        "worker_count": worker_count,
        "shard_count": shard_index,
        "worker_elapsed_seconds": [
            float(item["elapsed_seconds"]) for item in worker_reports
        ],
    })
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    shutil.rmtree(worker_root)
    audit_v4_big_hand_dataset(destination)
    return report


def merge_v4_big_hand_datasets(
    output: Path | str,
    sources: Sequence[Path | str],
) -> dict[str, object]:
    """Copy complete, contiguous BigHand datasets into one audited dataset."""
    if not sources:
        raise ValueError("at least one source dataset is required")

    destination = Path(output)
    _ensure_empty_destination(destination)
    source_records = []
    for source_value in sources:
        source = Path(source_value)
        manifest_path = source / "manifest.json"
        sidecar_path = source / BIG_HAND_SIDECAR_NAME
        if not manifest_path.is_file() or not sidecar_path.is_file():
            raise ValueError(f"incomplete BigHand dataset: {source}")
        report = json.loads(manifest_path.read_text("ascii"))
        shards = tuple(sorted(source.glob(
            f"{V4_DEMONSTRATION_PREFIX}_*.npz"
        )))
        if len(shards) != int(report.get("shard_count", -1)):
            raise ValueError(f"shard count does not match manifest: {source}")
        round_count = int(report["round_count"])
        seed_start = int(report["seed_start"])
        seed_end = int(report["seed_end"])
        if seed_end != seed_start + round_count - 1:
            raise ValueError(f"invalid seed range in manifest: {source}")
        source_records.append((seed_start, source, report, shards))

    source_records.sort(key=lambda item: item[0])
    reference = source_records[0][2]
    compatibility_fields = (
        "collector",
        "big_hand_sidecar_format_version",
        "v4_demonstration_format_version",
        "v4_tensor_version",
        "decision_feature_version",
        "action_encoding_version",
        "agent_seed",
        "teacher_seed",
        "big_hand_config",
        "samples_per_shard",
        "compressed",
    )
    expected_seed = int(reference["seed_start"])
    for _, source, report, _ in source_records:
        for field in compatibility_fields:
            if report.get(field) != reference.get(field):
                raise ValueError(
                    f"incompatible {field} in source dataset: {source}"
                )
        if int(report["seed_start"]) != expected_seed:
            raise ValueError(
                "source seed ranges must be contiguous and non-overlapping"
            )
        expected_seed = int(report["seed_end"]) + 1

    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shard_index = 0
    final_sidecar = destination / BIG_HAND_SIDECAR_NAME
    with gzip.open(
        final_sidecar,
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as out:
        for _, source, _, shards in source_records:
            for source_shard in shards:
                target = destination / (
                    f"{V4_DEMONSTRATION_PREFIX}_{shard_index:06d}.npz"
                )
                shutil.copy2(source_shard, target)
                shard_index += 1
            with gzip.open(
                source / BIG_HAND_SIDECAR_NAME,
                "rt",
                encoding="utf-8",
            ) as source_sidecar:
                shutil.copyfileobj(source_sidecar, out)

    reports = [item[2] for item in source_records]
    category_counts = _sum_nested_counts(
        reports,
        "category_counts",
        V4_CATEGORY_NAMES,
    )
    action_names = sorted({
        name
        for report in reports
        for name in report["action_counts"]
    })
    action_counts = _sum_nested_counts(
        reports,
        "action_counts",
        action_names,
    )
    end_names = sorted({
        name
        for report in reports
        for name in report["end_reason_counts"]
    })
    end_reason_counts = _sum_nested_counts(
        reports,
        "end_reason_counts",
        end_names,
    )
    round_count = sum(int(item["round_count"]) for item in reports)
    report = _collection_report(
        round_count=round_count,
        seed_start=int(reports[0]["seed_start"]),
        agent_seed=int(reference["agent_seed"]),
        teacher_seed=int(reference["teacher_seed"]),
        samples_per_shard=int(reference["samples_per_shard"]),
        compressed=bool(reference["compressed"]),
        kan_probability=float(reference["big_hand_config"]["kan_probability"]),
        seven_pairs_max_time_penalty=float(
            reference["big_hand_config"]["seven_pairs_max_time_penalty"]
        ),
        dadui_switch_margin=float(
            reference["big_hand_config"]["dadui_switch_margin"]
        ),
        decision_count=sum(int(item["decision_count"]) for item in reports),
        skipped_decision_count=sum(
            int(item["skipped_decision_count"]) for item in reports
        ),
        filtered_kan_decision_count=sum(
            int(item["filtered_kan_decision_count"])
            for item in reports
        ),
        sample_count=sum(int(item["sample_count"]) for item in reports),
        category_counts=Counter(category_counts),
        action_counts=Counter(action_counts),
        end_reason_counts=Counter(end_reason_counts),
        elapsed_seconds=time.perf_counter() - started,
    )
    report.update({
        "merge_source_count": len(source_records),
        "merge_sources": [str(item[1]) for item in source_records],
        "shard_count": shard_index,
    })
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="ascii",
    )
    audit_v4_big_hand_dataset(destination)
    return report


def audit_v4_big_hand_dataset(directory: Path | str) -> dict[str, object]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text("ascii"))
    route_stats = {
        route.value: Counter()
        for route in _TARGET_ROUTES
    }
    route_score_sums = Counter()
    action_counts: Counter[str] = Counter()
    hu_pattern_counts: Counter[str] = Counter()
    route_transition_counts: Counter[str] = Counter()
    seen_seeds = set()
    counters: Counter[str] = Counter()

    for episode in _read_sidecar(root / BIG_HAND_SIDECAR_NAME):
        seed = int(episode["episode_seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate episode seed in sidecar: {seed}")
        seen_seeds.add(seed)
        scores = tuple(int(value) for value in episode["score_deltas"])
        if sum(scores) != 0:
            raise ValueError(f"non-zero-sum episode at seed {seed}")
        counters["round_count"] += 1
        for transition in episode["route_transitions"]:
            route_transition_counts[
                f"{transition['from_route']}->{transition['to_route']}:"
                f"{transition['reason']}"
            ] += 1
        for player in episode["players"]:
            for decision in player["decisions"]:
                counters["decision_count"] += 1
                counters["trainable_decision_count"] += int(
                    decision["trainable"]
                )
                counters["sample_count"] += int(decision["sample_written"])
                counters["filtered_kan_decision_count"] += int(
                    decision["kan_filtered"]
                )
                counters["teacher_exact_difference_count"] += int(
                    decision["teacher_exact_difference"]
                )
                counters["teacher_target_disagreement_count"] += int(
                    decision["teacher_target_disagreement"]
                )
                counters["teacher_comparable_decision_count"] += int(
                    decision["teacher_comparable"]
                )
                strategic = len(decision["legal_action_ids"]) > 1
                counters["strategic_decision_count"] += int(strategic)
                counters["strategic_teacher_exact_difference_count"] += int(
                    strategic and decision["teacher_exact_difference"]
                )
                counters[
                    "strategic_teacher_target_disagreement_count"
                ] += int(
                    strategic and decision["teacher_target_disagreement"]
                )
                counters["pon_opportunity_count"] += int(
                    decision["pon_opportunity"]
                )
                counters["pon_selected_count"] += int(
                    decision["pon_selected"]
                )
                counters["pon_pass_count"] += int(
                    decision["pon_opportunity"]
                    and decision["selected_action_id"] == PASS_ACTION_ID
                )
                if decision["normal_shanten_gap"] is not None:
                    counters["normal_shanten_comparison_count"] += 1
                    gap = int(decision["normal_shanten_gap"])
                    counters["normal_shanten_sacrifice_count"] += int(gap > 0)
                    counters["normal_shanten_gap_sum"] += gap
                action_counts[decision["action_category"]] += 1
            route = player["audit_route"]
            if route is None:
                counters["unplanned_player_rounds"] += 1
                continue
            stats = route_stats[route]
            stats["player_rounds"] += 1
            stats["target_completed"] += int(player["target_completed"])
            stats["hu_count"] += int(player["hu"] is not None)
            route_score_sums[route] += int(player["net_score"])
            if player["hu"] is not None:
                hu_pattern_counts[player["hu"]["pattern_name"]] += 1

    if counters["round_count"] != int(manifest["round_count"]):
        raise ValueError("sidecar round count does not match manifest")
    if counters["sample_count"] != int(manifest["sample_count"]):
        raise ValueError("sidecar sample count does not match manifest")
    if counters["decision_count"] != int(manifest["decision_count"]):
        raise ValueError("sidecar decision count does not match manifest")

    routes = {}
    for route in _TARGET_ROUTES:
        name = route.value
        stats = route_stats[name]
        count = int(stats["player_rounds"])
        routes[name] = {
            "player_rounds": count,
            "target_completed": int(stats["target_completed"]),
            "target_completion_rate": (
                stats["target_completed"] / count if count else 0.0
            ),
            "hu_count": int(stats["hu_count"]),
            "hu_rate": stats["hu_count"] / count if count else 0.0,
            "average_net_score": (
                route_score_sums[name] / count if count else 0.0
            ),
        }
    comparable = int(counters["teacher_comparable_decision_count"])
    strategic = int(counters["strategic_decision_count"])
    normal_count = int(counters["normal_shanten_comparison_count"])
    audit = {
        "format": "gymj_v4_big_hand_data_audit_v1",
        "round_count": int(counters["round_count"]),
        "planned_player_rounds": sum(
            route["player_rounds"] for route in routes.values()
        ),
        "unplanned_player_rounds": int(counters["unplanned_player_rounds"]),
        "routes": routes,
        "decisions": {
            "total": int(counters["decision_count"]),
            "trainable": int(counters["trainable_decision_count"]),
            "samples_written": int(counters["sample_count"]),
            "kan_filtered": int(counters["filtered_kan_decision_count"]),
            "action_counts": dict(sorted(action_counts.items())),
        },
        "teacher_comparison": {
            "comparable_decisions": comparable,
            "exact_difference_count": int(
                counters["teacher_exact_difference_count"]
            ),
            "exact_difference_rate": (
                counters["teacher_exact_difference_count"] / comparable
                if comparable else 0.0
            ),
            "target_disagreement_count": int(
                counters["teacher_target_disagreement_count"]
            ),
            "target_disagreement_rate": (
                counters["teacher_target_disagreement_count"] / comparable
                if comparable else 0.0
            ),
            "strategic_decisions": strategic,
            "strategic_exact_difference_count": int(
                counters["strategic_teacher_exact_difference_count"]
            ),
            "strategic_exact_difference_rate": (
                counters["strategic_teacher_exact_difference_count"]
                / strategic if strategic else 0.0
            ),
            "strategic_target_disagreement_count": int(
                counters["strategic_teacher_target_disagreement_count"]
            ),
            "strategic_target_disagreement_rate": (
                counters["strategic_teacher_target_disagreement_count"]
                / strategic if strategic else 0.0
            ),
        },
        "pon": {
            "opportunity_count": int(counters["pon_opportunity_count"]),
            "selected_count": int(counters["pon_selected_count"]),
            "pass_count": int(counters["pon_pass_count"]),
            "selection_rate": (
                counters["pon_selected_count"]
                / counters["pon_opportunity_count"]
                if counters["pon_opportunity_count"] else 0.0
            ),
        },
        "normal_shanten_sacrifice": {
            "comparable_discard_decisions": normal_count,
            "sacrifice_count": int(
                counters["normal_shanten_sacrifice_count"]
            ),
            "sacrifice_rate": (
                counters["normal_shanten_sacrifice_count"] / normal_count
                if normal_count else 0.0
            ),
            "average_gap": (
                counters["normal_shanten_gap_sum"] / normal_count
                if normal_count else 0.0
            ),
        },
        "hu_pattern_counts": dict(sorted(hu_pattern_counts.items())),
        "route_transition_counts": dict(sorted(
            route_transition_counts.items()
        )),
    }
    (root / BIG_HAND_AUDIT_JSON).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / BIG_HAND_AUDIT_MARKDOWN).write_text(
        _audit_markdown(audit),
        encoding="utf-8",
    )
    return audit


def _decision_record(
    *,
    player_id: int,
    decision_index: int,
    player_record: Mapping[str, object],
    observation,
    features,
    plan: Optional[BigHandPlan],
    decision,
    teacher_decision,
    transition_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected = int(decision.selected_action_id)
    current_hand = features.public_state.hand
    route = (
        _resolve_route(plan, current_hand, int(observation.wall_remaining))
        if plan is not None else None
    )
    current_hand_route_shanten = (
        _route_shanten_from_features(
            current_hand,
            route,
            plan.flush_suit,
        )
        if plan is not None and route is not None else None
    )
    post_hand = (
        _post_action_hand(features, selected, route, plan.flush_suit)
        if plan is not None and route is not None else None
    )
    route_shanten_after = (
        _route_shanten_from_features(
            post_hand,
            route,
            plan.flush_suit,
        )
        if post_hand is not None and route is not None
        else None
    )
    if selected in (TSUMO_ACTION_ID, RON_ACTION_ID) and route is not None:
        route_shanten_after = -1
    previous_route = player_record["audit_route"]
    previous_shanten = player_record["final_observed_route_shanten"]
    if route is None:
        route_shanten_before = None
    elif previous_route == route.value and previous_shanten is not None:
        route_shanten_before = int(previous_shanten)
    elif player_record["initial_agent_route"] is None and plan is not None:
        route_shanten_before = int(plan.initial_shanten)
    else:
        route_shanten_before = None
    supervised = tuple(int(value) for value in decision.supervised_action_ids)
    target_ids = tuple(int(value) for value in decision.target_action_ids)
    teacher_targets = tuple(
        int(value) for value in teacher_decision.target_action_ids
    )
    teacher_comparable = True
    if teacher_decision.trainable:
        teacher_agrees = selected in teacher_targets
    else:
        teacher_agrees = selected == int(teacher_decision.selected_action_id)
    normal_gap = _normal_shanten_gap(features, supervised, selected)
    legal_ids = tuple(
        action_id
        for action_id, action in enumerate(features.actions)
        if action.legal
    )
    return {
        "player_id": player_id,
        "decision_index": decision_index,
        "wall_remaining": int(observation.wall_remaining),
        "agent_route": plan.route.value if plan is not None else None,
        "route": route.value if route is not None else None,
        "flush_suit": plan.flush_suit if plan is not None else None,
        "current_hand_is_stable": bool(current_hand.post_discard_state),
        "current_hand_route_shanten": current_hand_route_shanten,
        "route_shanten_before": route_shanten_before,
        "route_shanten_after_action": route_shanten_after,
        "route_shanten_improvement": (
            route_shanten_before - route_shanten_after
            if route_shanten_before is not None
            and route_shanten_after is not None
            else None
        ),
        "selected_action_id": selected,
        "action_category": _action_category(selected),
        "legal_action_ids": list(legal_ids),
        "trainable": bool(decision.trainable),
        "supervised_action_ids": list(supervised),
        "target_action_ids": list(target_ids),
        "teacher_selected_action_id": int(
            teacher_decision.selected_action_id
        ),
        "teacher_target_action_ids": list(teacher_targets),
        "teacher_comparable": teacher_comparable,
        "teacher_exact_difference": (
            selected != int(teacher_decision.selected_action_id)
        ),
        "teacher_target_disagreement": not teacher_agrees,
        "pon_opportunity": (
            PON_ACTION_ID in legal_ids and PASS_ACTION_ID in legal_ids
        ),
        "pon_selected": selected == PON_ACTION_ID,
        "normal_shanten_gap": normal_gap,
        "route_transitions": list(transition_records),
    }


def _forced_decision_record(
    *,
    player_id: int,
    decision_index: int,
    player_record: Mapping[str, object],
    observation,
    plan: Optional[BigHandPlan],
    decision,
    teacher_decision,
) -> dict[str, object]:
    selected = int(decision.selected_action_id)
    teacher_selected = int(teacher_decision.selected_action_id)
    if decision.trainable or teacher_decision.trainable:
        raise RuntimeError("single legal action must not be supervised")
    previous_shanten = player_record["final_observed_route_shanten"]
    return {
        "player_id": player_id,
        "decision_index": decision_index,
        "wall_remaining": int(observation.wall_remaining),
        "agent_route": plan.route.value if plan is not None else None,
        "route": player_record["audit_route"],
        "flush_suit": plan.flush_suit if plan is not None else None,
        "current_hand_is_stable": None,
        "current_hand_route_shanten": None,
        "route_shanten_before": previous_shanten,
        "route_shanten_after_action": previous_shanten,
        "route_shanten_improvement": (
            0 if previous_shanten is not None else None
        ),
        "selected_action_id": selected,
        "action_category": _action_category(selected),
        "legal_action_ids": [selected],
        "trainable": False,
        "supervised_action_ids": [],
        "target_action_ids": [],
        "teacher_selected_action_id": teacher_selected,
        "teacher_target_action_ids": [],
        "teacher_comparable": True,
        "teacher_exact_difference": selected != teacher_selected,
        "teacher_target_disagreement": selected != teacher_selected,
        "pon_opportunity": False,
        "pon_selected": False,
        "normal_shanten_gap": None,
        "route_transitions": [],
    }


def _post_action_hand(features, action_id: int, route, flush_suit):
    if _is_kan_action_id(action_id):
        return None
    if 0 <= action_id < ANKAN_ACTION_START:
        candidate = features.actions[action_id]
        return candidate.post_hand if candidate.post_hand_known else None
    if action_id == PON_ACTION_ID:
        follow_up = features.pon_follow_up
        if not follow_up.available:
            return None
        hands = [
            hand
            for tile_id, hand in enumerate(follow_up.post_discard_hands)
            if follow_up.discard_mask[tile_id]
        ]
        if not hands:
            return None
        return min(
            hands,
            key=lambda hand: _route_shanten_from_features(
                hand,
                route,
                flush_suit,
            ),
        )
    return features.public_state.hand


def _normal_shanten_gap(
    features,
    supervised_action_ids: Sequence[int],
    selected_action_id: int,
) -> Optional[int]:
    discards = tuple(
        action_id
        for action_id in supervised_action_ids
        if 0 <= action_id < ANKAN_ACTION_START
    )
    if len(discards) != len(supervised_action_ids):
        return None
    if not discards or selected_action_id not in discards:
        return None
    shanten_by_action = {
        action_id: int(
            features.actions[action_id]
            .post_hand.routes[_NORMAL_ROUTE_INDEX].shanten
        )
        for action_id in discards
    }
    return shanten_by_action[selected_action_id] - min(shanten_by_action.values())


def _resolve_route(plan: BigHandPlan, hand, wall_remaining: int) -> BigHandRoute:
    if plan.route != BigHandRoute.PAIR_FLEXIBLE:
        return plan.route
    seven_pairs = float(
        hand.routes[_SEVEN_PAIRS_ROUTE_INDEX].shanten
    )
    if seven_pairs > 0:
        progress = 1.0 - wall_remaining / 56.0
        seven_pairs += min(1.0, max(0.0, progress))
    all_triplets = float(
        hand.routes[_ALL_TRIPLETS_ROUTE_INDEX].shanten
    )
    return (
        BigHandRoute.SEVEN_PAIRS
        if seven_pairs <= all_triplets
        else BigHandRoute.ALL_TRIPLETS
    )


def _route_shanten_from_features(
    hand,
    route: BigHandRoute,
    flush_suit: Optional[int],
) -> int:
    if route == BigHandRoute.SEVEN_PAIRS:
        return int(hand.routes[_SEVEN_PAIRS_ROUTE_INDEX].shanten)
    if route == BigHandRoute.ALL_TRIPLETS:
        return int(hand.routes[_ALL_TRIPLETS_ROUTE_INDEX].shanten)
    if route == BigHandRoute.FLUSH:
        if flush_suit is None or not 0 <= flush_suit < 3:
            raise ValueError("flush route requires a valid target suit")
        return int(hand.flushes[flush_suit].shanten)
    raise ValueError(f"unsupported audit route: {route}")


def _update_player_plan_record(
    player: dict[str, object],
    plan: BigHandPlan,
    decision: Mapping[str, object],
) -> None:
    if player["initial_agent_route"] is None:
        player["initial_agent_route"] = plan.route.value
        player["initial_route"] = decision["route"]
        player["flush_suit"] = plan.flush_suit
        player["flush_suit_name"] = (
            _FLUSH_SUIT_NAMES[plan.flush_suit]
            if plan.flush_suit is not None else None
        )
        player["plan_initial_shanten"] = int(plan.initial_shanten)
        player["initial_observed_route_shanten"] = int(plan.initial_shanten)
    player["final_agent_route"] = plan.route.value
    player["audit_route"] = decision["route"]
    if decision["route_shanten_after_action"] is not None:
        player["final_observed_route_shanten"] = decision[
            "route_shanten_after_action"
        ]


def _finish_episode_record(episode: dict[str, object], arena: Arena) -> None:
    result = arena.result
    settlement = arena.round_settlement
    scores = [int(value) for value in settlement.ledger.score_deltas]
    if sum(scores) != 0:
        raise RuntimeError("big-hand collection settlement is not zero-sum")
    hu_by_player = {}
    for item in settlement.hu_settlements:
        pattern = item.features.pattern
        hu_by_player[int(item.player_id)] = {
            "pattern": int(pattern),
            "pattern_name": _enum_name(pattern),
            "is_flush": bool(item.features.is_flush),
            "total_fan": int(item.score.total_fan),
            "hu_points": int(item.score.hu_points),
        }
    episode["end_reason"] = _enum_name(result.reason)
    episode["score_deltas"] = scores
    episode["hu_players"] = [int(value) for value in result.hu_players]
    episode["fangpao_player"] = int(result.fangpao_player)
    episode["hu_settlements"] = list(hu_by_player.values())
    for player_id, player in enumerate(episode["players"]):
        hu = hu_by_player.get(player_id)
        player["net_score"] = scores[player_id]
        player["hu"] = hu
        route = player["audit_route"]
        if (
            route is not None
            and hu is not None
            and player["final_agent_route"] == BigHandRoute.PAIR_FLEXIBLE.value
        ):
            player["audit_route"] = _completed_route_override(route, hu)
        player["target_completed"] = _target_completed(
            player["audit_route"],
            hu,
        )


def _completed_route_override(route: str, hu: Mapping[str, object]) -> str:
    pattern = int(hu["pattern"])
    if pattern in {int(value) for value in _SEVEN_PAIRS_HU_PATTERNS}:
        return BigHandRoute.SEVEN_PAIRS.value
    if pattern == int(HuPattern.ALL_TRIPLETS):
        return BigHandRoute.ALL_TRIPLETS.value
    return route


def _target_completed(
    route: Optional[str],
    hu: Optional[Mapping[str, object]],
) -> bool:
    if route is None or hu is None:
        return False
    pattern = int(hu["pattern"])
    if route == BigHandRoute.FLUSH.value:
        return bool(hu["is_flush"])
    if route == BigHandRoute.ALL_TRIPLETS.value:
        return pattern == int(HuPattern.ALL_TRIPLETS)
    if route == BigHandRoute.SEVEN_PAIRS.value:
        return pattern in {int(value) for value in _SEVEN_PAIRS_HU_PATTERNS}
    return False


def _new_episode_record(seed: int) -> dict[str, object]:
    return {
        "format_version": BIG_HAND_SIDECAR_FORMAT_VERSION,
        "episode_seed": seed,
        "players": [
            {
                "player_id": player_id,
                "initial_agent_route": None,
                "initial_route": None,
                "final_agent_route": None,
                "audit_route": None,
                "flush_suit": None,
                "flush_suit_name": None,
                "plan_initial_shanten": None,
                "initial_observed_route_shanten": None,
                "final_observed_route_shanten": None,
                "target_completed": False,
                "net_score": 0,
                "hu": None,
                "decisions": [],
            }
            for player_id in range(4)
        ],
        "route_transitions": [],
    }


def _shadow_teachers(seed: int, base_seed: int) -> tuple[ShantenTeacher, ...]:
    return tuple(
        ShantenTeacher(
            seed=(
                base_seed
                ^ seed
                ^ (0x9E3779B9 * (player_id + 1))
            ),
            kan_probability=0.0,
            ready_policy="declare",
        )
        for player_id in range(4)
    )


def _is_kan_action_id(action_id: int) -> bool:
    return (
        ANKAN_ACTION_START <= action_id < PON_ACTION_ID
        or action_id == DAIMINKAN_ACTION_ID
    )


def _action_category(action_id: int) -> str:
    if 0 <= action_id < ANKAN_ACTION_START:
        return "discard"
    if ANKAN_ACTION_START <= action_id < KAKAN_ACTION_START:
        return "ankan"
    if KAKAN_ACTION_START <= action_id < PON_ACTION_ID:
        return "kakan"
    return {
        PON_ACTION_ID: "pon",
        DAIMINKAN_ACTION_ID: "daiminkan",
        TSUMO_ACTION_ID: "tsumo",
        RON_ACTION_ID: "ron",
        PASS_ACTION_ID: "pass",
        DECLARE_READY_ACTION_ID: "ready",
    }.get(action_id, "unknown")


def _collection_report(
    *,
    round_count: int,
    seed_start: int,
    agent_seed: int,
    teacher_seed: int,
    samples_per_shard: int,
    compressed: bool,
    kan_probability: float,
    seven_pairs_max_time_penalty: float,
    dadui_switch_margin: float,
    decision_count: int,
    skipped_decision_count: int,
    filtered_kan_decision_count: int,
    sample_count: int,
    category_counts: Mapping[str, int],
    action_counts: Mapping[str, int],
    end_reason_counts: Mapping[str, int],
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "collector": "v4_big_hand",
        "big_hand_sidecar_format_version": BIG_HAND_SIDECAR_FORMAT_VERSION,
        "v4_demonstration_format_version": V4_DEMONSTRATION_FORMAT_VERSION,
        "v4_tensor_version": V4_TENSOR_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "round_count": round_count,
        "seed_start": seed_start,
        "seed_end": seed_start + round_count - 1,
        "agent_seed": agent_seed,
        "teacher_seed": teacher_seed,
        "big_hand_config": {
            "kan_probability": kan_probability,
            "seven_pairs_max_time_penalty": seven_pairs_max_time_penalty,
            "dadui_switch_margin": dadui_switch_margin,
        },
        "samples_per_shard": samples_per_shard,
        "compressed": compressed,
        "decision_count": decision_count,
        "skipped_decision_count": skipped_decision_count,
        "filtered_kan_decision_count": filtered_kan_decision_count,
        "sample_count": sample_count,
        "category_counts": {
            name: int(category_counts.get(name, 0))
            for name in V4_CATEGORY_NAMES
        },
        "action_counts": {
            name: int(value)
            for name, value in sorted(action_counts.items())
        },
        "end_reason_counts": {
            name: int(value)
            for name, value in sorted(end_reason_counts.items())
        },
        "sidecar": BIG_HAND_SIDECAR_NAME,
        "audit_json": BIG_HAND_AUDIT_JSON,
        "audit_markdown": BIG_HAND_AUDIT_MARKDOWN,
        "elapsed_seconds": elapsed_seconds,
    }


def _audit_markdown(audit: Mapping[str, object]) -> str:
    lines = [
        "# V4 BigHand 数据审计",
        "",
        f"- 牌局数：{audit['round_count']}",
        f"- 已规划玩家局：{audit['planned_player_rounds']}",
        f"- 未进入决策而未规划：{audit['unplanned_player_rounds']}",
        "",
        "## 路线结果",
        "",
        "| 路线 | 玩家局 | 完成目标 | 完成率 | 胡牌率 | 平均净分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        BigHandRoute.SEVEN_PAIRS.value: "七对",
        BigHandRoute.ALL_TRIPLETS.value: "大对子",
        BigHandRoute.FLUSH.value: "清一色",
    }
    for name, values in audit["routes"].items():
        lines.append(
            f"| {labels[name]} | {values['player_rounds']} | "
            f"{values['target_completed']} | "
            f"{values['target_completion_rate']:.3%} | "
            f"{values['hu_rate']:.3%} | "
            f"{values['average_net_score']:.3f} |"
        )
    teacher = audit["teacher_comparison"]
    pon = audit["pon"]
    sacrifice = audit["normal_shanten_sacrifice"]
    decisions = audit["decisions"]
    lines.extend([
        "",
        "## 决策差异",
        "",
        f"- 总决策：{decisions['total']}",
        f"- 写入监督样本：{decisions['samples_written']}",
        f"- 因选择杠而过滤：{decisions['kan_filtered']}",
        f"- 与 Teacher 精确动作不同："
        f"{teacher['exact_difference_count']} "
        f"({teacher['exact_difference_rate']:.3%})",
        f"- 不在 Teacher 目标集合："
        f"{teacher['target_disagreement_count']} "
        f"({teacher['target_disagreement_rate']:.3%})",
        f"- 有多个合法动作的策略决策：{teacher['strategic_decisions']}",
        f"- 策略决策与 Teacher 精确动作不同："
        f"{teacher['strategic_exact_difference_count']} "
        f"({teacher['strategic_exact_difference_rate']:.3%})",
        f"- 策略决策不在 Teacher 目标集合："
        f"{teacher['strategic_target_disagreement_count']} "
        f"({teacher['strategic_target_disagreement_rate']:.3%})",
        "",
        "## 碰牌与普通向听",
        "",
        f"- 碰/Pass 机会：{pon['opportunity_count']}",
        f"- 选择碰：{pon['selected_count']} "
        f"({pon['selection_rate']:.3%})",
        f"- 选择 Pass：{pon['pass_count']}",
        f"- 可比较的弃牌决策："
        f"{sacrifice['comparable_discard_decisions']}",
        f"- 放弃普通型最小向听：{sacrifice['sacrifice_count']} "
        f"({sacrifice['sacrifice_rate']:.3%})",
        f"- 平均普通向听差：{sacrifice['average_gap']:.4f}",
        "",
        "## 动作计数",
        "",
    ])
    for name, count in decisions["action_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    return "\n".join(lines)


def _collect_worker(arguments: tuple[object, ...]) -> dict[str, object]:
    (
        output,
        round_count,
        seed_start,
        agent_seed,
        teacher_seed,
        samples_per_shard,
        compressed,
        kan_probability,
        seven_pairs_max_time_penalty,
        dadui_switch_margin,
    ) = arguments
    return collect_v4_big_hand_demonstrations(
        Path(output),
        round_count=int(round_count),
        seed_start=int(seed_start),
        agent_seed=int(agent_seed),
        teacher_seed=int(teacher_seed),
        samples_per_shard=int(samples_per_shard),
        compressed=bool(compressed),
        progress_interval=0,
        kan_probability=float(kan_probability),
        seven_pairs_max_time_penalty=float(seven_pairs_max_time_penalty),
        dadui_switch_margin=float(dadui_switch_margin),
    )


def _read_sidecar(path: Path) -> Iterable[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid sidecar JSON at line {line_number}"
                    ) from error


def _sum_nested_counts(
    reports: Sequence[Mapping[str, object]],
    field: str,
    names: Iterable[str],
) -> dict[str, int]:
    return {
        name: sum(int(report[field].get(name, 0)) for report in reports)
        for name in names
    }


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value).rsplit(".", 1)[-1]


def _validate_collection_arguments(
    round_count: int,
    samples_per_shard: int,
    progress_interval: int,
) -> None:
    if round_count <= 0:
        raise ValueError("round count must be positive")
    if samples_per_shard <= 0:
        raise ValueError("samples per shard must be positive")
    if progress_interval < 0:
        raise ValueError("progress interval cannot be negative")


def _ensure_empty_destination(destination: Path) -> None:
    if not destination.exists():
        return
    occupied = (
        tuple(destination.glob(f"{V4_DEMONSTRATION_PREFIX}_*.npz"))
        or (destination / "manifest.json").exists()
        or (destination / BIG_HAND_SIDECAR_NAME).exists()
    )
    if occupied:
        raise FileExistsError(
            f"V4 BigHand output already exists in {destination}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and audit V4 BigHand demonstrations."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--agent-seed", type=int, default=20260928)
    parser.add_argument("--teacher-seed", type=int, default=20260929)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--kan-probability", type=float, default=0.5)
    parser.add_argument(
        "--seven-pairs-max-time-penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument("--dadui-switch-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    common = {
        "round_count": args.rounds,
        "seed_start": args.seed_start,
        "agent_seed": args.agent_seed,
        "teacher_seed": args.teacher_seed,
        "samples_per_shard": args.samples_per_shard,
        "compressed": not args.no_compression,
        "kan_probability": args.kan_probability,
        "seven_pairs_max_time_penalty": (
            args.seven_pairs_max_time_penalty
        ),
        "dadui_switch_margin": args.dadui_switch_margin,
    }
    if args.workers == 1:
        report = collect_v4_big_hand_demonstrations(
            args.output,
            progress_interval=args.progress_interval,
            **common,
        )
    else:
        report = collect_v4_big_hand_demonstrations_parallel(
            args.output,
            worker_count=args.workers,
            **common,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
