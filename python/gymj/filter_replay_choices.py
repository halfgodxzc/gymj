"""Select bounded public discard comparisons for a new replay experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import shutil

from . import _gymj
from .chongfeng import FIXED_CHICKEN_TILE_IDS
from . import filter_effective_draw_choices as legacy


REPLAY_SCREEN_VERSION = 1
DEFAULT_MAX_PER_EPISODE = 4
DEFAULT_MAX_DECISIONS = 20_000
DEFAULT_SELECTION_SEED = 2026101407
FAMILIES = ("more_effective_draws", "chicken_vs_non_chicken")


def chicken_opportunity(observation, tracker, player_id: int, tile_id: int) -> dict:
    """Describe necessary public call conditions, never opponents' actual hands."""
    if not 0 <= player_id < 4 or not 0 <= tile_id < legacy.TILE_TYPE_COUNT:
        raise ValueError("invalid player or discard tile id")
    fixed = tile_id in FIXED_CHICKEN_TILE_IDS
    global_first = fixed and tile_id not in tracker.seen_fixed_chickens
    opening_follow = (fixed and not global_first
                      and not tracker.player_has_discarded[player_id])
    kind = ("global_first" if global_first else "opening_follow"
            if opening_follow else "ordinary_fixed" if fixed else "none")
    unseen = 4 - int(observation.visible_counts[tile_id])
    if not 0 <= unseen <= 4:
        raise ValueError("visible tile count must be in range 0..4")
    responders = [relative for relative in range(1, 4)
                  if int(observation.ready_kinds[relative]) == 0]
    pon_possible = fixed and unseen >= 2 and bool(responders)
    minkan_possible = (fixed and unseen >= 3 and bool(responders)
                       and int(observation.wall_remaining) > 0)
    return {
        "fixed_chicken": fixed,
        "kind": kind,
        "chongfeng_opportunity": global_first or opening_follow,
        "global_first": global_first,
        "opening_follow": opening_follow,
        "unseen_copies": unseen,
        "public_pon_possible": pon_possible,
        "public_minkan_possible": minkan_possible,
        "possible_responder_relative_ids": responders if pon_possible or minkan_possible else [],
        "responsibility_possible": (global_first or opening_follow)
        and (pon_possible or minkan_possible),
        "actual_opponent_can_call": None,
    }


def screen_key_discards(candidates, selected_id, observation, tracker, player_id):
    by_id = {item["action_id"]: item for item in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate discard tile type")
    if selected_id not in by_id:
        raise ValueError("selected discard is absent from candidates")
    if any(not 0 <= tile_id < legacy.TILE_TYPE_COUNT for tile_id in by_id):
        raise ValueError("screening accepts discard actions only")
    enriched = [{**item, "chicken": chicken_opportunity(
        observation, tracker, player_id, item["action_id"],
    )} for item in sorted(candidates, key=lambda item: item["action_id"])]
    by_id = {item["action_id"]: item for item in enriched}
    selected = by_id[selected_id]
    alternatives = []
    for other in enriched:
        if other["action_id"] == selected_id or other["shanten"] != selected["shanten"]:
            continue
        routes = selected["closest_routes"]
        deltas = {route: {
            metric: other["effective_draws"][route][metric]
            - selected["effective_draws"][route][metric]
            for metric in ("kinds", "copies")
        } for route in legacy.ROUTES}
        labels = []
        # When closest routes tie, retain only Pareto improvements in copies.
        if (all(deltas[route]["copies"] >= 0 for route in routes)
                and any(deltas[route]["copies"] > 0 for route in routes)):
            labels.append(FAMILIES[0])
        chicken_pair = (selected["chicken"]["fixed_chicken"]
                        != other["chicken"]["fixed_chicken"])
        chicken_item = selected if selected["chicken"]["fixed_chicken"] else other
        if chicken_pair and chicken_item["chicken"]["chongfeng_opportunity"]:
            labels.append(FAMILIES[1])
        if labels:
            alternatives.append({
                "action_id": other["action_id"], "families": labels,
                "effective_draw_deltas": deltas,
                "selected_chicken": selected["chicken"],
                "alternative_chicken": other["chicken"],
            })
    if not alternatives:
        return None
    return {
        "replay_screen_version": REPLAY_SCREEN_VERSION,
        "selected_action_id": selected_id,
        "families": [label for label in FAMILIES
                     if any(label in pair["families"] for pair in alternatives)],
        "candidates": enriched, "alternatives": alternatives,
        "same_shanten_definition": "identical normal/seven_pairs/all_triplets vector",
    }


def screen_key_episode(sidecar):
    cases, counts = legacy.screen_episode(sidecar, discard_screener=screen_key_discards)
    for case in cases:
        record = sidecar["decisions"][case["decision_index"]]
        case["source_prediction"] = {
            key: record[key] for key in ("policy_logits", "old_log_probability", "entropy")
            if key in record
        }
    return cases, counts


def _priority(case, selection_seed):
    payload = f"{selection_seed}:{case['episode_seed']}:{case['dealer']}:{case['decision_index']}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest(), "big")


def cap_episode_cases(cases, max_per_episode=DEFAULT_MAX_PER_EPISODE,
                      selection_seed=DEFAULT_SELECTION_SEED):
    if max_per_episode <= 0:
        raise ValueError("max per episode must be positive")
    ordered = sorted(cases, key=lambda case: (_priority(case, selection_seed), case["decision_index"]))
    selected = {}
    # Reserve one opportunity per family where the episode budget permits.
    for family in FAMILIES:
        if len(selected) >= max_per_episode:
            break
        if any(family in case["families"] for case in selected.values()):
            continue
        first = next((case for case in ordered if family in case["families"]), None)
        if first is not None:
            selected[first["decision_index"]] = first
    for case in ordered:
        if len(selected) >= max_per_episode:
            break
        selected[case["decision_index"]] = case
    return sorted(selected.values(), key=lambda case: case["decision_index"])


def filter_replay_sidecars(inputs, output, *, workers=1,
                          max_per_episode=DEFAULT_MAX_PER_EPISODE,
                          max_decisions=DEFAULT_MAX_DECISIONS,
                          selection_seed=DEFAULT_SELECTION_SEED):
    if min(workers, max_per_episode, max_decisions) <= 0:
        raise ValueError("workers and decision limits must be positive")
    paths = [Path(path).resolve() for path in inputs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide distinct input sidecar files")
    sources = [{"path": str(path), "sha256": legacy._sha256(path)} for path in paths]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dependencies = (Path(__file__), Path(legacy.__file__),
                    *(Path(__file__).with_name(name) for name in (
                        "chongfeng.py", "diagnostic_dataset.py", "trajectory.py",
                    )))
    manifest = {
        "replay_screen_version": REPLAY_SCREEN_VERSION, "status": "running",
        "sources": sources, "workers": workers,
        "max_per_episode": max_per_episode, "max_decisions": max_decisions,
        "selection_seed": selection_seed, "continuations_run": 0,
        "extension_sha256": legacy._sha256(Path(_gymj.__file__)),
        "source_hashes": {str(path): legacy._sha256(path) for path in dependencies},
        "action_encoding_version": legacy.ACTION_ENCODING_VERSION,
        "observation_version": legacy.OBSERVATION_VERSION,
        "decision_feature_version": legacy.DECISION_FEATURE_VERSION,
        "pattern_feature_version": legacy.PATTERN_FEATURE_VERSION,
        "same_shanten_definition": "identical normal/seven_pairs/all_triplets vector",
        "selection_rule": "public features only; per-wall family reservations then seeded hash; global seeded hash",
        "responsibility_note": "public possibility only; never read opponents' concealed hands",
        "executor_compatibility": "requires new two-family executor; not legacy SCREEN_VERSION 1",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total, eligible_families, selected_families, selected_players = (Counter() for _ in range(4))
    chicken_kinds = Counter()
    seen_seeds, heap = set(), []
    try:
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for path in dependencies:
            shutil.copy2(path, snapshot / path.name)
        with gzip.open(output / "source_episodes.jsonl.gz", "wt", encoding="utf-8") as episodes, \
                gzip.open(output / "eligible_cases.jsonl.gz", "wt", encoding="utf-8") as eligible:
            for path, line_number, episode, (cases, counts) in legacy._screen_sources(
                    paths, workers, episode_screener=screen_key_episode):
                wall_seed = int(episode["episode_seed"])
                if wall_seed in seen_seeds:
                    raise ValueError(f"duplicate episode seed (including seat variants): {wall_seed}")
                seen_seeds.add(wall_seed)
                episode_index = total["episodes"]
                total.update(counts)
                episodes.write(json.dumps(episode, separators=(",", ":")) + "\n")
                total["eligible_before_episode_cap"] += len(cases)
                retained = cap_episode_cases(cases, max_per_episode, selection_seed)
                for case in retained:
                    case.update(source_episode_index=episode_index, source_file=str(path),
                                source_line=line_number)
                    eligible.write(json.dumps(case, separators=(",", ":")) + "\n")
                    eligible_families.update(case["families"])
                    # Store IDs only, not thousands of large feature dictionaries.
                    item = (-_priority(case, selection_seed), -wall_seed,
                            -case["decision_index"], episode_index, case["decision_index"])
                    if len(heap) < max_decisions:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                total["eligible_after_episode_cap"] += len(retained)
                if total["episodes"] == 1 or total["episodes"] % 100 == 0:
                    print(f"Screened {total['episodes']} episodes; "
                          f"{total['eligible_after_episode_cap']} capped decisions", flush=True)
        selected_ids = {(item[3], item[4]) for item in heap}
        with gzip.open(output / "candidates.jsonl.gz", "wt", encoding="utf-8") as selected:
            for _, case in legacy._read_episodes(output / "eligible_cases.jsonl.gz"):
                if (case["source_episode_index"], case["decision_index"]) not in selected_ids:
                    continue
                selected.write(json.dumps(case, separators=(",", ":")) + "\n")
                total["selected_decisions"] += 1
                total["selected_alternative_pairs"] += len(case["alternatives"])
                selected_families.update(case["families"])
                selected_players[str(case["player_id"])] += 1
                total["selected_overlap_decisions"] += len(case["families"]) > 1
                for pair in case["alternatives"]:
                    if FAMILIES[1] in pair["families"]:
                        chicken = (pair["selected_chicken"]
                                   if pair["selected_chicken"]["fixed_chicken"]
                                   else pair["alternative_chicken"])
                        chicken_kinds[chicken["kind"]] += 1
                        total["chicken_pairs_with_public_responsibility_risk"] += chicken["responsibility_possible"]
        if total["selected_decisions"] != len(heap):
            raise ValueError("selected decision count mismatch")
        for item in sources:
            if legacy._sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("input changed during screening")
        for path in dependencies:
            if legacy._sha256(path) != manifest["source_hashes"][str(path)]:
                raise ValueError("screening source changed during run")
        if legacy._sha256(Path(_gymj.__file__)) != manifest["extension_sha256"]:
            raise ValueError("native extension changed during screening")
        summary = {
            "counts": {key: total[key] for key in (
                "episodes", "decisions", "forced_decisions", "selected_discards",
                "multi_discard_decisions", "eligible_before_episode_cap",
                "eligible_after_episode_cap", "selected_decisions", "selected_alternative_pairs",
                "selected_overlap_decisions", "chicken_pairs_with_public_responsibility_risk",
            )},
            "eligible_decisions_by_family": {key: eligible_families[key] for key in FAMILIES},
            "selected_decisions_by_family": {key: selected_families[key] for key in FAMILIES},
            "selected_decisions_by_player": {str(seat): selected_players[str(seat)] for seat in range(4)},
            "selected_chicken_pairs_by_kind": dict(chicken_kinds),
            "family_counts_overlap": True,
            "continuations_run": 0,
            "statistical_unit": "episode_seed; derived decisions are not independent games",
            "estimated_continuations_if_all_pairs_used": total["selected_decisions"]
            + total["selected_alternative_pairs"],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        manifest["outputs"] = {name: legacy._sha256(output / name) for name in (
            "source_episodes.jsonl.gz", "eligible_cases.jsonl.gz", "candidates.jsonl.gz", "summary.json",
        )}
        return summary
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-per-episode", type=int, default=DEFAULT_MAX_PER_EPISODE)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args()
    print(json.dumps(filter_replay_sidecars(
        args.input, args.output, workers=args.workers,
        max_per_episode=args.max_per_episode, max_decisions=args.max_decisions,
        selection_seed=args.selection_seed,
    ), indent=2))


if __name__ == "__main__":
    main()
