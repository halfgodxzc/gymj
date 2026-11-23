"""Screen complete decision sidecars; never continue alternative branches."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from . import _gymj
from ._gymj import (
    ACTION_ENCODING_VERSION, DECISION_FEATURE_VERSION, OBSERVATION_VERSION,
    PATTERN_FEATURE_VERSION, TILE_TYPE_COUNT, ActionEncoder, ActionType, Arena,
    DecisionFeatureEncoder, ObservationEncoder,
)
from .chongfeng import ChongfengTracker
from .diagnostic_dataset import DIAGNOSTIC_SIDECAR_VERSION, observation_fingerprint
from .trajectory import pack_legal_mask


SCREEN_VERSION = 1
ROUTES = ("normal", "seven_pairs", "all_triplets")
GROUPS = tuple(f"strict_{route}" for route in ROUTES) + (
    "tied_closest_routes", "different_route_vector", "non_closest_route_draws",
)


def _draw_record(draws) -> dict:
    return {
        "kinds": int(draws.kinds),
        "copies": int(draws.copies),
        "remaining_by_tile": list(map(int, draws.remaining_by_tile)),
    }


def discard_record(features, action_id: int, player_id: int, chicken) -> dict:
    """Only public candidate features participate in screening."""
    candidate = features.actions[action_id]
    if not candidate.legal or not candidate.post_hand_known:
        raise ValueError("discard candidate has no known legal post-hand")
    hand = candidate.post_hand
    shanten = [int(route.shanten) for route in hand.routes]
    best = min(shanten)
    return {
        "action_id": action_id,
        "shanten": shanten,
        "best_shanten": best,
        "closest_routes": [name for name, value in zip(ROUTES, shanten)
                           if value == best],
        "effective_draws": {
            name: _draw_record(route.effective_draws)
            for name, route in zip(ROUTES, hand.routes)
        },
        "flushes": [{
            "shanten": int(flush.shanten),
            "off_suit_tiles": int(flush.off_suit_tiles),
            "meld_compatible": bool(flush.meld_compatible),
            "effective_draws": _draw_record(flush.effective_draws),
        } for flush in hand.flushes],
        "shape_wait_mask": int(hand.shape_wait_mask),
        "live_wait_mask": int(hand.live_wait_mask),
        "wait_values": [{
            "tile_id": tile_id,
            "pattern_mask": int(value.pattern_mask),
            "base_fan": int(value.base_fan),
            "base_hu_points": int(value.base_hu_points),
            "remaining_copies": int(value.remaining_copies),
        } for tile_id, value in enumerate(hand.wait_values)
           if int(hand.shape_wait_mask) & (1 << tile_id)],
        "chongfeng_opportunity": chicken.can_chongfeng(player_id, action_id),
    }


def compare_discards(selected: Mapping, alternative: Mapping) -> dict | None:
    if selected["action_id"] == alternative["action_id"]:
        return None
    if selected["best_shanten"] != alternative["best_shanten"]:
        return None
    relevant = set(selected["closest_routes"]) | set(alternative["closest_routes"])
    # Routes are never summed: their improving tile sets can overlap.
    deltas = {
        route: {
            metric: alternative["effective_draws"][route][metric]
                    - selected["effective_draws"][route][metric]
            for metric in ("kinds", "copies")
        } for route in ROUTES
    }
    if not any(deltas[route][metric] for route in ROUTES
               for metric in ("kinds", "copies")):
        return None
    closest_draws_differ = any(deltas[route][metric] for route in relevant
                              for metric in ("kinds", "copies"))
    same_vector = selected["shanten"] == alternative["shanten"]
    if not closest_draws_differ:
        group = "non_closest_route_draws"
    elif same_vector and len(selected["closest_routes"]) == 1:
        group = "strict_" + selected["closest_routes"][0]
    elif same_vector:
        group = "tied_closest_routes"
    else:
        group = "different_route_vector"
    return {
        "action_id": alternative["action_id"],
        "group": group,
        "effective_draw_deltas": deltas,
    }


def screen_discards(candidates: Sequence[dict], selected_id: int) -> dict | None:
    by_id = {item["action_id"]: item for item in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate discard tile type")
    if selected_id not in by_id:
        raise ValueError("selected discard is absent from candidates")
    selected = by_id[selected_id]
    alternatives = [pair for item in candidates
                    if (pair := compare_discards(selected, item)) is not None]
    if not alternatives:
        return None
    primary = None
    if len(selected["closest_routes"]) == 1:
        route = selected["closest_routes"][0]
        comparable = [item for item in candidates
                      if item["shanten"] == selected["shanten"]]
        # Preselect using copies only, before seeing continuation outcomes.
        # A copies tie retains the actual action, then uses lowest tile ID.
        best = max(comparable, key=lambda item: (
            item["effective_draws"][route]["copies"],
            item["action_id"] == selected_id, -item["action_id"],
        ))
        primary = {
            "route": route,
            "action_id": best["action_id"],
            "copies_delta": best["effective_draws"][route]["copies"]
                            - selected["effective_draws"][route]["copies"],
            "changes_action": best["action_id"] != selected_id,
        }
    return {
        "selected_action_id": selected_id,
        "candidates": list(candidates),
        "alternatives": alternatives,
        "primary_max_copies": primary,
    }


def screen_episode(sidecar: Mapping, *, discard_screener=None) -> tuple[list[dict], dict]:
    """Replay the original episode once and validate it before returning cases."""
    if sidecar.get("diagnostic_sidecar_version") != DIAGNOSTIC_SIDECAR_VERSION:
        raise ValueError("unsupported diagnostic sidecar version")
    decisions = sidecar["decisions"]
    if not decisions or sidecar["decision_count"] != len(decisions):
        raise ValueError("sidecar must contain a complete nonempty decision list")
    seed, dealer = int(sidecar["episode_seed"]), int(sidecar["dealer"])
    arena = Arena(seed=seed, dealer=dealer)
    arena.start_round()
    chicken = ChongfengTracker()
    cases = []
    counts = Counter()
    for index, record in enumerate(decisions):
        if arena.finished or record["decision_index"] != index:
            raise ValueError("sidecar decisions are not contiguous or ended early")
        decision = arena.current_decision()
        player_id = int(decision.player)
        if player_id != record["player_id"]:
            raise ValueError("replayed decision player differs from sidecar")
        encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)
        legal_ids = list(map(int, encoded.legal_action_ids))
        if sorted(record["legal_action_ids"]) != sorted(legal_ids):
            raise ValueError("replayed legal actions differ from sidecar")
        observation = ObservationEncoder.encode(arena, player_id)
        fingerprint = observation_fingerprint(
            observation, pack_legal_mask(encoded.legal_mask),
        )
        if fingerprint != record["observation_sha256"]:
            raise ValueError("replayed observation fingerprint differs from sidecar")
        action_id = int(record["action_id"])
        if action_id not in legal_ids:
            raise ValueError("recorded action is not legal")
        action = encoded.action_at(action_id)
        expected = {"type": int(action.type), "actor": int(action.actor),
                    "tile": int(action.tile.id), "target": int(action.target)}
        if any(record["action"][key] != value for key, value in expected.items()):
            raise ValueError("recorded action semantics differ from decoded action")
        counts["decisions"] += 1
        if len(legal_ids) == 1:
            counts["forced_decisions"] += 1
        if action.type == ActionType.DISCARD:
            counts["selected_discards"] += 1
            discards = sorted(tile for tile in legal_ids if tile < TILE_TYPE_COUNT)
            if len(discards) > 1:
                counts["multi_discard_decisions"] += 1
                features = DecisionFeatureEncoder.encode(arena, player_id)
                candidates = [discard_record(features, tile, player_id, chicken)
                              for tile in discards]
                case = (screen_discards(candidates, action_id)
                        if discard_screener is None else discard_screener(
                            candidates, action_id, observation, chicken, player_id,
                        ))
                if case is not None:
                    case.update({
                        "episode_seed": seed, "dealer": dealer,
                        "decision_index": index, "player_id": player_id,
                        "observation_sha256": fingerprint,
                        "legal_action_ids": legal_ids,
                        "hand_counts": list(map(int, observation.hand_counts)),
                        "visible_counts": list(map(int, observation.visible_counts)),
                        "own_meld_count": len(arena.player(player_id).melds),
                        "wall_remaining": int(observation.wall_remaining),
                        "phase": arena.phase.name,
                        "ready_kinds_relative": list(map(int, observation.ready_kinds)),
                        "must_discard_after_call": bool(features.progress.must_discard_after_call),
                        "last_draw_kind": features.progress.last_draw_kind.name,
                        "policy_sha256": sidecar.get("policy_sha256"),
                        "sampling": sidecar.get("sampling"),
                    })
                    cases.append(case)
            chicken.observe_discard(player_id, int(action.tile.id))
        arena.submit_action(action)
        if bool(arena.finished) != (index == len(decisions) - 1):
            raise ValueError("replayed termination differs from sidecar")
    scores = list(map(int, arena.round_settlement.ledger.score_deltas))
    if scores != sidecar["terminal"]["score_deltas"] or sum(scores) != 0:
        raise ValueError("replayed terminal scores differ from sidecar")
    counts["episodes"] = 1
    counts["screened_decisions"] = len(cases)
    return cases, dict(counts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_episodes(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def _screen_sources(paths, workers, episode_screener=None):
    if episode_screener is None:
        episode_screener = screen_episode
    def records():
        for path in paths:
            for line_number, episode in _read_episodes(path):
                yield path, line_number, episode

    if workers == 1:
        for path, line_number, episode in records():
            yield path, line_number, episode, episode_screener(episode)
        return
    # Bound both source episodes and large candidate results held in memory.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        source = iter(records())
        pending = deque()
        for _ in range(workers * 2):
            item = next(source, None)
            if item is None:
                break
            pending.append((item, pool.submit(episode_screener, item[2])))
        while pending:
            (path, line_number, episode), future = pending.popleft()
            yield path, line_number, episode, future.result()
            item = next(source, None)
            if item is not None:
                pending.append((item, pool.submit(episode_screener, item[2])))


def filter_sidecars(inputs: Sequence[Path], output: Path, workers: int = 1) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    paths = [Path(path).resolve() for path in inputs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide distinct input sidecar files")
    sources = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "screen_version": SCREEN_VERSION, "status": "running",
        "workers": workers,
        "sources": sources,
        "extension_sha256": _sha256(Path(_gymj.__file__)),
        "script_sha256": _sha256(Path(__file__)),
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "pattern_feature_version": PATTERN_FEATURE_VERSION,
        "effective_draw_definition": "one-draw route shanten reduction; 4-visible copies",
        "primary_rule": "same shanten vector, unique closest route, max copies; retain selected on tie",
        "continuations_run": 0,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total, groups, player_counts = Counter(), Counter(), Counter()
    seen = set()
    try:
        with gzip.open(output / "source_episodes.jsonl.gz", "wt", encoding="utf-8") as episodes, \
                gzip.open(output / "candidates.jsonl.gz", "wt", encoding="utf-8") as candidates:
            for path, line_number, episode, (cases, counts) in _screen_sources(paths, workers):
                key = (int(episode["episode_seed"]), int(episode["dealer"]))
                if key in seen:
                    raise ValueError(f"duplicate episode seed/dealer: {key}")
                seen.add(key)
                episode_index = total["episodes"]
                total.update(counts)
                episodes.write(json.dumps(episode, separators=(",", ":")) + "\n")
                for case in cases:
                    case["source_episode_index"] = episode_index
                    case["source_file"] = str(path)
                    case["source_line"] = line_number
                    candidates.write(json.dumps(case, separators=(",", ":")) + "\n")
                    player_counts[str(case["player_id"])] += 1
                    for pair in case["alternatives"]:
                        groups[pair["group"]] += 1
                        total["alternative_branches"] += 1
                    primary = case["primary_max_copies"]
                    if primary and primary["changes_action"]:
                        total["primary_more_copies_decisions"] += 1
                if total["episodes"] == 1 or total["episodes"] % 100 == 0:
                    print(f"Screened {total['episodes']} episodes; "
                          f"{total['screened_decisions']} decisions; "
                          f"{total['alternative_branches']} alternatives", flush=True)
        if any(_sha256(Path(item["path"])) != item["sha256"] for item in sources):
            raise ValueError("input changed during screening")
        summary = {
            "counts": {key: total[key] for key in (
                "episodes", "decisions", "forced_decisions", "selected_discards",
                "multi_discard_decisions", "screened_decisions", "alternative_branches",
                "primary_more_copies_decisions",
            )},
            "alternative_pairs_by_group": {group: groups[group] for group in GROUPS},
            "screened_decisions_by_player": {str(seat): player_counts[str(seat)] for seat in range(4)},
            "estimated_greedy_continuations": total["screened_decisions"] + total["alternative_branches"],
            "statistical_unit": "episode_seed; do not treat same-round decisions as independent",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        manifest["outputs"] = {
            name: _sha256(output / name)
            for name in ("candidates.jsonl.gz", "source_episodes.jsonl.gz", "summary.json")
        }
        return summary
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Complete diagnostic episode JSONL(.gz); repeat for shards")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing directories are never overwritten")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel original-episode screening workers; preserves input order")
    args = parser.parse_args()
    print(json.dumps(filter_sidecars(args.input, args.output, args.workers), indent=2))


if __name__ == "__main__":
    main()
