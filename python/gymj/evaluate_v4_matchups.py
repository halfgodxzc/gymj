from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

from .chongfeng import ChongfengTracker
from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    BA_TONG_TILE_ID,
    DECISION_FEATURE_VERSION,
    OBSERVATION_VERSION,
    TILE_TYPE_COUNT,
    YAO_JI_TILE_ID,
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    ObservationEncoder,
)
from .env import EnvState
from .settlement_components import (
    COMPONENT_NAMES,
    settlement_component_deltas,
)
from .teacher import (
    ANKAN_ACTION_START,
    DAIMINKAN_ACTION_ID,
    DECLARE_READY_ACTION_ID,
    KAKAN_ACTION_START,
    PASS_ACTION_ID,
    PON_ACTION_ID,
    RON_ACTION_ID,
    TSUMO_ACTION_ID,
    ShantenTeacher,
)
from .v4_agent import V4Agent


V4_MATCHUP_EVALUATION_VERSION = 2
V4_EVALUATION_PROTOCOL_VERSION = 1
V4_EVALUATION_GAMES_PER_MATCHUP = 1_000
V4_EVALUATION_SEED_START = 932_000_001
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 932_000_999
SEAT_PAIR_ORDER = (
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (1, 2),
)
V4_MATCHUPS = ("baseline_v4", "teacher")


@dataclass(frozen=True)
class EvaluationCase:
    game_index: int
    wall_index: int
    seed: int
    dealer: int
    candidate_seats: Tuple[int, int]


def build_evaluation_schedule(
    games: int = V4_EVALUATION_GAMES_PER_MATCHUP,
    seed_start: int = V4_EVALUATION_SEED_START,
) -> tuple[EvaluationCase, ...]:
    if games <= 0:
        raise ValueError("evaluation game count must be positive")
    if seed_start < 0:
        raise ValueError("evaluation seed must be non-negative")
    pair_count = len(SEAT_PAIR_ORDER)
    return tuple(
        EvaluationCase(
            game_index=game_index,
            wall_index=game_index // pair_count,
            seed=seed_start + game_index // pair_count,
            dealer=(game_index // pair_count) % 4,
            candidate_seats=SEAT_PAIR_ORDER[game_index % pair_count],
        )
        for game_index in range(games)
    )


def protocol_manifest(
    games: int = V4_EVALUATION_GAMES_PER_MATCHUP,
    seed_start: int = V4_EVALUATION_SEED_START,
    teacher_games: int | None = None,
) -> dict[str, object]:
    if teacher_games is None:
        teacher_games = games
    if teacher_games <= 0:
        raise ValueError("teacher game count must be positive")
    schedule = build_evaluation_schedule(games, seed_start)
    teacher_schedule = build_evaluation_schedule(teacher_games, seed_start)
    seat_counts = [0, 0, 0, 0]
    pair_counts = {
        f"{left}-{right}": 0
        for left, right in itertools.combinations(range(4), 2)
    }
    for case in schedule:
        for seat in case.candidate_seats:
            seat_counts[seat] += 1
        pair_counts[
            f"{case.candidate_seats[0]}-{case.candidate_seats[1]}"
        ] += 1
    schedule_json = json.dumps(
        [asdict(case) for case in schedule],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "protocol_version": V4_EVALUATION_PROTOCOL_VERSION,
        "games_per_matchup": games,
        "matchup_game_counts": {
            "baseline_v4": games,
            "teacher": teacher_games,
        },
        "seed_start": seed_start,
        "seed_end": schedule[-1].seed,
        "unique_walls": len({case.seed for case in schedule}),
        "matchup_unique_walls": {
            "baseline_v4": len({case.seed for case in schedule}),
            "teacher": len({case.seed for case in teacher_schedule}),
        },
        "candidate_seat_counts": seat_counts,
        "candidate_pair_counts": pair_counts,
        "seat_pair_order": [list(pair) for pair in SEAT_PAIR_ORDER],
        "schedule_sha256": hashlib.sha256(schedule_json).hexdigest().upper(),
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "matchups": [
            "two_candidate_two_baseline_v4",
            "two_candidate_two_teacher",
        ],
        "teacher_kan_probability": 0.0,
        "teacher_ready_policy": "declare",
        "inference": "deterministic_argmax",
        "big_hand_definition": "winning settlement total_fan > 1",
        "score_ci": (
            "95% wall-cluster bootstrap with "
            f"{BOOTSTRAP_RESAMPLES} resamples"
        ),
    }


def evaluate_v4_matchups(
    candidate_checkpoint: Path | str,
    baseline_checkpoint: Path | str,
    *,
    games_per_matchup: int = V4_EVALUATION_GAMES_PER_MATCHUP,
    teacher_games: int | None = None,
    seed_start: int = V4_EVALUATION_SEED_START,
    workers: int = 8,
) -> dict[str, object]:
    if games_per_matchup <= 0:
        raise ValueError("games per matchup must be positive")
    if teacher_games is None:
        teacher_games = games_per_matchup
    if teacher_games <= 0:
        raise ValueError("teacher game count must be positive")
    if seed_start < 0:
        raise ValueError("seed start must be non-negative")
    if workers <= 0:
        raise ValueError("worker count must be positive")
    candidate_path = Path(candidate_checkpoint).resolve()
    baseline_path = Path(baseline_checkpoint).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    schedules = {
        "baseline_v4": build_evaluation_schedule(
            games_per_matchup,
            seed_start,
        ),
        "teacher": build_evaluation_schedule(teacher_games, seed_start),
    }
    work_items = tuple(
        (opponent, case)
        for opponent in V4_MATCHUPS
        for case in schedules[opponent]
    )
    worker_count = min(workers, len(work_items))
    assignments = [
        work_items[index::worker_count]
        for index in range(worker_count)
    ]
    arguments = [
        (assignment, str(candidate_path), str(baseline_path))
        for assignment in assignments
    ]
    started = time.perf_counter()
    if worker_count == 1:
        pieces = [_evaluate_worker(*arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pieces = list(executor.map(_evaluate_worker_arguments, arguments))
    rounds = sorted(
        (round_ for piece in pieces for round_ in piece),
        key=lambda item: (
            V4_MATCHUPS.index(item["opponent"]),
            item["game_index"],
        ),
    )

    grouped = {
        opponent: [
            round_ for round_ in rounds
            if round_["opponent"] == opponent
        ]
        for opponent in V4_MATCHUPS
    }
    for opponent, opponent_rounds in grouped.items():
        schedule = schedules[opponent]
        if len(opponent_rounds) != len(schedule):
            raise RuntimeError(
                f"{opponent} returned an unexpected round count"
            )
        _validate_rounds(opponent_rounds, schedule)

    results = {
        opponent: _summarize_matchup(grouped[opponent], opponent)
        for opponent in V4_MATCHUPS
    }
    candidate_identity = _file_identity(candidate_path)
    baseline_identity = _file_identity(baseline_path)
    protocol = protocol_manifest(
        games_per_matchup,
        seed_start,
        teacher_games,
    )
    return {
        "evaluation_format_version": V4_MATCHUP_EVALUATION_VERSION,
        "protocol": protocol,
        "candidate_checkpoint": candidate_identity,
        "baseline_v4_checkpoint": baseline_identity,
        "workers": worker_count,
        "elapsed_seconds": time.perf_counter() - started,
        "matchups": results,
    }


def _evaluate_worker_arguments(arguments) -> list[dict[str, object]]:
    return _evaluate_worker(*arguments)


def _evaluate_worker(
    work_items: Sequence[tuple[str, EvaluationCase]],
    candidate_path: str,
    baseline_path: str,
) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    candidate = V4Agent.from_checkpoint(
        candidate_path,
        device="cpu",
        greedy=True,
    )
    baseline = V4Agent.from_checkpoint(
        baseline_path,
        device="cpu",
        greedy=True,
    )
    results = []
    for opponent, case in work_items:
        results.append(_play_round(case, opponent, candidate, baseline))
    return results


def _play_round(
    case: EvaluationCase,
    opponent: str,
    candidate: V4Agent,
    baseline: V4Agent,
) -> dict[str, object]:
    if opponent not in V4_MATCHUPS:
        raise ValueError(f"unknown V4 matchup opponent: {opponent}")
    candidate_seats = frozenset(case.candidate_seats)
    teachers = tuple(
        ShantenTeacher(
            seed=(
                case.seed
                ^ (0x9E3779B9 * (seat + 1))
                ^ (0x85EBCA6B * (case.game_index + 1))
            ),
            kan_probability=0.0,
            ready_policy="declare",
        )
        for seat in range(4)
    )
    arena = Arena(seed=case.seed, dealer=case.dealer)
    arena.start_round()
    action_counts = [Counter() for _ in range(4)]
    chongfeng_tracker = ChongfengTracker()
    fixed_discard_opportunities = [0] * 4
    fixed_discard_selections = [0] * 4
    step_count = 0
    while not arena.finished:
        decision = arena.current_decision()
        seat = int(decision.player)
        encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)
        if seat in candidate_seats:
            action_id = candidate.select_action_id(
                DecisionFeatureEncoder.encode(arena, seat)
            )
        elif opponent == "baseline_v4":
            action_id = baseline.select_action_id(
                DecisionFeatureEncoder.encode(arena, seat)
            )
        else:
            legal_mask = tuple(
                int(encoded.is_legal(action_id))
                for action_id in range(ACTION_SPACE_SIZE)
            )
            teacher_decision = teachers[seat].decide(EnvState(
                player_id=seat,
                observation=ObservationEncoder.encode(arena, seat),
                legal_mask=legal_mask,
            ))
            action_id = teacher_decision.selected_action_id
        if not encoded.is_legal(action_id):
            raise RuntimeError("V4 matchup agent selected an illegal action")
        selected_action = encoded.action_at(action_id)
        has_chongfeng_discard = any(
            encoded.is_legal(tile_id)
            and chongfeng_tracker.can_chongfeng(seat, tile_id)
            for tile_id in (YAO_JI_TILE_ID, BA_TONG_TILE_ID)
        )
        if has_chongfeng_discard:
            fixed_discard_opportunities[seat] += 1
        if selected_action.type.name == "DISCARD":
            is_chongfeng_candidate = chongfeng_tracker.observe_discard(
                seat,
                int(selected_action.tile.id),
            )
            if is_chongfeng_candidate:
                fixed_discard_selections[seat] += 1
        action_counts[seat][_action_category(action_id)] += 1
        arena.submit_action(selected_action)
        step_count += 1
        if step_count > 1_000:
            raise RuntimeError(
                f"V4 matchup exceeded 1000 steps at seed={case.seed}"
            )

    result = arena.result
    score_deltas = tuple(
        int(score) for score in arena.round_settlement.ledger.score_deltas
    )
    if sum(score_deltas) != 0:
        raise RuntimeError("V4 matchup settlement is not zero-sum")
    fangpao_player = int(result.fangpao_player)
    if not 0 <= fangpao_player < 4:
        fangpao_player = None
    hu_details = []
    for settlement in arena.round_settlement.hu_settlements:
        player = int(settlement.player_id)
        hu_details.append({
            "player": player,
            "side": (
                "candidate" if player in candidate_seats else "opponent"
            ),
            "pattern": str(settlement.features.pattern),
            "shape_fan": int(settlement.score.shape_fan),
            "total_fan": int(settlement.score.total_fan),
            "hu_points": int(settlement.score.hu_points),
        })
    first_fixed_discard_outcomes = [[] for _ in range(4)]
    outcome_tracker = ChongfengTracker()
    discard_history = tuple(arena.discard_history)
    ron_claimed_index = (
        len(discard_history) - 1
        if result.reason.name == "RON"
        else None
    )
    for index, event in enumerate(discard_history):
        actor = int(event.actor)
        if not outcome_tracker.observe_discard(actor, int(event.tile.id)):
            continue
        if event.was_called:
            outcome = "responsibility"
        elif index == ron_claimed_index:
            outcome = "ron_claimed"
        else:
            outcome = "chongfeng"
        first_fixed_discard_outcomes[actor].append({
            "tile": int(event.tile.id),
            "outcome": outcome,
        })
    return {
        "game_index": case.game_index,
        "wall_index": case.wall_index,
        "seed": case.seed,
        "dealer": case.dealer,
        "candidate_seats": list(case.candidate_seats),
        "opponent": opponent,
        "score_deltas": list(score_deltas),
        "candidate_score": sum(
            score_deltas[seat] for seat in candidate_seats
        ),
        "end_reason": str(result.reason),
        "hu_players": [int(player) for player in result.hu_players],
        "fangpao_player": fangpao_player,
        "hu_details": hu_details,
        "action_counts_by_player": [
            dict(counts) for counts in action_counts
        ],
        "score_components_by_player": [
            list(row)
            for row in settlement_component_deltas(arena.round_settlement)
        ],
        "fixed_discard_opportunities_by_player": (
            fixed_discard_opportunities
        ),
        "fixed_discard_selections_by_player": fixed_discard_selections,
        "first_fixed_discard_outcomes_by_player": (
            first_fixed_discard_outcomes
        ),
    }


def _validate_rounds(
    rounds: Sequence[Mapping[str, object]],
    schedule: Sequence[EvaluationCase],
) -> None:
    if len(rounds) != len(schedule):
        raise RuntimeError("round count does not match evaluation schedule")
    for result, case in zip(rounds, schedule):
        if result["game_index"] != case.game_index:
            raise RuntimeError("evaluation game order does not match schedule")
        if result["seed"] != case.seed or result["dealer"] != case.dealer:
            raise RuntimeError("evaluation round identity does not match schedule")
        if tuple(result["candidate_seats"]) != case.candidate_seats:
            raise RuntimeError("evaluation seats do not match schedule")
        if sum(result["score_deltas"]) != 0:
            raise RuntimeError("evaluation settlement is not zero-sum")


def _summarize_side(
    rounds: Sequence[Mapping[str, object]],
    side: str,
) -> dict[str, object]:
    if side not in {"candidate", "opponent"}:
        raise ValueError("side must be candidate or opponent")
    seat_rounds = len(rounds) * 2
    hu_count = 0
    fangpao_count = 0
    big_hand_count = 0
    pattern_counts = Counter()
    fan_counts = Counter()
    action_counts = Counter()
    decision_count = 0
    rounds_with_hu = 0
    score_total = 0
    component_totals = Counter({name: 0 for name in COMPONENT_NAMES})
    fixed_discard_opportunities = 0
    fixed_discard_selections = 0
    chongfeng_count = 0
    responsibility_count = 0
    for result in rounds:
        candidate_seats = set(result["candidate_seats"])
        side_seats = (
            candidate_seats
            if side == "candidate"
            else set(range(4)) - candidate_seats
        )
        score_total += sum(
            result["score_deltas"][seat] for seat in side_seats
        )
        details = [
            detail for detail in result["hu_details"]
            if detail["side"] == side
        ]
        hu_count += len(details)
        rounds_with_hu += int(bool(details))
        for detail in details:
            big_hand_count += int(detail["total_fan"] > 1)
            pattern_counts[detail["pattern"]] += 1
            fan_counts[str(detail["total_fan"])] += 1
        fangpao = result["fangpao_player"]
        fangpao_count += int(
            fangpao is not None and fangpao in side_seats
        )
        for seat in side_seats:
            counts = result["action_counts_by_player"][seat]
            action_counts.update(counts)
            decision_count += sum(counts.values())
            component_rows = result.get("score_components_by_player")
            if component_rows is not None:
                for name, value in zip(
                    COMPONENT_NAMES,
                    component_rows[seat],
                ):
                    component_totals[name] += int(value)
            opportunities = result.get(
                "fixed_discard_opportunities_by_player",
            )
            if opportunities is not None:
                fixed_discard_opportunities += int(opportunities[seat])
            selections = result.get(
                "fixed_discard_selections_by_player",
            )
            if selections is not None:
                fixed_discard_selections += int(selections[seat])
            outcomes = result.get(
                "first_fixed_discard_outcomes_by_player",
            )
            if outcomes is not None:
                seat_outcomes = outcomes[seat]
                if isinstance(seat_outcomes, str):
                    seat_outcomes = ({"outcome": seat_outcomes},)
                elif seat_outcomes is None:
                    seat_outcomes = ()
                for outcome in seat_outcomes:
                    chongfeng_count += int(
                        outcome["outcome"] == "chongfeng"
                    )
                    responsibility_count += int(
                        outcome["outcome"] == "responsibility"
                    )

    return {
        "seat_rounds": seat_rounds,
        "mean_score_per_seat": score_total / seat_rounds,
        "hu_count": hu_count,
        "hu_rate_per_seat_round": hu_count / seat_rounds,
        "rounds_with_hu": rounds_with_hu,
        "round_hu_rate": rounds_with_hu / len(rounds),
        "fangpao_count": fangpao_count,
        "fangpao_rate_per_seat_round": fangpao_count / seat_rounds,
        "round_fangpao_rate": fangpao_count / len(rounds),
        "big_hand_count": big_hand_count,
        "big_hand_rate_among_hu": (
            big_hand_count / hu_count if hu_count else 0.0
        ),
        "hu_pattern_counts": dict(pattern_counts),
        "hu_total_fan_counts": dict(fan_counts),
        "decision_count": decision_count,
        "action_counts": dict(action_counts),
        "action_rates_per_decision": {
            name: count / decision_count
            for name, count in action_counts.items()
        },
        "action_rates_per_seat_round": {
            name: count / seat_rounds
            for name, count in action_counts.items()
        },
        "score_component_totals": dict(component_totals),
        "score_component_means_per_seat_round": {
            name: value / seat_rounds
            for name, value in component_totals.items()
        },
        "fixed_discard_opportunity_decisions": (
            fixed_discard_opportunities
        ),
        "fixed_discard_selections": fixed_discard_selections,
        "fixed_discard_selection_rate": (
            fixed_discard_selections / fixed_discard_opportunities
            if fixed_discard_opportunities
            else 0.0
        ),
        "chongfeng_count": chongfeng_count,
        "responsibility_first_fixed_discard_count": responsibility_count,
        "chongfeng_rate_among_first_fixed_discards": (
            chongfeng_count / (chongfeng_count + responsibility_count)
            if chongfeng_count + responsibility_count
            else 0.0
        ),
        "absolute_seats": _summarize_absolute_seats(rounds, side),
    }


def _summarize_absolute_seats(
    rounds: Sequence[Mapping[str, object]],
    side: str,
) -> dict[str, object]:
    summaries = {}
    for seat in range(4):
        seat_rounds = []
        for result in rounds:
            candidate_seats = set(result["candidate_seats"])
            belongs_to_side = (
                seat in candidate_seats
                if side == "candidate"
                else seat not in candidate_seats
            )
            if belongs_to_side:
                seat_rounds.append(result)
        score_total = sum(
            result["score_deltas"][seat] for result in seat_rounds
        )
        hu_count = sum(
            detail["player"] == seat
            for result in seat_rounds
            for detail in result["hu_details"]
        )
        fangpao_count = sum(
            result["fangpao_player"] == seat for result in seat_rounds
        )
        actions = Counter()
        for result in seat_rounds:
            actions.update(result["action_counts_by_player"][seat])
        count = len(seat_rounds)
        summaries[str(seat)] = {
            "seat_rounds": count,
            "mean_score": score_total / count if count else 0.0,
            "hu_rate": hu_count / count if count else 0.0,
            "fangpao_rate": fangpao_count / count if count else 0.0,
            "action_counts": dict(actions),
        }
    return summaries


def _wall_cluster_bootstrap_interval(
    rounds: Sequence[Mapping[str, object]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    if not rounds:
        raise ValueError("bootstrap requires at least one round")
    if resamples <= 0:
        raise ValueError("bootstrap resample count must be positive")
    clusters = defaultdict(list)
    for result in rounds:
        clusters[int(result["seed"])].append(
            float(result["candidate_score"]) / 2.0
        )
    cluster_values = tuple(
        np.asarray(values, dtype=np.float64)
        for _, values in sorted(clusters.items())
    )
    if len(cluster_values) == 1:
        value = float(cluster_values[0].mean())
        return [value, value]

    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    cluster_count = len(cluster_values)
    for index in range(resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        combined = np.concatenate([cluster_values[item] for item in sampled])
        means[index] = combined.mean()
    low, high = np.quantile(means, (0.025, 0.975))
    return [float(low), float(high)]


def _action_category(action_id: int) -> str:
    if 0 <= action_id < TILE_TYPE_COUNT:
        return "discard"
    if ANKAN_ACTION_START <= action_id < KAKAN_ACTION_START:
        return "ankan"
    if KAKAN_ACTION_START <= action_id < PON_ACTION_ID:
        return "kakan"
    names = {
        PON_ACTION_ID: "pon",
        DAIMINKAN_ACTION_ID: "daiminkan",
        TSUMO_ACTION_ID: "tsumo",
        RON_ACTION_ID: "ron",
        PASS_ACTION_ID: "pass",
        DECLARE_READY_ACTION_ID: "declare_ready",
    }
    if action_id not in names:
        raise ValueError(f"unknown action id {action_id}")
    return names[action_id]


def _summarize_matchup(
    rounds: Sequence[Mapping[str, object]],
    opponent: str,
) -> dict[str, object]:
    candidate_scores = np.asarray([
        float(round_["candidate_score"]) / 2.0
        for round_ in rounds
    ])
    return {
        "opponent": opponent,
        "episodes": len(rounds),
        "unique_walls": len({round_["seed"] for round_ in rounds}),
        "candidate_mean_score_per_seat": float(candidate_scores.mean()),
        "opponent_mean_score_per_seat": float(-candidate_scores.mean()),
        "candidate_score_95ci": _wall_cluster_bootstrap_interval(rounds),
        "candidate_positive_score_rounds": int((candidate_scores > 0).sum()),
        "candidate_negative_score_rounds": int((candidate_scores < 0).sum()),
        "candidate_zero_score_rounds": int((candidate_scores == 0).sum()),
        "end_reasons": dict(Counter(
            round_["end_reason"] for round_ in rounds
        )),
        "sides": {
            side: _summarize_side(rounds, side)
            for side in ("candidate", "opponent")
        },
        "rounds": list(rounds),
    }


def _file_identity(path: Path) -> dict[str, object]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest().upper()
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest,
    }


def write_v4_matchup_report(
    path: Path | str,
    evaluation: Mapping[str, object],
) -> None:
    names = {
        "baseline_v4": "起点模型",
        "teacher": "Teacher",
    }
    lines = [
        "# V4 PPO固定牌墙正式评估",
        "",
        (
            "对基线模型局数："
            f"{evaluation['protocol']['matchup_game_counts']['baseline_v4']}"
        ),
        (
            "对Teacher局数："
            f"{evaluation['protocol']['matchup_game_counts']['teacher']}"
        ),
        f"独立牌墙：{evaluation['protocol']['unique_walls']}",
        f"评估日程SHA256：`{evaluation['protocol']['schedule_sha256']}`",
        "",
        "| 对手 | 每席净分 | 95%置信区间 | 胡牌率 | 放炮率 | 大牌占比 | 每席碰牌 | 每席杠牌 | 每席报听 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for opponent in V4_MATCHUPS:
        result = evaluation["matchups"][opponent]
        candidate = result["sides"]["candidate"]
        actions = candidate["action_rates_per_seat_round"]
        low, high = result["candidate_score_95ci"]
        kan_rate = sum(
            actions.get(name, 0.0)
            for name in ("ankan", "kakan", "daiminkan")
        )
        lines.append(
            f"| {names[opponent]} "
            f"| {result['candidate_mean_score_per_seat']:+.3f} "
            f"| [{low:+.3f}, {high:+.3f}] "
            f"| {candidate['hu_rate_per_seat_round']:.3%} "
            f"| {candidate['fangpao_rate_per_seat_round']:.3%} "
            f"| {candidate['big_hand_rate_among_hu']:.3%} "
            f"| {actions.get('pon', 0.0):.4f} "
            f"| {kan_rate:.4f} "
            f"| {actions.get('declare_ready', 0.0):.4f} |"
        )
    lines += [
        "",
        "## 模型身份",
        "",
        f"候选模型SHA256：`{evaluation['candidate_checkpoint']['sha256']}`",
        f"起点模型SHA256：`{evaluation['baseline_v4_checkpoint']['sha256']}`",
    ]
    _atomic_write_text(Path(path), "\n".join(lines) + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a V4 candidate against baseline V4 and Teacher.",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-v4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int, default=1_000)
    parser.add_argument(
        "--teacher-games",
        type=int,
        help="Teacher matchup games; defaults to --games-per-matchup",
    )
    parser.add_argument("--seed-start", type=int, default=932_000_001)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    markdown_path = args.output.with_suffix(".md")
    if args.output.exists() or markdown_path.exists():
        parser.error("evaluation output already exists")
    evaluation = evaluate_v4_matchups(
        args.candidate,
        args.baseline_v4,
        games_per_matchup=args.games_per_matchup,
        teacher_games=args.teacher_games,
        seed_start=args.seed_start,
        workers=args.workers,
    )
    _atomic_write_text(
        args.output,
        json.dumps(evaluation, indent=2, ensure_ascii=True),
    )
    write_v4_matchup_report(markdown_path, evaluation)
    print(json.dumps({
        "candidate": evaluation["candidate_checkpoint"],
        "baseline_v4": evaluation["baseline_v4_checkpoint"],
        "protocol": evaluation["protocol"],
        "elapsed_seconds": evaluation["elapsed_seconds"],
        "matchups": {
            opponent: {
                key: value
                for key, value in result.items()
                if key != "rounds"
            }
            for opponent, result in evaluation["matchups"].items()
        },
        "output": str(args.output.resolve()),
        "report": str(markdown_path.resolve()),
    }, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
