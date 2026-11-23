from __future__ import annotations

from dataclasses import dataclass
import math

from ._gymj import (
    ACTION_SPACE_SIZE,
    ActionEncoder,
    ActionType,
    Arena,
    DecisionFeatureEncoder,
    ObservationEncoder,
    PrivilegedPatternEncoder,
    PrivilegedStateEncoder,
    RoundEndReason,
)
from .critic_rollout import (
    DEFAULT_SCORE_DIVISOR,
    DEFAULT_SHANTEN_DIVISOR,
    monte_carlo_returns,
)
from .big_hand_agent import BigHandAgent
from .chicken_attribution import SubmittedDecision, aggregate_chicken_attributions
from .reward import DEFAULT_SHANTEN_REWARD_SCALE, PLAYER_COUNT, ShantenRewardTracker
from .teacher import ShantenTeacher
from .v4_agent import V4Agent


SELF_PLAY_OPPONENT = "self_play"
STABLE_R5_OPPONENT = "stable_r5"
TEACHER_OPPONENT = "teacher"
BIG_HAND_OPPONENT = "big_hand"
OPPONENT_KINDS = frozenset({
    SELF_PLAY_OPPONENT,
    STABLE_R5_OPPONENT,
    TEACHER_OPPONENT,
    BIG_HAND_OPPONENT,
})


@dataclass(frozen=True)
class V4PPOSample:
    episode_seed: int
    player_id: int
    decision_index: int
    decision_features: object
    privileged_state: object
    privileged_patterns: object
    action_id: int
    old_log_probability: float
    old_entropy: float
    actor_trainable: bool
    score_return: float
    shanten_return: float
    score_returns_all: tuple[float, ...] | None = None
    chicken_attribution_points: float | None = None

    def __post_init__(self) -> None:
        if self.chicken_attribution_points is not None and not math.isfinite(self.chicken_attribution_points):
            raise ValueError("chicken attribution points must be finite or absent")
        if not -(1 << 63) <= self.episode_seed < (1 << 63):
            raise ValueError("episode seed must fit in int64")
        if not 0 <= self.player_id < PLAYER_COUNT:
            raise ValueError("player id must be in range 0..3")
        if not 0 <= self.decision_index < (1 << 32):
            raise ValueError("decision index must fit in uint32")
        if not 0 <= self.action_id < ACTION_SPACE_SIZE:
            raise ValueError("action id is outside the action space")
        action = self.decision_features.actions[self.action_id]
        adapter_supported = action.semantics.type in (
            ActionType.ANKAN,
            ActionType.KAKAN,
            ActionType.DAIMINKAN,
        )
        if not action.legal or not (
            action.supported_by_v4 or adapter_supported
        ):
            raise ValueError("recorded PPO action must be selectable by V4")
        for name, value in (
            ("old log probability", self.old_log_probability),
            ("old entropy", self.old_entropy),
            ("score return", self.score_return),
            ("shanten return", self.shanten_return),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class V4PPOEpisode:
    seed: int
    step_count: int
    samples: tuple[V4PPOSample, ...]
    score_deltas: tuple[int, int, int, int]
    end_reason: RoundEndReason
    hu_players: tuple[int, ...]
    score_divisor: float
    shanten_divisor: float
    chicken_attribution_audit: dict | None = None


@dataclass(frozen=True)
class _PendingSample:
    decision_index: int
    decision_features: object
    privileged_state: object
    privileged_patterns: object
    action_id: int
    old_log_probability: float
    old_entropy: float
    actor_trainable: bool
    shanten_reward: float


def run_v4_ppo_episode(
    agent: V4Agent,
    seed: int,
    *,
    dealer: int = 0,
    max_steps: int = 1000,
    score_divisor: float = DEFAULT_SCORE_DIVISOR,
    shanten_divisor: float = DEFAULT_SHANTEN_DIVISOR,
    shanten_reward_enabled: bool = True,
) -> V4PPOEpisode:
    """Collect one on-policy V4 episode with terminal Monte Carlo labels."""

    return run_v4_opponent_pool_episode(
        agent,
        seed,
        candidate_seats=tuple(range(PLAYER_COUNT)),
        opponent_kind=SELF_PLAY_OPPONENT,
        dealer=dealer,
        max_steps=max_steps,
        score_divisor=score_divisor,
        shanten_divisor=shanten_divisor,
        shanten_reward_enabled=shanten_reward_enabled,
    )


def run_v4_opponent_pool_episode(
    candidate_agent: V4Agent,
    seed: int,
    *,
    candidate_seats: tuple[int, ...],
    opponent_kind: str,
    baseline_agent: V4Agent | None = None,
    dealer: int = 0,
    max_steps: int = 1000,
    score_divisor: float = DEFAULT_SCORE_DIVISOR,
    shanten_divisor: float = DEFAULT_SHANTEN_DIVISOR,
    shanten_reward_enabled: bool = True,
) -> V4PPOEpisode:
    """Collect candidate-policy samples against a fixed opponent pool."""

    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    if score_divisor <= 0.0 or shanten_divisor <= 0.0:
        raise ValueError("return divisors must be positive")
    if opponent_kind not in OPPONENT_KINDS:
        raise ValueError(f"unknown opponent kind: {opponent_kind}")

    candidate_seat_set = frozenset(int(seat) for seat in candidate_seats)
    if len(candidate_seat_set) != len(candidate_seats):
        raise ValueError("candidate seats must be unique")
    if not candidate_seat_set or any(
        not 0 <= seat < PLAYER_COUNT for seat in candidate_seat_set
    ):
        raise ValueError("candidate seats must be a non-empty subset of 0..3")
    if opponent_kind == SELF_PLAY_OPPONENT:
        if len(candidate_seat_set) != PLAYER_COUNT:
            raise ValueError("self play requires all four candidate seats")
        if baseline_agent is not None:
            raise ValueError("self play does not use a baseline agent")
    elif len(candidate_seat_set) == PLAYER_COUNT:
        raise ValueError("fixed-opponent games require at least one opponent seat")
    if opponent_kind == STABLE_R5_OPPONENT and baseline_agent is None:
        raise ValueError("stable R5 games require a baseline agent")
    if opponent_kind != STABLE_R5_OPPONENT and baseline_agent is not None:
        raise ValueError("baseline agent is only valid for stable R5 games")

    arena = Arena(seed=seed, dealer=dealer)
    arena.start_round()
    candidate_agent.reset_episode(seed)
    if baseline_agent is not None:
        baseline_agent.reset_episode(seed)
    teachers = tuple(
        ShantenTeacher(
            seed=seed ^ (0x9E3779B9 * (seat + 1)),
            kan_probability=0.0,
            ready_policy="declare",
        )
        for seat in range(PLAYER_COUNT)
    )
    big_hand_agent = BigHandAgent(
        seed=0xC2B2AE35,
        kan_probability=0.5,
        seven_pairs_max_time_penalty=1.0,
        dadui_switch_margin=0.0,
    )
    big_hand_agent.reset_episode(seed)
    reward_tracker = ShantenRewardTracker(
        shanten_scale=DEFAULT_SHANTEN_REWARD_SCALE if shanten_reward_enabled else 0.0,
        hu_reward=0.0,
    )
    pending: list[list[_PendingSample]] = [
        [] for _ in range(PLAYER_COUNT)
    ]
    submitted_decisions = []

    for step_count in range(1, max_steps + 1):
        decision = arena.current_decision()
        player_id = int(decision.player)
        encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)
        pending_sample = None
        if player_id in candidate_seat_set:
            features = DecisionFeatureEncoder.encode(arena, player_id)
            privileged_state = PrivilegedStateEncoder.encode(arena, player_id)
            privileged_patterns = PrivilegedPatternEncoder.encode(
                arena,
                player_id,
            )
            selection = candidate_agent.select_action(features)
            action_id = selection.action_id
            pending_sample = (
                features,
                privileged_state,
                privileged_patterns,
                selection,
            )
        elif opponent_kind == STABLE_R5_OPPONENT:
            if baseline_agent is None:
                raise RuntimeError("stable R5 opponent is unavailable")
            action_id = baseline_agent.select_action_id(
                DecisionFeatureEncoder.encode(arena, player_id)
            )
        elif opponent_kind == TEACHER_OPPONENT:
            legal_mask = tuple(
                int(encoded.is_legal(action_id))
                for action_id in range(ACTION_SPACE_SIZE)
            )
            action_id = teachers[player_id].select_action_id(
                ObservationEncoder.encode(arena, player_id),
                legal_mask,
            )
        elif opponent_kind == BIG_HAND_OPPONENT:
            legal_mask = tuple(
                int(encoded.is_legal(action_id))
                for action_id in range(ACTION_SPACE_SIZE)
            )
            action_id = big_hand_agent.select_action_id(
                ObservationEncoder.encode(arena, player_id),
                legal_mask,
            )
        else:
            raise RuntimeError("non-candidate seat has no fixed opponent")

        if not encoded.is_legal(action_id):
            raise RuntimeError("opponent-pool agent selected an illegal action")
        action = encoded.action_at(action_id)
        submitted_decisions.append(SubmittedDecision(step_count - 1, action))
        reward_step = reward_tracker.step(arena, action)
        if pending_sample is not None:
            (
                features,
                privileged_state,
                privileged_patterns,
                selection,
            ) = pending_sample
            pending[player_id].append(_PendingSample(
                decision_index=step_count - 1,
                decision_features=features,
                privileged_state=privileged_state,
                privileged_patterns=privileged_patterns,
                action_id=selection.action_id,
                old_log_probability=selection.log_probability,
                old_entropy=selection.entropy,
                actor_trainable=selection.actor_trainable,
                shanten_reward=reward_step.shanten_reward,
            ))

        if not arena.finished:
            continue

        score_deltas = tuple(
            int(value)
            for value in arena.round_settlement.ledger.score_deltas
        )
        if len(score_deltas) != PLAYER_COUNT or sum(score_deltas) != 0:
            raise RuntimeError("terminal score deltas must be four-player zero sum")
        settlement = arena.round_settlement
        if not hasattr(settlement, "chicken_attributions"):
            raise RuntimeError("PPO chicken attribution requires rebuilding the Arena extension")
        chicken_points = aggregate_chicken_attributions(
            settlement, arena.history, submitted_decisions,
        )
        chicken_audit = {
            "version": 1,
            "decision_points": [
                {"player_id": player, "decision_index": index, "points": points,
                 "candidate": player in candidate_seat_set,
                 "action_id": ActionEncoder.encode_action(submitted_decisions[index].action),
                 "action_type": submitted_decisions[index].action.type.name,
                 "tile": submitted_decisions[index].action.tile.id}
                for (player, index), points in sorted(chicken_points.items())
            ],
            "entries": [
                {"player_id": entry.player_id, "event_sequence": entry.event_sequence,
                 "transfer_index": entry.transfer_index, "kind": entry.kind.name,
                 "tile": entry.tile.id, "points": entry.points}
                for entry in settlement.chicken_attributions
            ],
            "unattributed_deltas": list(settlement.chicken_unattributed_deltas),
            "ledger": [
                {"transfer_index": index, "category": transfer.category.name,
                 "from_player": transfer.from_player, "to_player": transfer.to_player,
                 "points": transfer.points, "tile": transfer.tile.id,
                 "chicken_owner": transfer.chicken_owner}
                for index, transfer in enumerate(settlement.ledger.transfers)
                if transfer.category.name in ("CHICKEN", "RESPONSIBILITY_CHICKEN")
            ],
        }
        samples = []
        for seat in sorted(candidate_seat_set):
            shanten_returns = monte_carlo_returns([
                item.shanten_reward for item in pending[seat]
            ])
            for item, shanten_return in zip(pending[seat], shanten_returns):
                samples.append(V4PPOSample(
                    episode_seed=seed,
                    player_id=seat,
                    decision_index=item.decision_index,
                    decision_features=item.decision_features,
                    privileged_state=item.privileged_state,
                    privileged_patterns=item.privileged_patterns,
                    action_id=item.action_id,
                    old_log_probability=item.old_log_probability,
                    old_entropy=item.old_entropy,
                    actor_trainable=item.actor_trainable,
                    score_return=score_deltas[seat] / score_divisor,
                    score_returns_all=tuple(score_deltas[(seat+j)%PLAYER_COUNT] / score_divisor for j in range(PLAYER_COUNT)),
                    shanten_return=shanten_return / shanten_divisor,
                    chicken_attribution_points=chicken_points.get((seat, item.decision_index), 0),
                ))
        samples.sort(key=lambda sample: sample.decision_index)
        result = arena.result
        return V4PPOEpisode(
            seed=seed,
            step_count=step_count,
            samples=tuple(samples),
            score_deltas=score_deltas,
            end_reason=result.reason,
            hu_players=tuple(int(player) for player in result.hu_players),
            score_divisor=score_divisor,
            shanten_divisor=shanten_divisor,
            chicken_attribution_audit=chicken_audit,
        )

    raise RuntimeError(f"V4 PPO rollout exceeded {max_steps} steps")


__all__ = [
    "BIG_HAND_OPPONENT",
    "OPPONENT_KINDS",
    "SELF_PLAY_OPPONENT",
    "STABLE_R5_OPPONENT",
    "TEACHER_OPPONENT",
    "V4PPOEpisode",
    "V4PPOSample",
    "run_v4_opponent_pool_episode",
    "run_v4_ppo_episode",
]
