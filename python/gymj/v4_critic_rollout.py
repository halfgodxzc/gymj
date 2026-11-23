from __future__ import annotations

from dataclasses import dataclass
import math

from ._gymj import (
    ACTION_SPACE_SIZE,
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    PrivilegedPatternEncoder,
    PrivilegedStateEncoder,
    RoundEndReason,
)
from .critic_rollout import (
    DEFAULT_SCORE_DIVISOR,
    DEFAULT_SHANTEN_DIVISOR,
    monte_carlo_returns,
)
from .reward import PLAYER_COUNT, ShantenRewardTracker
from .v4_agent import V4Agent


@dataclass(frozen=True)
class V4CriticSample:
    episode_seed: int
    player_id: int
    decision_index: int
    decision_features: object
    privileged_state: object
    privileged_patterns: object
    action_id: int
    score_return: float
    shanten_return: float

    def __post_init__(self) -> None:
        if not -(1 << 63) <= self.episode_seed < (1 << 63):
            raise ValueError("episode seed must fit in int64")
        if not 0 <= self.player_id < PLAYER_COUNT:
            raise ValueError("player id must be in range 0..3")
        if not 0 <= self.decision_index < (1 << 32):
            raise ValueError("decision index must fit in uint32")
        if not 0 <= self.action_id < ACTION_SPACE_SIZE:
            raise ValueError("action id is outside the action space")
        if not self.decision_features.actions[self.action_id].legal:
            raise ValueError("recorded V4 Critic action must be legal")
        if not math.isfinite(self.score_return):
            raise ValueError("score return must be finite")
        if not math.isfinite(self.shanten_return):
            raise ValueError("shanten return must be finite")


@dataclass(frozen=True)
class V4CriticEpisode:
    seed: int
    step_count: int
    samples: tuple[V4CriticSample, ...]
    score_deltas: tuple[int, int, int, int]
    end_reason: RoundEndReason
    hu_players: tuple[int, ...]
    score_divisor: float
    shanten_divisor: float


@dataclass(frozen=True)
class _PendingSample:
    decision_index: int
    decision_features: object
    privileged_state: object
    privileged_patterns: object
    action_id: int
    shanten_reward: float


def run_v4_critic_episode(
    agent: V4Agent,
    seed: int,
    *,
    dealer: int = 0,
    max_steps: int = 1000,
    score_divisor: float = DEFAULT_SCORE_DIVISOR,
    shanten_divisor: float = DEFAULT_SHANTEN_DIVISOR,
) -> V4CriticEpisode:
    """Collect one complete V4 self-play episode with MC value labels."""

    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    if score_divisor <= 0.0:
        raise ValueError("score divisor must be positive")
    if shanten_divisor <= 0.0:
        raise ValueError("shanten divisor must be positive")

    arena = Arena(seed=seed, dealer=dealer)
    arena.start_round()
    agent.reset_episode(seed)
    reward_tracker = ShantenRewardTracker()
    pending: list[list[_PendingSample]] = [
        [] for _ in range(PLAYER_COUNT)
    ]

    for step_count in range(1, max_steps + 1):
        decision = arena.current_decision()
        player_id = int(decision.player)
        decision_features = DecisionFeatureEncoder.encode(arena, player_id)
        privileged_state = PrivilegedStateEncoder.encode(arena, player_id)
        privileged_patterns = PrivilegedPatternEncoder.encode(
            arena,
            player_id,
        )
        encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)
        action_id = agent.select_action_id(decision_features)
        if not encoded.is_legal(action_id):
            raise RuntimeError("V4 agent selected an illegal action")
        action = encoded.action_at(action_id)
        reward_step = reward_tracker.step(arena, action)
        pending[player_id].append(_PendingSample(
            decision_index=step_count - 1,
            decision_features=decision_features,
            privileged_state=privileged_state,
            privileged_patterns=privileged_patterns,
            action_id=action_id,
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
        samples = []
        for seat in range(PLAYER_COUNT):
            shanten_returns = monte_carlo_returns([
                item.shanten_reward for item in pending[seat]
            ])
            for item, shanten_return in zip(
                pending[seat],
                shanten_returns,
            ):
                samples.append(V4CriticSample(
                    episode_seed=seed,
                    player_id=seat,
                    decision_index=item.decision_index,
                    decision_features=item.decision_features,
                    privileged_state=item.privileged_state,
                    privileged_patterns=item.privileged_patterns,
                    action_id=item.action_id,
                    score_return=score_deltas[seat] / score_divisor,
                    shanten_return=shanten_return / shanten_divisor,
                ))
        samples.sort(key=lambda sample: sample.decision_index)
        result = arena.result
        return V4CriticEpisode(
            seed=seed,
            step_count=step_count,
            samples=tuple(samples),
            score_deltas=score_deltas,
            end_reason=result.reason,
            hu_players=tuple(int(player) for player in result.hu_players),
            score_divisor=score_divisor,
            shanten_divisor=shanten_divisor,
        )

    raise RuntimeError(f"V4 Critic rollout exceeded {max_steps} steps")


__all__ = [
    "V4CriticEpisode",
    "V4CriticSample",
    "run_v4_critic_episode",
]
