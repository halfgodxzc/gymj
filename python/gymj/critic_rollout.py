from dataclasses import dataclass
from typing import Sequence, Tuple

from ._gymj import CompactObservation, RoundEndReason
from .agent import Agent
from .env import GymjEnv
from .reward import PLAYER_COUNT
from .trajectory import PackedLegalMask, pack_legal_mask


V2_AGENT_ID = 0
TEACHER_AGENT_ID = 1
DEFAULT_SCORE_DIVISOR = 30.0
DEFAULT_SHANTEN_DIVISOR = 10.0


@dataclass(frozen=True)
class CriticSample:
    episode_seed: int
    player_id: int
    agent_id: int
    observation: CompactObservation
    legal_mask_words: PackedLegalMask
    action_id: int
    score_return: float
    shanten_return: float


@dataclass(frozen=True)
class CriticEpisode:
    seed: int
    step_count: int
    samples: Tuple[CriticSample, ...]
    score_deltas: Tuple[int, int, int, int]
    end_reason: RoundEndReason
    hu_players: Tuple[int, ...]
    agent_ids: Tuple[int, int, int, int]
    score_divisor: float
    shanten_divisor: float


@dataclass(frozen=True)
class _PendingSample:
    observation: CompactObservation
    legal_mask_words: PackedLegalMask
    action_id: int
    shanten_reward: float


def monte_carlo_returns(
    rewards: Sequence[float],
    gamma: float = 1.0,
) -> Tuple[float, ...]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in range 0..1")

    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return tuple(returns)


def run_critic_episode(
    env: GymjEnv,
    agents: Sequence[Agent],
    agent_ids: Sequence[int],
    seed: int,
    max_steps: int = 1000,
    score_divisor: float = DEFAULT_SCORE_DIVISOR,
    shanten_divisor: float = DEFAULT_SHANTEN_DIVISOR,
) -> CriticEpisode:
    if len(agents) != PLAYER_COUNT:
        raise ValueError("critic rollout requires four seat agents")
    if len(agent_ids) != PLAYER_COUNT:
        raise ValueError("critic rollout requires four agent ids")
    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    if score_divisor <= 0.0:
        raise ValueError("score divisor must be positive")
    if shanten_divisor <= 0.0:
        raise ValueError("shanten divisor must be positive")

    normalized_agent_ids = tuple(int(value) for value in agent_ids)
    pending = [[] for _ in range(PLAYER_COUNT)]
    decision = env.reset(seed=seed)

    for step_count in range(1, max_steps + 1):
        player_id = decision.player_id
        action_id = agents[player_id].select_action_id(
            decision.observation,
            decision.legal_mask,
        )
        result = env.step(action_id)
        pending[player_id].append(_PendingSample(
            observation=decision.observation,
            legal_mask_words=pack_legal_mask(decision.legal_mask),
            action_id=action_id,
            shanten_reward=result.reward_step.shanten_reward,
        ))

        if result.done:
            score_deltas = env.terminal_score_deltas
            samples = []
            for seat in range(PLAYER_COUNT):
                shanten_returns = monte_carlo_returns(
                    [item.shanten_reward for item in pending[seat]],
                )
                for item, shanten_return in zip(
                    pending[seat],
                    shanten_returns,
                ):
                    samples.append(CriticSample(
                        episode_seed=seed,
                        player_id=seat,
                        agent_id=normalized_agent_ids[seat],
                        observation=item.observation,
                        legal_mask_words=item.legal_mask_words,
                        action_id=item.action_id,
                        score_return=score_deltas[seat] / score_divisor,
                        shanten_return=(
                            shanten_return / shanten_divisor
                        ),
                    ))

            round_result = env.round_result
            return CriticEpisode(
                seed=seed,
                step_count=step_count,
                samples=tuple(samples),
                score_deltas=score_deltas,
                end_reason=round_result.reason,
                hu_players=tuple(round_result.hu_players),
                agent_ids=normalized_agent_ids,
                score_divisor=score_divisor,
                shanten_divisor=shanten_divisor,
            )

        decision = result.next_decision
        if decision is None:
            raise RuntimeError("unfinished rollout has no next decision")

    raise RuntimeError(f"critic rollout exceeded {max_steps} steps")


__all__ = [
    "CriticEpisode",
    "CriticSample",
    "DEFAULT_SCORE_DIVISOR",
    "DEFAULT_SHANTEN_DIVISOR",
    "TEACHER_AGENT_ID",
    "V2_AGENT_ID",
    "monte_carlo_returns",
    "run_critic_episode",
]
