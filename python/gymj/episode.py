from dataclasses import dataclass
from typing import Tuple

from ._gymj import RoundEndReason
from .agent import Agent
from .env import GymjEnv
from .reward import PLAYER_COUNT, RewardVector
from .trajectory import TrajectoryCollector, Transition


PlayerTrajectories = Tuple[
    Tuple[Transition, ...],
    Tuple[Transition, ...],
    Tuple[Transition, ...],
    Tuple[Transition, ...],
]


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    step_count: int
    reward_totals: RewardVector
    trajectories: PlayerTrajectories
    end_reason: RoundEndReason
    hu_players: Tuple[int, ...]


def run_episode(
    env: GymjEnv,
    agent: Agent,
    seed: int,
    max_steps: int = 1000,
) -> EpisodeResult:
    if max_steps <= 0:
        raise ValueError("max steps must be positive")

    reset_episode = getattr(agent, "reset_episode", None)
    if callable(reset_episode):
        reset_episode(seed)

    collector = TrajectoryCollector()
    decision = env.reset(seed=seed)
    reward_totals = [0.0] * PLAYER_COUNT

    for step_count in range(1, max_steps + 1):
        action_id = agent.select_action_id(
            decision.observation,
            decision.legal_mask,
        )
        result = env.step(action_id)
        collector.record(decision, result)

        for player_id, reward in enumerate(result.rewards):
            reward_totals[player_id] += reward

        if result.done:
            round_result = env.round_result
            trajectories = tuple(
                collector.trajectory_for(player_id)
                for player_id in range(PLAYER_COUNT)
            )
            return EpisodeResult(
                seed=seed,
                step_count=step_count,
                reward_totals=tuple(reward_totals),
                trajectories=trajectories,
                end_reason=round_result.reason,
                hu_players=tuple(round_result.hu_players),
            )

        decision = result.next_decision

    raise RuntimeError(f"episode exceeded {max_steps} steps")
