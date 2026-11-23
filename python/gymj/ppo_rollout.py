from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from ._gymj import ACTION_SPACE_SIZE, CompactObservation, PrivilegedState, RoundEndReason
from .behavior_cloning import BehaviorClonePolicy, tensorize_observations
from .critic_rollout import (
    DEFAULT_SCORE_DIVISOR,
    DEFAULT_SHANTEN_DIVISOR,
    monte_carlo_returns,
)
from .env import GymjEnv
from .policy_sampling import (
    KAN_ACTION_IDS,
    ActorSamplingConfig,
    sample_policy_actions,
)
from .reward import PLAYER_COUNT
from .settlement_components import settlement_component_deltas
from .trajectory import PackedLegalMask, pack_legal_mask


@dataclass(frozen=True)
class PPOSample:
    episode_seed: int
    player_id: int
    observation: CompactObservation
    legal_mask_words: PackedLegalMask
    action_id: int
    old_log_probability: float
    actor_trainable: bool
    score_return: float
    shanten_return: float
    component_return: Tuple[float, float, float, float]
    privileged_state: PrivilegedState | None = None


@dataclass(frozen=True)
class PPOEpisode:
    seed: int
    samples: Tuple[PPOSample, ...]
    score_deltas: Tuple[int, int, int, int]
    component_deltas: Tuple[Tuple[int, int, int, int], ...]
    end_reason: RoundEndReason
    hu_players: Tuple[int, ...]


@dataclass(frozen=True)
class _PendingSample:
    observation: CompactObservation
    legal_mask_words: PackedLegalMask
    action_id: int
    old_log_probability: float
    actor_trainable: bool
    shanten_reward: float
    privileged_state: PrivilegedState | None = None


class RolloutPolicy:
    def __init__(
        self,
        policy: BehaviorClonePolicy,
        sampling_config: ActorSamplingConfig,
        seed: int,
    ):
        self.policy = policy
        self.sampling_config = sampling_config
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.policy.eval()

    def select(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> tuple[int, float, bool]:
        if len(legal_mask) != ACTION_SPACE_SIZE:
            raise ValueError("legal mask must match the action space")
        if any(value not in (0, 1, False, True) for value in legal_mask):
            raise ValueError("legal mask values must be zero or one")
        available = [
            index for index, valid in enumerate(legal_mask)
            if valid and (
                self.sampling_config.kan_learning_enabled
                or index not in KAN_ACTION_IDS
            )
        ]
        if not available:
            raise ValueError("legal mask has no selectable actions")
        if len(available) == 1:
            return available[0], 0.0, False
        batch = tensorize_observations(
            (observation,),
            include_action_features=self.policy.config.action_feature_enabled,
            legal_mask_words=(pack_legal_mask(legal_mask),),
        )
        legal = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0)
        with torch.inference_mode():
            sampled = sample_policy_actions(
                self.policy(batch),
                legal,
                self.sampling_config,
                self.generator,
            )
        return (
            int(sampled.action_ids.item()),
            float(sampled.log_probability.item()),
            int(legal.sum().item()) > 1,
        )


def run_ppo_episode(
    env: GymjEnv,
    rollout_policy: RolloutPolicy,
    seed: int,
    max_steps: int = 1000,
    score_divisor: float = DEFAULT_SCORE_DIVISOR,
    shanten_divisor: float = DEFAULT_SHANTEN_DIVISOR,
    collect_privileged: bool = False,
) -> PPOEpisode:
    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    if score_divisor <= 0.0 or shanten_divisor <= 0.0:
        raise ValueError("return divisors must be positive")

    pending = [[] for _ in range(PLAYER_COUNT)]
    decision = env.reset(seed=seed)
    for _ in range(max_steps):
        player_id = decision.player_id
        action_id, old_log_probability, actor_trainable = (
            rollout_policy.select(
                decision.observation,
                decision.legal_mask,
            )
        )
        privileged_state = (
            env.privileged_state_for_training() if collect_privileged else None
        )
        result = env.step(action_id)
        pending[player_id].append(_PendingSample(
            observation=decision.observation,
            legal_mask_words=pack_legal_mask(decision.legal_mask),
            action_id=action_id,
            old_log_probability=old_log_probability,
            actor_trainable=actor_trainable,
            shanten_reward=result.reward_step.shanten_reward,
            privileged_state=privileged_state,
        ))

        if result.done:
            score_deltas = env.terminal_score_deltas
            component_deltas = settlement_component_deltas(
                env.round_settlement,
            )
            samples = []
            for seat in range(PLAYER_COUNT):
                shanten_returns = monte_carlo_returns(
                    [item.shanten_reward for item in pending[seat]],
                )
                for item, shanten_return in zip(
                    pending[seat],
                    shanten_returns,
                ):
                    samples.append(PPOSample(
                        episode_seed=seed,
                        player_id=seat,
                        observation=item.observation,
                        legal_mask_words=item.legal_mask_words,
                        action_id=item.action_id,
                        old_log_probability=item.old_log_probability,
                        actor_trainable=item.actor_trainable,
                        score_return=score_deltas[seat] / score_divisor,
                        shanten_return=shanten_return / shanten_divisor,
                        component_return=tuple(
                            value / score_divisor
                            for value in component_deltas[seat]
                        ),
                        privileged_state=item.privileged_state,
                    ))
            round_result = env.round_result
            return PPOEpisode(
                seed=seed,
                samples=tuple(samples),
                score_deltas=score_deltas,
                component_deltas=component_deltas,
                end_reason=round_result.reason,
                hu_players=tuple(round_result.hu_players),
            )

        decision = result.next_decision
        if decision is None:
            raise RuntimeError("unfinished rollout has no next decision")

    raise RuntimeError(f"PPO rollout exceeded {max_steps} steps")


__all__ = [
    "PPOEpisode",
    "PPOSample",
    "RolloutPolicy",
    "run_ppo_episode",
]
