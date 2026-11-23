from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ._gymj import ActionType


PLAYER_COUNT = 4
DEFAULT_SHANTEN_REWARD_SCALE = 10.0
DEFAULT_HU_REWARD = 100.0
RewardVector = Tuple[float, float, float, float]


def calculate_discard_reward(
    before_shanten: int,
    after_shanten: int,
    scale: float = DEFAULT_SHANTEN_REWARD_SCALE,
) -> float:
    """Return the immediate reward for one discard."""
    if scale < 0:
        raise ValueError("shanten reward scale must be non-negative")
    return float(scale * (before_shanten - after_shanten))


def calculate_hu_rewards(
    hu_players: Sequence[int],
    hu_reward: float = DEFAULT_HU_REWARD,
) -> RewardVector:
    """Return the non-zero-sum training bonus for terminal Hu players."""
    if hu_reward < 0:
        raise ValueError("Hu reward must be non-negative")

    rewards = [0.0] * PLAYER_COUNT
    for player_id in hu_players:
        if not 0 <= player_id < PLAYER_COUNT:
            raise ValueError("Hu player must be in range 0..3")
        rewards[player_id] += hu_reward
    return tuple(rewards)


@dataclass(frozen=True)
class RewardStep:
    rewards: RewardVector
    action_actor: int
    before_shanten: Optional[int]
    after_shanten: Optional[int]
    shanten_reward: float
    hu_players: Tuple[int, ...]


class ShantenRewardTracker:
    """Submit an Arena action and measure its first-stage training reward."""

    def __init__(
        self,
        shanten_scale: float = DEFAULT_SHANTEN_REWARD_SCALE,
        hu_reward: float = DEFAULT_HU_REWARD,
    ):
        if shanten_scale < 0:
            raise ValueError("shanten reward scale must be non-negative")
        if hu_reward < 0:
            raise ValueError("Hu reward must be non-negative")
        self.shanten_scale = shanten_scale
        self.hu_reward = hu_reward

    def step(self, arena, action) -> RewardStep:
        actor = int(action.actor)
        if not 0 <= actor < PLAYER_COUNT:
            raise ValueError("action actor must be in range 0..3")

        before_shanten = None
        if action.type == ActionType.DISCARD:
            before_shanten = int(arena.player(actor).shanten)

        was_finished = bool(arena.finished)
        arena.submit_action(action)

        rewards = [0.0] * PLAYER_COUNT
        after_shanten = None
        shanten_reward = 0.0
        if before_shanten is not None:
            after_shanten = int(arena.player(actor).shanten)
            shanten_reward = calculate_discard_reward(
                before_shanten,
                after_shanten,
                self.shanten_scale,
            )
            rewards[actor] += shanten_reward

        hu_players = ()
        if not was_finished and arena.finished:
            hu_players = tuple(int(player_id) for player_id in arena.result.hu_players)
            terminal_rewards = calculate_hu_rewards(hu_players, self.hu_reward)
            rewards = [
                reward + terminal_reward
                for reward, terminal_reward in zip(rewards, terminal_rewards)
            ]

        return RewardStep(
            rewards=tuple(rewards),
            action_actor=actor,
            before_shanten=before_shanten,
            after_shanten=after_shanten,
            shanten_reward=shanten_reward,
            hu_players=hu_players,
        )
