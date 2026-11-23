from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ._gymj import ACTION_SPACE_SIZE, CompactObservation
from .env import EnvState, StepResult
from .reward import PLAYER_COUNT


PackedLegalMask = Tuple[int, int]


def pack_legal_mask(legal_mask: Sequence[int]) -> PackedLegalMask:
    if len(legal_mask) != ACTION_SPACE_SIZE:
        raise ValueError(
            f"legal mask must contain {ACTION_SPACE_SIZE} entries"
        )

    words = [0, 0]
    for action_id, is_legal in enumerate(legal_mask):
        if is_legal not in (0, 1):
            raise ValueError("legal mask entries must be 0 or 1")
        if is_legal:
            words[action_id // 64] |= 1 << (action_id % 64)
    return words[0], words[1]


def is_action_legal(mask_words: PackedLegalMask, action_id: int) -> bool:
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        return False
    return bool(
        (mask_words[action_id // 64] >> (action_id % 64)) & 1
    )


@dataclass
class Transition:
    player_id: int
    observation: CompactObservation
    legal_mask_words: PackedLegalMask
    action_id: int
    reward: float = 0.0
    done: bool = False


class TrajectoryCollector:
    def __init__(self):
        self._trajectories: List[List[Transition]] = [
            [] for _ in range(PLAYER_COUNT)
        ]
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def trajectory_for(self, player_id: int) -> Tuple[Transition, ...]:
        self._validate_player(player_id)
        return tuple(self._trajectories[player_id])

    def reset(self) -> None:
        for trajectory in self._trajectories:
            trajectory.clear()
        self._finished = False

    def record(
        self,
        decision: EnvState,
        result: StepResult,
    ) -> Transition:
        if self._finished:
            raise RuntimeError("cannot record after the trajectory is finished")
        if decision.player_id != result.actor_id:
            raise ValueError("decision player does not match action actor")
        if result.reward_step.action_actor != result.actor_id:
            raise ValueError("reward actor does not match action actor")
        if result.done != (result.next_decision is None):
            raise ValueError("done and next decision are inconsistent")

        actor_id = result.actor_id
        self._validate_player(actor_id)
        mask_words = pack_legal_mask(decision.legal_mask)
        if not is_action_legal(mask_words, result.action_id):
            raise ValueError("recorded action is not legal in its decision")

        for player_id, reward in enumerate(result.rewards):
            if reward != 0 and player_id != actor_id:
                if not self._trajectories[player_id]:
                    raise RuntimeError(
                        "cannot assign reward to a player without a transition"
                    )

        transition = Transition(
            player_id=actor_id,
            observation=decision.observation,
            legal_mask_words=mask_words,
            action_id=result.action_id,
        )
        self._trajectories[actor_id].append(transition)

        for player_id, reward in enumerate(result.rewards):
            if reward != 0:
                self._trajectories[player_id][-1].reward += reward

        if result.done:
            for trajectory in self._trajectories:
                if trajectory:
                    trajectory[-1].done = True
            self._finished = True

        return transition

    @staticmethod
    def _validate_player(player_id: int) -> None:
        if not 0 <= player_id < PLAYER_COUNT:
            raise ValueError("player id must be in range 0..3")
