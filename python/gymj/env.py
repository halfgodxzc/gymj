from dataclasses import dataclass
from typing import Optional, Tuple

from ._gymj import (
    ACTION_SPACE_SIZE,
    ActionEncoder,
    Arena,
    CompactObservation,
    EncodedDecision,
    ObservationEncoder,
    PrivilegedStateEncoder,
    RoundResult,
    RoundSettlementResult,
)
from .reward import RewardStep, RewardVector, ShantenRewardTracker


LegalMask = Tuple[int, ...]


@dataclass(frozen=True)
class EnvState:
    player_id: int
    observation: CompactObservation
    legal_mask: LegalMask


@dataclass(frozen=True)
class StepResult:
    actor_id: int
    action_id: int
    reward_step: RewardStep
    done: bool
    next_decision: Optional[EnvState]

    @property
    def rewards(self) -> RewardVector:
        return self.reward_step.rewards


class GymjEnv:
    def __init__(self, dealer: int = 0, log_path: str = ""):
        self.dealer = dealer
        self.log_path = log_path
        self._arena: Optional[Arena] = None
        self._encoded_decision: Optional[EncodedDecision] = None
        self._reward_tracker = ShantenRewardTracker()

    @property
    def round_result(self) -> RoundResult:
        if self._arena is None:
            raise RuntimeError("environment must be reset before reading result")
        if not self._arena.finished:
            raise RuntimeError("round result is only available after completion")
        return self._arena.result

    @property
    def terminal_score_deltas(self) -> Tuple[int, int, int, int]:
        if self._arena is None:
            raise RuntimeError("environment must be reset before reading scores")
        if not self._arena.finished:
            raise RuntimeError("scores are only available after completion")
        return tuple(
            int(value)
            for value in self._arena.round_settlement.ledger.score_deltas
        )

    @property
    def round_settlement(self) -> RoundSettlementResult:
        if self._arena is None:
            raise RuntimeError("environment must be reset before reading settlement")
        if not self._arena.finished:
            raise RuntimeError("settlement is only available after completion")
        return self._arena.round_settlement

    def reset(self, seed: int = 0) -> EnvState:
        self._encoded_decision = None
        self._arena = Arena(
            seed=seed,
            dealer=self.dealer,
            log_path=self.log_path,
        )
        self._arena.start_round()
        return self._current_state()

    def privileged_state_for_training(self):
        if self._arena is None or self._arena.finished:
            raise RuntimeError("privileged state requires an active round")
        return PrivilegedStateEncoder.encode(
            self._arena, self._arena.current_decision().player,
        )

    def action_for_id(self, action_id: int):
        """Return the pending legal Action without submitting it."""
        if self._arena is None:
            raise RuntimeError("environment must be reset before reading action")
        if self._arena.finished:
            raise RuntimeError("finished environment has no pending action")
        encoded = self._encoded_decision
        if encoded is None:
            raise RuntimeError("environment has no pending decision")
        if not isinstance(action_id, int):
            raise TypeError("action id must be an integer")
        if not 0 <= action_id < ACTION_SPACE_SIZE:
            raise ValueError(
                f"action id must be in range 0..{ACTION_SPACE_SIZE - 1}"
            )
        if not encoded.is_legal(action_id):
            raise ValueError("action id is not legal in the current decision")
        return encoded.action_at(action_id)

    def _current_state(self) -> EnvState:
        if self._arena is None:
            raise RuntimeError("environment must be reset before reading state")
        if self._arena.finished:
            raise RuntimeError("finished environment has no current state")

        decision = self._arena.current_decision()
        observation = ObservationEncoder.encode(
            self._arena,
            decision.player,
        )
        encoded = ActionEncoder.encode_legal_actions(
            decision.legal_actions
        )
        self._encoded_decision = encoded

        return EnvState(
            player_id=decision.player,
            observation=observation,
            legal_mask=tuple(encoded.legal_mask),
        )

    def step(self, action_id: int) -> StepResult:
        if self._arena is None:
            raise RuntimeError("environment must be reset before step")
        if self._arena.finished:
            raise RuntimeError("cannot step a finished environment")

        encoded = self._encoded_decision
        if encoded is None:
            raise RuntimeError("environment has no pending decision")
        if not isinstance(action_id, int):
            raise TypeError("action id must be an integer")
        if not 0 <= action_id < ACTION_SPACE_SIZE:
            raise ValueError(
                f"action id must be in range 0..{ACTION_SPACE_SIZE - 1}"
            )
        if not encoded.is_legal(action_id):
            raise ValueError("action id is not legal in the current decision")

        action = self.action_for_id(action_id)
        actor_id = int(action.actor)
        reward_step = self._reward_tracker.step(self._arena, action)

        self._encoded_decision = None
        done = bool(self._arena.finished)
        next_decision = None if done else self._current_state()

        return StepResult(
            actor_id=actor_id,
            action_id=action_id,
            reward_step=reward_step,
            done=done,
            next_decision=next_decision,
        )
