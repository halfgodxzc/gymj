from dataclasses import dataclass
from typing import Iterable, Tuple

from ._gymj import ACTION_SPACE_SIZE, CompactObservation
from .env import EnvState
from .reward import PLAYER_COUNT
from .trajectory import PackedLegalMask, pack_legal_mask


MASK_WORD_BITS = 64
MASK_WORD_COUNT = 2


def pack_action_ids(action_ids: Iterable[int]) -> PackedLegalMask:
    words = [0] * MASK_WORD_COUNT
    for action_id in action_ids:
        _validate_action_id(action_id)
        words[action_id // MASK_WORD_BITS] |= (
            1 << (action_id % MASK_WORD_BITS)
        )
    return words[0], words[1]


def action_ids_from_mask(
    mask_words: PackedLegalMask,
) -> Tuple[int, ...]:
    _validate_mask_words("action mask", mask_words)
    return tuple(
        action_id
        for action_id in range(ACTION_SPACE_SIZE)
        if _mask_contains(mask_words, action_id)
    )


@dataclass(frozen=True)
class TeacherDecision:
    selected_action_id: int
    supervision_mask_words: PackedLegalMask
    target_action_mask_words: PackedLegalMask
    trainable: bool

    def __post_init__(self) -> None:
        _validate_action_id(self.selected_action_id)
        _validate_mask_words(
            "supervision mask",
            self.supervision_mask_words,
        )
        _validate_mask_words(
            "target action mask",
            self.target_action_mask_words,
        )
        if not _is_subset(
            self.target_action_mask_words,
            self.supervision_mask_words,
        ):
            raise ValueError(
                "target actions must be a subset of supervised actions"
            )

        has_targets = any(self.target_action_mask_words)
        if self.trainable != has_targets:
            raise ValueError(
                "trainable decisions must have targets and skipped "
                "decisions must not"
            )

    @property
    def supervised_action_ids(self) -> Tuple[int, ...]:
        return action_ids_from_mask(self.supervision_mask_words)

    @property
    def target_action_ids(self) -> Tuple[int, ...]:
        return action_ids_from_mask(self.target_action_mask_words)


@dataclass(frozen=True)
class DemonstrationSample:
    episode_seed: int
    player_id: int
    observation: CompactObservation
    legal_mask_words: PackedLegalMask
    supervision_mask_words: PackedLegalMask
    target_action_mask_words: PackedLegalMask
    selected_action_id: int

    def __post_init__(self) -> None:
        if not -(1 << 63) <= self.episode_seed < (1 << 63):
            raise ValueError("episode seed must fit in int64")
        if not 0 <= self.player_id < PLAYER_COUNT:
            raise ValueError("player id must be in range 0..3")
        _validate_action_id(self.selected_action_id)
        _validate_mask_words("legal mask", self.legal_mask_words)
        _validate_mask_words(
            "supervision mask",
            self.supervision_mask_words,
        )
        _validate_mask_words(
            "target action mask",
            self.target_action_mask_words,
        )

        if not _mask_contains(
            self.legal_mask_words,
            self.selected_action_id,
        ):
            raise ValueError("selected action must be legal")
        if not _is_subset(
            self.supervision_mask_words,
            self.legal_mask_words,
        ):
            raise ValueError(
                "supervised actions must be a subset of legal actions"
            )
        if not _is_subset(
            self.target_action_mask_words,
            self.supervision_mask_words,
        ):
            raise ValueError(
                "target actions must be a subset of supervised actions"
            )
        if not any(self.target_action_mask_words):
            raise ValueError("demonstration sample must contain a target")

        observation_mask = tuple(
            int(word) for word in self.observation.legal_mask
        )
        if observation_mask != self.legal_mask_words:
            raise ValueError(
                "sample legal mask must match the observation legal mask"
            )

    @property
    def target_action_ids(self) -> Tuple[int, ...]:
        return action_ids_from_mask(self.target_action_mask_words)

    @classmethod
    def from_teacher_decision(
        cls,
        episode_seed: int,
        state: EnvState,
        decision: TeacherDecision,
    ) -> "DemonstrationSample":
        if not decision.trainable:
            raise ValueError(
                "skipped teacher decisions cannot become demonstrations"
            )
        return cls(
            episode_seed=episode_seed,
            player_id=state.player_id,
            observation=state.observation,
            legal_mask_words=pack_legal_mask(state.legal_mask),
            supervision_mask_words=decision.supervision_mask_words,
            target_action_mask_words=decision.target_action_mask_words,
            selected_action_id=decision.selected_action_id,
        )


def _validate_action_id(action_id: int) -> None:
    if not isinstance(action_id, int):
        raise TypeError("action id must be an integer")
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError(
            f"action id must be in range 0..{ACTION_SPACE_SIZE - 1}"
        )


def _validate_mask_words(
    name: str,
    mask_words: PackedLegalMask,
) -> None:
    if len(mask_words) != MASK_WORD_COUNT:
        raise ValueError(f"{name} must contain exactly two words")
    for word in mask_words:
        if not isinstance(word, int):
            raise TypeError(f"{name} words must be integers")
        if not 0 <= word < (1 << MASK_WORD_BITS):
            raise ValueError(f"{name} words must fit in uint64")

    used_high_bits = ACTION_SPACE_SIZE - MASK_WORD_BITS
    if mask_words[1] >> used_high_bits:
        raise ValueError(f"{name} contains actions outside the action space")


def _mask_contains(
    mask_words: PackedLegalMask,
    action_id: int,
) -> bool:
    word = action_id // MASK_WORD_BITS
    bit = action_id % MASK_WORD_BITS
    return bool((mask_words[word] >> bit) & 1)


def _is_subset(
    subset: PackedLegalMask,
    superset: PackedLegalMask,
) -> bool:
    return all(
        subset_word & ~superset_word == 0
        for subset_word, superset_word in zip(subset, superset)
    )
