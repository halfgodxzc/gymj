import random
from typing import Optional, Sequence, Tuple

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    TILE_TYPE_COUNT,
    CompactObservation,
    calculate_shanten,
)
from .demonstration import TeacherDecision, pack_action_ids
from .env import EnvState, LegalMask


ANKAN_ACTION_START = TILE_TYPE_COUNT
KAKAN_ACTION_START = TILE_TYPE_COUNT * 2
PON_ACTION_ID = TILE_TYPE_COUNT * 3
DAIMINKAN_ACTION_ID = PON_ACTION_ID + 1
TSUMO_ACTION_ID = PON_ACTION_ID + 2
RON_ACTION_ID = PON_ACTION_ID + 3
PASS_ACTION_ID = PON_ACTION_ID + 4
DECLARE_READY_ACTION_ID = PON_ACTION_ID + 5
MAX_HAND_MELDS = 4
SUPPORTED_ACTION_ENCODING_VERSION = 2


class ShantenTeacher:
    """Rule teacher for the first behavior-cloning dataset."""

    def __init__(
        self,
        seed: Optional[int] = None,
        kan_probability: float = 0.5,
        ready_policy: str = "skip",
    ):
        if not 0.0 <= kan_probability <= 1.0:
            raise ValueError("kan probability must be in range 0..1")
        if ready_policy not in {"skip", "declare", "pass"}:
            raise ValueError("ready policy must be skip, declare, or pass")
        self._random = random.Random(seed)
        self._kan_probability = kan_probability
        self._ready_policy = ready_policy

    def decide(self, state: EnvState) -> TeacherDecision:
        return self._decide(state.observation, state.legal_mask)

    def select_action_id(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> int:
        return self._decide(observation, legal_mask).selected_action_id

    def _decide(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> TeacherDecision:
        legal_action_ids = _legal_action_ids(legal_mask)
        if len(legal_action_ids) == 1:
            return _skipped_decision(legal_action_ids[0])

        hu_action_ids = tuple(
            action_id
            for action_id in (TSUMO_ACTION_ID, RON_ACTION_ID)
            if action_id in legal_action_ids
        )
        if hu_action_ids:
            return _trainable_decision(
                selected_action_id=self._random.choice(hu_action_ids),
                supervision_action_ids=legal_action_ids,
                target_action_ids=hu_action_ids,
            )

        if DECLARE_READY_ACTION_ID in legal_action_ids:
            if PASS_ACTION_ID not in legal_action_ids:
                raise ValueError("ready declaration must allow Pass")
            if self._ready_policy == "skip":
                return _skipped_decision(PASS_ACTION_ID)
            target = (
                DECLARE_READY_ACTION_ID
                if self._ready_policy == "declare"
                else PASS_ACTION_ID
            )
            return _trainable_decision(
                selected_action_id=target,
                supervision_action_ids=(PASS_ACTION_ID, DECLARE_READY_ACTION_ID),
                target_action_ids=(target,),
            )

        kan_action_ids = _legal_kan_action_ids(legal_action_ids)
        discard_action_ids = tuple(
            action_id
            for action_id in legal_action_ids
            if 0 <= action_id < TILE_TYPE_COUNT
        )
        if discard_action_ids:
            target_action_ids = self._best_discards(
                observation,
                discard_action_ids,
            )
            selected_action_id = self._random.choice(target_action_ids)
            selected_action_id = self._maybe_select_kan(
                selected_action_id,
                kan_action_ids,
            )
            return _trainable_decision(
                selected_action_id=selected_action_id,
                supervision_action_ids=discard_action_ids,
                target_action_ids=target_action_ids,
            )

        if (
            PON_ACTION_ID in legal_action_ids
            and PASS_ACTION_ID in legal_action_ids
        ):
            target_action_id = self._pon_or_pass(observation)
            selected_action_id = self._maybe_select_kan(
                target_action_id,
                kan_action_ids,
            )
            return _trainable_decision(
                selected_action_id=selected_action_id,
                supervision_action_ids=(PON_ACTION_ID, PASS_ACTION_ID),
                target_action_ids=(target_action_id,),
            )

        if _only_pass_and_kan(legal_action_ids):
            selected_action_id = self._maybe_select_kan(
                PASS_ACTION_ID,
                kan_action_ids,
            )
            return _skipped_decision(selected_action_id)

        raise RuntimeError(
            "shanten teacher cannot evaluate legal actions: "
            f"{legal_action_ids}"
        )

    def _best_discards(
        self,
        observation: CompactObservation,
        discard_action_ids: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        hand_counts = _hand_counts(observation)
        meld_count = _own_meld_count(observation)
        scores = []
        for tile_id in discard_action_ids:
            if hand_counts[tile_id] == 0:
                raise ValueError("legal discard is absent from the hand")
            after_discard = hand_counts.copy()
            after_discard[tile_id] -= 1
            scores.append(
                (
                    tile_id,
                    int(calculate_shanten(after_discard, meld_count).best),
                )
            )

        best_shanten = min(score for _, score in scores)
        return tuple(
            tile_id
            for tile_id, score in scores
            if score == best_shanten
        )

    def _pon_or_pass(self, observation: CompactObservation) -> int:
        hand_counts = _hand_counts(observation)
        meld_count = _own_meld_count(observation)
        pass_shanten = int(calculate_shanten(hand_counts, meld_count).best)

        called_tile = int(observation.current_discard)
        if not 0 <= called_tile < TILE_TYPE_COUNT:
            raise ValueError("Pon decision must have a current discard")
        if hand_counts[called_tile] < 2:
            raise ValueError("Pon decision requires two matching hand tiles")
        if meld_count >= MAX_HAND_MELDS:
            raise ValueError("Pon decision cannot exceed four melds")

        after_pon = hand_counts.copy()
        after_pon[called_tile] -= 2
        post_pon_shanten = min(
            int(calculate_shanten(after_discard, meld_count + 1).best)
            for after_discard in _hands_after_each_discard(after_pon)
        )
        if post_pon_shanten < pass_shanten:
            return PON_ACTION_ID
        return PASS_ACTION_ID

    def _maybe_select_kan(
        self,
        fallback_action_id: int,
        kan_action_ids: Tuple[int, ...],
    ) -> int:
        if not kan_action_ids or self._kan_probability == 0.0:
            return fallback_action_id
        if (
            self._kan_probability == 1.0
            or self._random.random() < self._kan_probability
        ):
            return self._random.choice(kan_action_ids)
        return fallback_action_id


def _legal_action_ids(legal_mask: Sequence[int]) -> Tuple[int, ...]:
    if len(legal_mask) != ACTION_SPACE_SIZE:
        raise ValueError(
            f"legal mask must contain {ACTION_SPACE_SIZE} entries"
        )
    if any(value not in (0, 1, False, True) for value in legal_mask):
        raise ValueError("legal mask entries must be zero or one")

    legal_action_ids = tuple(
        action_id
        for action_id, is_legal in enumerate(legal_mask)
        if is_legal
    )
    if not legal_action_ids:
        raise ValueError("legal mask contains no legal actions")
    return legal_action_ids


def _legal_kan_action_ids(
    legal_action_ids: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(
        action_id
        for action_id in legal_action_ids
        if (
            ANKAN_ACTION_START <= action_id < PON_ACTION_ID
            or action_id == DAIMINKAN_ACTION_ID
        )
    )


def _only_pass_and_kan(legal_action_ids: Tuple[int, ...]) -> bool:
    return (
        PASS_ACTION_ID in legal_action_ids
        and all(
            action_id == PASS_ACTION_ID
            or action_id == DAIMINKAN_ACTION_ID
            or ANKAN_ACTION_START <= action_id < PON_ACTION_ID
            for action_id in legal_action_ids
        )
    )


def _hand_counts(observation: CompactObservation) -> list[int]:
    counts = [int(count) for count in observation.hand_counts]
    if len(counts) != TILE_TYPE_COUNT:
        raise ValueError(
            f"hand counts must contain {TILE_TYPE_COUNT} entries"
        )
    return counts


def _own_meld_count(observation: CompactObservation) -> int:
    public_meld_count = int(observation.meld_count)
    melds = observation.melds
    if not 0 <= public_meld_count <= len(melds):
        raise ValueError("observation meld count is out of range")

    own_meld_count = sum(
        1
        for meld in melds[:public_meld_count]
        if int(meld.valid) and int(meld.relative_owner) == 0
    )
    if own_meld_count > MAX_HAND_MELDS:
        raise ValueError("player cannot have more than four melds")
    return own_meld_count


def _hands_after_each_discard(
    hand_counts: list[int],
) -> Tuple[list[int], ...]:
    hands = []
    for tile_id, count in enumerate(hand_counts):
        if count == 0:
            continue
        after_discard = hand_counts.copy()
        after_discard[tile_id] -= 1
        hands.append(after_discard)
    if not hands:
        raise ValueError("Pon must leave at least one discardable tile")
    return tuple(hands)


def _trainable_decision(
    selected_action_id: int,
    supervision_action_ids: Sequence[int],
    target_action_ids: Sequence[int],
) -> TeacherDecision:
    return TeacherDecision(
        selected_action_id=selected_action_id,
        supervision_mask_words=pack_action_ids(supervision_action_ids),
        target_action_mask_words=pack_action_ids(target_action_ids),
        trainable=True,
    )


def _skipped_decision(selected_action_id: int) -> TeacherDecision:
    return TeacherDecision(
        selected_action_id=selected_action_id,
        supervision_mask_words=(0, 0),
        target_action_mask_words=(0, 0),
        trainable=False,
    )


if (
    ACTION_ENCODING_VERSION != SUPPORTED_ACTION_ENCODING_VERSION
    or DECLARE_READY_ACTION_ID + 1 != ACTION_SPACE_SIZE
):
    raise RuntimeError("teacher action constants do not match action encoding")
