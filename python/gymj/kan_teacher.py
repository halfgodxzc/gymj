import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

from ._gymj import (
    ACTION_SPACE_SIZE,
    TILE_TYPE_COUNT,
    CompactObservation,
    MeldType,
    calculate_shanten,
)
from .agent import Agent
from .env import EnvState
from .teacher import (
    ANKAN_ACTION_START,
    DAIMINKAN_ACTION_ID,
    KAKAN_ACTION_START,
    PASS_ACTION_ID,
    PON_ACTION_ID,
    RON_ACTION_ID,
    TSUMO_ACTION_ID,
    ShantenTeacher,
)


MAX_TILE_COPIES = 4
MAX_HAND_MELDS = 4


class KanKind(str, Enum):
    ANKAN = "ankan"
    KAKAN = "kakan"
    DAIMINKAN = "daiminkan"


class KanConfidence(str, Enum):
    HIGH = "high"
    EXPLORATORY = "exploratory"
    LOW = "low"


@dataclass(frozen=True)
class ReplacementDrawOutcome:
    tile_id: int
    unseen_copies: int
    probability: float
    best_shanten: int
    best_discard_ids: Tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "unseen_copies": self.unseen_copies,
            "probability": self.probability,
            "best_shanten": self.best_shanten,
            "best_discard_ids": list(self.best_discard_ids),
        }


@dataclass(frozen=True)
class KanEvaluation:
    action_id: int
    kind: KanKind
    tile_id: int
    fallback_action_id: int
    baseline_shanten: int
    expected_shanten_after_kan: float
    improve_probability: float
    same_probability: float
    worse_probability: float
    expected_shanten_delta: float
    unseen_counts: Tuple[int, ...]
    outcomes: Tuple[ReplacementDrawOutcome, ...]
    confidence: KanConfidence
    can_supervise: bool

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "tile_id": self.tile_id,
            "fallback_action_id": self.fallback_action_id,
            "baseline_shanten": self.baseline_shanten,
            "expected_shanten_after_kan": self.expected_shanten_after_kan,
            "improve_probability": self.improve_probability,
            "same_probability": self.same_probability,
            "worse_probability": self.worse_probability,
            "expected_shanten_delta": self.expected_shanten_delta,
            "unseen_counts": list(self.unseen_counts),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "confidence": self.confidence.value,
            "can_supervise": self.can_supervise,
        }


@dataclass(frozen=True)
class KanDecision:
    fallback_action_id: int
    selected_action_id: int
    kan_selection_probability: float
    selected_evaluation: Optional[KanEvaluation]
    evaluations: Tuple[KanEvaluation, ...]
    can_supervise: bool

    @property
    def selected_kan(self) -> bool:
        return self.selected_evaluation is not None

    def to_dict(self) -> dict:
        return {
            "fallback_action_id": self.fallback_action_id,
            "selected_action_id": self.selected_action_id,
            "selected_kan": self.selected_kan,
            "kan_selection_probability": self.kan_selection_probability,
            "selected_evaluation": (
                None
                if self.selected_evaluation is None
                else self.selected_evaluation.to_dict()
            ),
            "evaluations": [
                evaluation.to_dict() for evaluation in self.evaluations
            ],
            "can_supervise": self.can_supervise,
        }


class KanHeuristicEvaluator:
    """Evaluate legal kans using only information visible to the player."""

    def __init__(
        self,
        *,
        high_max_worse_probability: float = 0.10,
        exploratory_min_improve_probability: float = 0.20,
        exploratory_max_worse_probability: float = 0.30,
    ):
        _validate_probability(
            "high max worse probability",
            high_max_worse_probability,
        )
        _validate_probability(
            "exploratory min improve probability",
            exploratory_min_improve_probability,
        )
        _validate_probability(
            "exploratory max worse probability",
            exploratory_max_worse_probability,
        )
        if (
            high_max_worse_probability
            > exploratory_max_worse_probability
        ):
            raise ValueError(
                "high-confidence risk cannot exceed exploratory risk"
            )

        self._high_max_worse_probability = (
            high_max_worse_probability
        )
        self._exploratory_min_improve_probability = (
            exploratory_min_improve_probability
        )
        self._exploratory_max_worse_probability = (
            exploratory_max_worse_probability
        )

    def evaluate_state(
        self,
        state: EnvState,
        fallback_action_id: int,
    ) -> Tuple[KanEvaluation, ...]:
        return self.evaluate(
            state.observation,
            state.legal_mask,
            fallback_action_id,
        )

    def evaluate_best_shanten_baseline(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> Tuple[int, Tuple[KanEvaluation, ...]]:
        """Compare legal kans with the best non-kan shanten alternative."""
        legal_action_ids = _legal_action_ids(legal_mask)
        hu_actions = tuple(
            action_id
            for action_id in legal_action_ids
            if action_id in (TSUMO_ACTION_ID, RON_ACTION_ID)
        )
        if hu_actions:
            return min(hu_actions), ()

        kan_actions = tuple(
            action_id
            for action_id in legal_action_ids
            if _is_kan_action(action_id)
        )
        baseline_actions = tuple(
            action_id
            for action_id in legal_action_ids
            if not _is_kan_action(action_id)
            and (
                0 <= action_id < TILE_TYPE_COUNT
                or action_id in (PASS_ACTION_ID, PON_ACTION_ID)
            )
        )
        if not baseline_actions:
            raise ValueError("kan evaluation has no non-kan baseline action")

        hand = _hand_counts(observation)
        meld_count = len(_own_melds(observation))
        fallback_action_id = min(
            baseline_actions,
            key=lambda action_id: (
                _fallback_shanten(
                    observation,
                    hand,
                    meld_count,
                    action_id,
                ),
                action_id,
            ),
        )
        if not kan_actions:
            return fallback_action_id, ()
        return (
            fallback_action_id,
            self.evaluate(
                observation,
                legal_mask,
                fallback_action_id,
            ),
        )

    def evaluate(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
        fallback_action_id: int,
    ) -> Tuple[KanEvaluation, ...]:
        legal_action_ids = _legal_action_ids(legal_mask)
        if fallback_action_id not in legal_action_ids:
            raise ValueError("fallback action must be legal")
        if _is_kan_action(fallback_action_id):
            raise ValueError("fallback action must not already be a kan")

        kan_action_ids = tuple(
            action_id
            for action_id in legal_action_ids
            if _is_kan_action(action_id)
        )
        if not kan_action_ids:
            return ()

        hand = _hand_counts(observation)
        visible = _visible_counts(observation, hand)
        own_melds = _own_melds(observation)
        baseline_shanten = _fallback_shanten(
            observation,
            hand,
            len(own_melds),
            fallback_action_id,
        )
        unseen_counts = tuple(
            MAX_TILE_COPIES - count for count in visible
        )
        unseen_total = sum(unseen_counts)
        if unseen_total <= 0:
            raise ValueError("kan evaluation has no unseen replacement tiles")

        return tuple(
            self._evaluate_kan(
                observation=observation,
                hand=hand,
                own_melds=own_melds,
                visible=visible,
                unseen_counts=unseen_counts,
                unseen_total=unseen_total,
                fallback_action_id=fallback_action_id,
                baseline_shanten=baseline_shanten,
                action_id=action_id,
            )
            for action_id in kan_action_ids
        )

    def _evaluate_kan(
        self,
        *,
        observation: CompactObservation,
        hand: list[int],
        own_melds: Tuple[object, ...],
        visible: list[int],
        unseen_counts: Tuple[int, ...],
        unseen_total: int,
        fallback_action_id: int,
        baseline_shanten: int,
        action_id: int,
    ) -> KanEvaluation:
        kind, tile_id = _decode_kan(action_id, observation)
        after_kan, post_kan_meld_count = _apply_kan(
            hand,
            own_melds,
            kind,
            tile_id,
        )

        outcomes = []
        for draw_tile, unseen_copies in enumerate(unseen_counts):
            if unseen_copies == 0:
                continue
            after_draw = after_kan.copy()
            after_draw[draw_tile] += 1
            best_shanten, best_discards = _best_discard_shanten(
                after_draw,
                post_kan_meld_count,
            )
            outcomes.append(ReplacementDrawOutcome(
                tile_id=draw_tile,
                unseen_copies=unseen_copies,
                probability=unseen_copies / unseen_total,
                best_shanten=best_shanten,
                best_discard_ids=best_discards,
            ))

        expected_after = sum(
            outcome.probability * outcome.best_shanten
            for outcome in outcomes
        )
        improve_probability = sum(
            outcome.probability
            for outcome in outcomes
            if outcome.best_shanten < baseline_shanten
        )
        same_probability = sum(
            outcome.probability
            for outcome in outcomes
            if outcome.best_shanten == baseline_shanten
        )
        worse_probability = sum(
            outcome.probability
            for outcome in outcomes
            if outcome.best_shanten > baseline_shanten
        )
        expected_delta = baseline_shanten - expected_after
        confidence = self._confidence(
            improve_probability,
            worse_probability,
            expected_delta,
        )

        return KanEvaluation(
            action_id=action_id,
            kind=kind,
            tile_id=tile_id,
            fallback_action_id=fallback_action_id,
            baseline_shanten=baseline_shanten,
            expected_shanten_after_kan=expected_after,
            improve_probability=improve_probability,
            same_probability=same_probability,
            worse_probability=worse_probability,
            expected_shanten_delta=expected_delta,
            unseen_counts=unseen_counts,
            outcomes=tuple(outcomes),
            confidence=confidence,
            can_supervise=confidence == KanConfidence.HIGH,
        )

    def _confidence(
        self,
        improve_probability: float,
        worse_probability: float,
        expected_delta: float,
    ) -> KanConfidence:
        if (
            worse_probability <= self._high_max_worse_probability
            and expected_delta >= 0.0
        ):
            return KanConfidence.HIGH
        if (
            improve_probability
            >= self._exploratory_min_improve_probability
            and worse_probability
            <= self._exploratory_max_worse_probability
        ):
            return KanConfidence.EXPLORATORY
        return KanConfidence.LOW


class KanExplorationPolicy:
    """Turn heuristic confidence into reproducible exploration decisions."""

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        high_probability: float = 0.80,
        exploratory_probability: float = 0.35,
        low_probability: float = 0.05,
    ):
        _validate_probability("high selection probability", high_probability)
        _validate_probability(
            "exploratory selection probability",
            exploratory_probability,
        )
        _validate_probability("low selection probability", low_probability)
        if not high_probability >= exploratory_probability >= low_probability:
            raise ValueError(
                "kan selection probabilities must decrease with confidence"
            )
        self._random = random.Random(seed)
        self._selection_probabilities = {
            KanConfidence.HIGH: high_probability,
            KanConfidence.EXPLORATORY: exploratory_probability,
            KanConfidence.LOW: low_probability,
        }

    def reset(self, seed: int) -> None:
        self._random.seed(seed)

    def select(
        self,
        fallback_action_id: int,
        evaluations: Sequence[KanEvaluation],
    ) -> KanDecision:
        candidates = tuple(evaluations)
        if not candidates:
            return KanDecision(
                fallback_action_id=fallback_action_id,
                selected_action_id=fallback_action_id,
                kan_selection_probability=0.0,
                selected_evaluation=None,
                evaluations=(),
                can_supervise=False,
            )

        best = max(
            candidates,
            key=lambda item: (
                item.expected_shanten_delta,
                -item.worse_probability,
                item.improve_probability,
                -item.action_id,
            ),
        )
        selection_probability = self._selection_probabilities[
            best.confidence
        ]
        selected = (
            selection_probability == 1.0
            or self._random.random() < selection_probability
        )
        return KanDecision(
            fallback_action_id=fallback_action_id,
            selected_action_id=(
                best.action_id if selected else fallback_action_id
            ),
            kan_selection_probability=selection_probability,
            selected_evaluation=best if selected else None,
            evaluations=candidates,
            can_supervise=selected and best.can_supervise,
        )


class KanHeuristicAgent:
    """Add auditable kan exploration to a non-kan base policy."""

    def __init__(
        self,
        base_agent: Optional[Agent] = None,
        *,
        seed: Optional[int] = None,
        evaluator: Optional[KanHeuristicEvaluator] = None,
        policy: Optional[KanExplorationPolicy] = None,
    ):
        self._base_agent = (
            base_agent
            if base_agent is not None
            else ShantenTeacher(seed=seed, kan_probability=0.0)
        )
        self._evaluator = evaluator or KanHeuristicEvaluator()
        self._policy = policy or KanExplorationPolicy(seed=seed)
        self._last_decision: Optional[KanDecision] = None

    @property
    def last_decision(self) -> Optional[KanDecision]:
        return self._last_decision

    def reset_episode(self, seed: int) -> None:
        reset_base = getattr(self._base_agent, "reset_episode", None)
        if callable(reset_base):
            reset_base(seed)
        self._policy.reset(seed)
        self._last_decision = None

    def select_with_audit(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> KanDecision:
        fallback_action_id = self._base_agent.select_action_id(
            observation,
            legal_mask,
        )
        if _is_kan_action(fallback_action_id):
            raise ValueError(
                "the base agent must have independent kan selection disabled"
            )

        # A legal hu is terminal and must never be replaced by exploration.
        if fallback_action_id in (TSUMO_ACTION_ID, RON_ACTION_ID):
            decision = KanDecision(
                fallback_action_id=fallback_action_id,
                selected_action_id=fallback_action_id,
                kan_selection_probability=0.0,
                selected_evaluation=None,
                evaluations=(),
                can_supervise=False,
            )
        else:
            evaluations = self._evaluator.evaluate(
                observation,
                legal_mask,
                fallback_action_id,
            )
            decision = self._policy.select(
                fallback_action_id,
                evaluations,
            )
        self._last_decision = decision
        return decision

    def select_action_id(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> int:
        return self.select_with_audit(
            observation,
            legal_mask,
        ).selected_action_id


def _legal_action_ids(legal_mask: Sequence[int]) -> Tuple[int, ...]:
    if len(legal_mask) != ACTION_SPACE_SIZE:
        raise ValueError(
            f"legal mask must contain {ACTION_SPACE_SIZE} entries"
        )
    if any(value not in (0, 1, False, True) for value in legal_mask):
        raise ValueError("legal mask entries must be zero or one")
    actions = tuple(
        action_id
        for action_id, is_legal in enumerate(legal_mask)
        if is_legal
    )
    if not actions:
        raise ValueError("legal mask contains no legal actions")
    return actions


def _hand_counts(observation: CompactObservation) -> list[int]:
    counts = [int(count) for count in observation.hand_counts]
    if len(counts) != TILE_TYPE_COUNT:
        raise ValueError("hand counts must contain 27 entries")
    if any(not 0 <= count <= MAX_TILE_COPIES for count in counts):
        raise ValueError("hand count exceeds physical tile copies")
    return counts


def _visible_counts(
    observation: CompactObservation,
    hand: Sequence[int],
) -> list[int]:
    counts = [int(count) for count in observation.visible_counts]
    if len(counts) != TILE_TYPE_COUNT:
        raise ValueError("visible counts must contain 27 entries")
    for tile_id, count in enumerate(counts):
        if not 0 <= count <= MAX_TILE_COPIES:
            raise ValueError("visible count exceeds physical tile copies")
        if count < hand[tile_id]:
            raise ValueError("visible count cannot be lower than hand count")
    return counts


def _own_melds(observation: CompactObservation) -> Tuple[object, ...]:
    public_meld_count = int(observation.meld_count)
    melds = observation.melds
    if not 0 <= public_meld_count <= len(melds):
        raise ValueError("observation meld count is out of range")
    own = tuple(
        meld
        for meld in melds[:public_meld_count]
        if int(meld.valid) and int(meld.relative_owner) == 0
    )
    if len(own) > MAX_HAND_MELDS:
        raise ValueError("player cannot have more than four melds")
    return own


def _fallback_shanten(
    observation: CompactObservation,
    hand: list[int],
    meld_count: int,
    fallback_action_id: int,
) -> int:
    if 0 <= fallback_action_id < TILE_TYPE_COUNT:
        if hand[fallback_action_id] == 0:
            raise ValueError("fallback discard is absent from the hand")
        after_discard = hand.copy()
        after_discard[fallback_action_id] -= 1
        return int(calculate_shanten(after_discard, meld_count).best)

    if fallback_action_id == PASS_ACTION_ID:
        return int(calculate_shanten(hand, meld_count).best)

    if fallback_action_id == PON_ACTION_ID:
        called_tile = int(observation.current_discard)
        if not 0 <= called_tile < TILE_TYPE_COUNT:
            raise ValueError("Pon fallback requires a current discard")
        if hand[called_tile] < 2 or meld_count >= MAX_HAND_MELDS:
            raise ValueError("Pon fallback has an invalid hand")
        after_pon = hand.copy()
        after_pon[called_tile] -= 2
        best, _ = _best_discard_shanten(after_pon, meld_count + 1)
        return best

    raise ValueError(
        "kan comparison requires a discard, Pon, or Pass fallback"
    )


def _decode_kan(
    action_id: int,
    observation: CompactObservation,
) -> Tuple[KanKind, int]:
    if ANKAN_ACTION_START <= action_id < KAKAN_ACTION_START:
        return KanKind.ANKAN, action_id - ANKAN_ACTION_START
    if KAKAN_ACTION_START <= action_id < PON_ACTION_ID:
        return KanKind.KAKAN, action_id - KAKAN_ACTION_START
    if action_id == DAIMINKAN_ACTION_ID:
        tile_id = int(observation.current_discard)
        if not 0 <= tile_id < TILE_TYPE_COUNT:
            raise ValueError("Daiminkan requires a current discard")
        return KanKind.DAIMINKAN, tile_id
    raise ValueError("action is not a kan")


def _apply_kan(
    hand: list[int],
    own_melds: Tuple[object, ...],
    kind: KanKind,
    tile_id: int,
) -> Tuple[list[int], int]:
    result = hand.copy()
    meld_count = len(own_melds)
    if kind == KanKind.ANKAN:
        if result[tile_id] != MAX_TILE_COPIES:
            raise ValueError("AnKan requires four matching concealed tiles")
        if meld_count >= MAX_HAND_MELDS:
            raise ValueError("AnKan cannot exceed four melds")
        result[tile_id] -= MAX_TILE_COPIES
        return result, meld_count + 1

    if kind == KanKind.KAKAN:
        if result[tile_id] == 0:
            raise ValueError("Kakan tile is absent from the hand")
        has_pon = any(
            int(meld.tile) == tile_id and meld.type == MeldType.PON
            for meld in own_melds
        )
        if not has_pon:
            raise ValueError("Kakan requires an existing matching Pon")
        result[tile_id] -= 1
        return result, meld_count

    if result[tile_id] < 3:
        raise ValueError("Daiminkan requires three matching concealed tiles")
    if meld_count >= MAX_HAND_MELDS:
        raise ValueError("Daiminkan cannot exceed four melds")
    result[tile_id] -= 3
    return result, meld_count + 1


def _best_discard_shanten(
    hand: list[int],
    meld_count: int,
) -> Tuple[int, Tuple[int, ...]]:
    scored = []
    for tile_id, count in enumerate(hand):
        if count == 0:
            continue
        after_discard = hand.copy()
        after_discard[tile_id] -= 1
        scored.append((
            tile_id,
            int(calculate_shanten(after_discard, meld_count).best),
        ))
    if not scored:
        raise ValueError("kan replacement draw leaves no discardable tile")
    best = min(shanten for _, shanten in scored)
    return best, tuple(
        tile_id for tile_id, shanten in scored if shanten == best
    )


def _is_kan_action(action_id: int) -> bool:
    return (
        ANKAN_ACTION_START <= action_id < PON_ACTION_ID
        or action_id == DAIMINKAN_ACTION_ID
    )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in range 0..1")
