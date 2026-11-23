import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    NO_FLUSH_SUIT,
    NO_TILE,
    TILE_TYPE_COUNT,
    CompactObservation,
    calculate_all_triplets_shanten,
    calculate_seven_pairs_shanten,
    calculate_shanten,
    select_flush_target,
)
from .demonstration import TeacherDecision, pack_action_ids
from .env import EnvState


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
INITIAL_LIVE_WALL_COUNT = 56


class BigHandRoute(str, Enum):
    PAIR_FLEXIBLE = "pair_flexible"
    SEVEN_PAIRS = "seven_pairs"
    ALL_TRIPLETS = "all_triplets"
    FLUSH = "flush"


@dataclass(frozen=True)
class BigHandPlan:
    route: BigHandRoute
    flush_suit: Optional[int]
    initial_shanten: int


@dataclass(frozen=True)
class BigHandTransition:
    player_slot: int
    from_route: BigHandRoute
    to_route: BigHandRoute
    reason: str
    wall_remaining: int


class BigHandAgent:
    """Stateful rule agent that commits each player to one big-hand route."""

    def __init__(
        self,
        seed: Optional[int] = None,
        kan_probability: float = 0.5,
        enabled_routes: Optional[Iterable[BigHandRoute]] = None,
        seven_pairs_max_time_penalty: float = 1.0,
        dadui_switch_margin: float = 0.0,
    ):
        if ACTION_ENCODING_VERSION != SUPPORTED_ACTION_ENCODING_VERSION:
            raise RuntimeError("unsupported action encoding version")
        if not 0.0 <= kan_probability <= 1.0:
            raise ValueError("kan probability must be in range 0..1")
        if seven_pairs_max_time_penalty < 0.0:
            raise ValueError("seven-pairs time penalty cannot be negative")
        if dadui_switch_margin < 0.0:
            raise ValueError("dadui switch margin cannot be negative")

        routes = (
            tuple(BigHandRoute(route) for route in enabled_routes)
            if enabled_routes is not None
            else (
                BigHandRoute.SEVEN_PAIRS,
                BigHandRoute.ALL_TRIPLETS,
                BigHandRoute.FLUSH,
            )
        )
        if not routes:
            raise ValueError("at least one big-hand route must be enabled")

        self._base_seed = seed
        self._random = random.Random(seed)
        self._kan_probability = kan_probability
        self._seven_pairs_max_time_penalty = float(
            seven_pairs_max_time_penalty
        )
        self._dadui_switch_margin = float(dadui_switch_margin)
        self._enabled_routes = frozenset(routes)
        self._plans: Dict[int, BigHandPlan] = {}
        self._transitions: list[BigHandTransition] = []

    @property
    def plans(self) -> Dict[int, BigHandPlan]:
        return dict(self._plans)

    @property
    def transitions(self) -> Tuple[BigHandTransition, ...]:
        return tuple(self._transitions)

    def plan_for_observation(
        self,
        observation: CompactObservation,
    ) -> Optional[BigHandPlan]:
        """Return the current plan for the observation's player, if chosen."""
        return self._plans.get(_player_slot(observation))

    def reset_episode(self, seed: int) -> None:
        episode_seed = int(seed) & ((1 << 64) - 1)
        if self._base_seed is not None:
            episode_seed ^= int(self._base_seed) & ((1 << 64) - 1)
        self._random.seed(episode_seed)
        self._plans.clear()
        self._transitions.clear()

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

        plan = self._plan_for(observation, legal_action_ids)

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
            return _skipped_decision(PASS_ACTION_ID)

        kan_action_ids = self._compatible_kan_actions(
            observation,
            legal_action_ids,
            plan,
        )
        discard_action_ids = tuple(
            action_id
            for action_id in legal_action_ids
            if 0 <= action_id < TILE_TYPE_COUNT
        )
        if discard_action_ids:
            target_action_ids = self._best_discards(
                observation,
                discard_action_ids,
                plan,
            )
            selected_action_id = self._random.choice(target_action_ids)
            selected_action_id = self._maybe_select_kan(
                selected_action_id,
                kan_action_ids,
            )
            self._update_flexible_plan_after_turn_action(
                observation,
                plan,
                selected_action_id,
            )
            return _trainable_decision(
                selected_action_id=selected_action_id,
                supervision_action_ids=discard_action_ids,
                target_action_ids=target_action_ids,
            )

        if PON_ACTION_ID in legal_action_ids and PASS_ACTION_ID in legal_action_ids:
            target_action_id = self._pon_or_pass(observation, plan)
            selected_action_id = self._maybe_select_kan(
                target_action_id,
                kan_action_ids,
            )
            if _is_kan_action(selected_action_id):
                self._commit_route(
                    observation,
                    BigHandRoute.ALL_TRIPLETS,
                    "kan",
                )
            return _trainable_decision(
                selected_action_id=selected_action_id,
                supervision_action_ids=(PON_ACTION_ID, PASS_ACTION_ID),
                target_action_ids=(target_action_id,),
            )

        if PASS_ACTION_ID in legal_action_ids and all(
            action_id == PASS_ACTION_ID or _is_kan_action(action_id)
            for action_id in legal_action_ids
        ):
            selected_action_id = self._maybe_select_kan(
                PASS_ACTION_ID,
                kan_action_ids,
            )
            if _is_kan_action(selected_action_id):
                self._commit_route(
                    observation,
                    BigHandRoute.ALL_TRIPLETS,
                    "kan",
                )
            return _skipped_decision(selected_action_id)

        raise RuntimeError(
            "big-hand agent cannot evaluate legal actions: "
            f"{legal_action_ids}"
        )

    def _plan_for(
        self,
        observation: CompactObservation,
        legal_action_ids: Tuple[int, ...],
    ) -> BigHandPlan:
        player_slot = _player_slot(observation)
        existing = self._plans.get(player_slot)
        if existing is not None:
            return existing

        hand = _hand_counts(observation)
        visible = _visible_counts(observation)
        meld_count, meld_suit_mask = _own_meld_summary(observation)
        discard_ids = tuple(
            action_id
            for action_id in legal_action_ids
            if 0 <= action_id < TILE_TYPE_COUNT
        )
        candidate_hands = (
            tuple(_discard(hand, tile_id) for tile_id in discard_ids)
            if discard_ids
            else (hand,)
        )

        route_scores: Dict[Tuple[BigHandRoute, Optional[int]], int] = {}
        pair_flexible = (
            BigHandRoute.PAIR_FLEXIBLE in self._enabled_routes
            or (
                BigHandRoute.SEVEN_PAIRS in self._enabled_routes
                and BigHandRoute.ALL_TRIPLETS in self._enabled_routes
            )
        )
        for candidate in candidate_hands:
            if pair_flexible and meld_count == 0:
                _keep_minimum(
                    route_scores,
                    (BigHandRoute.PAIR_FLEXIBLE, None),
                    min(
                        int(calculate_seven_pairs_shanten(candidate, 0)),
                        int(calculate_all_triplets_shanten(candidate, 0)),
                    ),
                )
            elif (
                BigHandRoute.SEVEN_PAIRS in self._enabled_routes
                and meld_count == 0
            ):
                _keep_minimum(
                    route_scores,
                    (BigHandRoute.SEVEN_PAIRS, None),
                    int(calculate_seven_pairs_shanten(candidate, meld_count)),
                )
            if (
                not pair_flexible
                and BigHandRoute.ALL_TRIPLETS in self._enabled_routes
            ):
                _keep_minimum(
                    route_scores,
                    (BigHandRoute.ALL_TRIPLETS, None),
                    int(calculate_all_triplets_shanten(candidate, meld_count)),
                )
            if BigHandRoute.FLUSH in self._enabled_routes:
                selection = select_flush_target(
                    candidate,
                    visible,
                    meld_count=meld_count,
                    fixed_meld_suit_mask=meld_suit_mask,
                    seed=self._random.getrandbits(64),
                )
                suit = int(selection["best_suit"])
                if suit != NO_FLUSH_SUIT:
                    _keep_minimum(
                        route_scores,
                        (BigHandRoute.FLUSH, suit),
                        _flush_shanten(candidate, meld_count, suit),
                    )

        if not route_scores:
            raise RuntimeError("no enabled big-hand route is compatible")

        best_shanten = min(route_scores.values())
        best_routes = sorted(
            (
                key
                for key, shanten in route_scores.items()
                if shanten == best_shanten
            ),
            key=lambda key: (key[0].value, -1 if key[1] is None else key[1]),
        )
        route, flush_suit = self._random.choice(best_routes)
        plan = BigHandPlan(route, flush_suit, best_shanten)
        self._plans[player_slot] = plan
        return plan

    def _best_discards(
        self,
        observation: CompactObservation,
        discard_action_ids: Tuple[int, ...],
        plan: BigHandPlan,
    ) -> Tuple[int, ...]:
        hand = _hand_counts(observation)
        visible = _visible_counts(observation)
        meld_count, _ = _own_meld_summary(observation)
        scored = []
        for tile_id in discard_action_ids:
            after_discard = _discard(hand, tile_id)
            if plan.route == BigHandRoute.PAIR_FLEXIBLE:
                seven_pairs_shanten = int(
                    calculate_seven_pairs_shanten(after_discard, 0)
                )
                all_triplets_shanten = int(
                    calculate_all_triplets_shanten(after_discard, 0)
                )
                seven_pairs_cost = self._seven_pairs_cost(
                    seven_pairs_shanten,
                    observation,
                )
                seven_pairs_copies = _effective_draw_copies(
                    after_discard,
                    visible,
                    0,
                    BigHandPlan(BigHandRoute.SEVEN_PAIRS, None, 0),
                )
                all_triplets_copies = _effective_draw_copies(
                    after_discard,
                    visible,
                    0,
                    BigHandPlan(BigHandRoute.ALL_TRIPLETS, None, 0),
                )
                scored.append((
                    tile_id,
                    (
                        min(seven_pairs_cost, all_triplets_shanten),
                        seven_pairs_cost + all_triplets_shanten,
                        -(seven_pairs_copies + all_triplets_copies),
                    ),
                ))
                continue

            shanten = _route_shanten(after_discard, meld_count, plan)
            effective_copies = _effective_draw_copies(
                after_discard,
                visible,
                meld_count,
                plan,
            )
            target_tiles = (
                _suit_tile_count(after_discard, plan.flush_suit)
                if plan.route == BigHandRoute.FLUSH
                else 0
            )
            scored.append(
                (tile_id, (shanten, -target_tiles, -effective_copies))
            )

        best_score = min(score for _, score in scored)
        return tuple(
            tile_id for tile_id, score in scored if score == best_score
        )

    def _pon_or_pass(
        self,
        observation: CompactObservation,
        plan: BigHandPlan,
    ) -> int:
        if plan.route == BigHandRoute.SEVEN_PAIRS:
            return PASS_ACTION_ID

        called_tile = int(observation.current_discard)
        if not 0 <= called_tile < TILE_TYPE_COUNT:
            raise ValueError("Pon decision must have a current discard")
        if (
            plan.route == BigHandRoute.FLUSH
            and called_tile // 9 != plan.flush_suit
        ):
            return PASS_ACTION_ID

        hand = _hand_counts(observation)
        meld_count, _ = _own_meld_summary(observation)
        if hand[called_tile] < 2:
            raise ValueError("Pon decision requires two matching hand tiles")
        if meld_count >= MAX_HAND_MELDS:
            raise ValueError("Pon decision cannot exceed four melds")

        if plan.route == BigHandRoute.PAIR_FLEXIBLE:
            seven_pairs_shanten = int(
                calculate_seven_pairs_shanten(hand, 0)
            )
            seven_pairs_cost = self._seven_pairs_cost(
                seven_pairs_shanten,
                observation,
            )
            after_pon = hand.copy()
            after_pon[called_tile] -= 2
            post_pon_shanten = min(
                int(calculate_all_triplets_shanten(after_discard, 1))
                for after_discard in _hands_after_each_discard(after_pon)
            )
            if (
                post_pon_shanten
                <= seven_pairs_cost + self._dadui_switch_margin
            ):
                self._commit_route(
                    observation,
                    BigHandRoute.ALL_TRIPLETS,
                    "pon",
                )
                return PON_ACTION_ID
            return PASS_ACTION_ID

        pass_shanten = _route_shanten(hand, meld_count, plan)
        after_pon = hand.copy()
        after_pon[called_tile] -= 2
        post_pon_shanten = min(
            _route_shanten(after_discard, meld_count + 1, plan)
            for after_discard in _hands_after_each_discard(after_pon)
        )
        return (
            PON_ACTION_ID
            if post_pon_shanten < pass_shanten
            else PASS_ACTION_ID
        )

    def _compatible_kan_actions(
        self,
        observation: CompactObservation,
        legal_action_ids: Tuple[int, ...],
        plan: BigHandPlan,
    ) -> Tuple[int, ...]:
        if plan.route == BigHandRoute.SEVEN_PAIRS:
            return ()

        if plan.route == BigHandRoute.PAIR_FLEXIBLE:
            hand = _hand_counts(observation)
            seven_pairs_shanten = int(
                calculate_seven_pairs_shanten(hand, 0)
            )
            seven_pairs_cost = self._seven_pairs_cost(
                seven_pairs_shanten,
                observation,
            )
            all_triplets_shanten = int(
                calculate_all_triplets_shanten(hand, 0)
            )
            if (
                all_triplets_shanten
                > seven_pairs_cost + self._dadui_switch_margin
            ):
                return ()

        compatible = []
        for action_id in legal_action_ids:
            tile_id = _kan_tile(action_id, observation)
            if tile_id is None:
                continue
            if (
                plan.route == BigHandRoute.FLUSH
                and tile_id // 9 != plan.flush_suit
            ):
                continue
            compatible.append(action_id)
        return tuple(compatible)

    def _seven_pairs_cost(
        self,
        raw_shanten: int,
        observation: CompactObservation,
    ) -> float:
        return _seven_pairs_cost(
            raw_shanten,
            int(observation.wall_remaining),
            self._seven_pairs_max_time_penalty,
        )

    def _update_flexible_plan_after_turn_action(
        self,
        observation: CompactObservation,
        plan: BigHandPlan,
        selected_action_id: int,
    ) -> None:
        if plan.route != BigHandRoute.PAIR_FLEXIBLE:
            return
        if _is_kan_action(selected_action_id):
            self._commit_route(
                observation,
                BigHandRoute.ALL_TRIPLETS,
                "kan",
            )
            return
        if not 0 <= selected_action_id < TILE_TYPE_COUNT:
            return

        after_discard = _discard(
            _hand_counts(observation),
            selected_action_id,
        )
        seven_pairs_shanten = int(
            calculate_seven_pairs_shanten(after_discard, 0)
        )
        if seven_pairs_shanten <= 0:
            self._commit_route(
                observation,
                BigHandRoute.SEVEN_PAIRS,
                "seven_pairs_tenpai",
            )
            return

        seven_pairs_cost = self._seven_pairs_cost(
            seven_pairs_shanten,
            observation,
        )
        all_triplets_shanten = int(
            calculate_all_triplets_shanten(after_discard, 0)
        )
        if all_triplets_shanten < seven_pairs_cost:
            self._commit_route(
                observation,
                BigHandRoute.ALL_TRIPLETS,
                "time_or_shape",
            )

    def _commit_route(
        self,
        observation: CompactObservation,
        route: BigHandRoute,
        reason: str,
    ) -> None:
        player_slot = _player_slot(observation)
        plan = self._plans.get(player_slot)
        if plan is None or plan.route != BigHandRoute.PAIR_FLEXIBLE:
            return
        self._plans[player_slot] = BigHandPlan(
            route=route,
            flush_suit=plan.flush_suit,
            initial_shanten=plan.initial_shanten,
        )
        self._transitions.append(BigHandTransition(
            player_slot=player_slot,
            from_route=plan.route,
            to_route=route,
            reason=reason,
            wall_remaining=int(observation.wall_remaining),
        ))

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
    actions = tuple(
        action_id
        for action_id, is_legal in enumerate(legal_mask)
        if is_legal
    )
    if not actions:
        raise ValueError("legal mask contains no legal actions")
    return actions


def _player_slot(observation: CompactObservation) -> int:
    slot = int(observation.dealer)
    if not 0 <= slot < 4:
        raise ValueError("relative dealer must identify one of four players")
    return slot


def _hand_counts(observation: CompactObservation) -> list[int]:
    counts = [int(count) for count in observation.hand_counts]
    if len(counts) != TILE_TYPE_COUNT:
        raise ValueError("hand counts must contain 27 entries")
    return counts


def _visible_counts(observation: CompactObservation) -> list[int]:
    counts = [int(count) for count in observation.visible_counts]
    if len(counts) != TILE_TYPE_COUNT:
        raise ValueError("visible counts must contain 27 entries")
    return counts


def _own_meld_summary(observation: CompactObservation) -> Tuple[int, int]:
    public_meld_count = int(observation.meld_count)
    melds = observation.melds
    if not 0 <= public_meld_count <= len(melds):
        raise ValueError("observation meld count is out of range")

    count = 0
    suit_mask = 0
    for meld in melds[:public_meld_count]:
        if not int(meld.valid) or int(meld.relative_owner) != 0:
            continue
        tile_id = int(meld.tile)
        if not 0 <= tile_id < TILE_TYPE_COUNT:
            raise ValueError("own meld tile is outside its valid range")
        count += 1
        suit_mask |= 1 << (tile_id // 9)
    if count > MAX_HAND_MELDS:
        raise ValueError("player cannot have more than four melds")
    return count, suit_mask


def _discard(hand: list[int], tile_id: int) -> list[int]:
    if not 0 <= tile_id < TILE_TYPE_COUNT or hand[tile_id] <= 0:
        raise ValueError("legal discard is absent from the hand")
    result = hand.copy()
    result[tile_id] -= 1
    return result


def _keep_minimum(mapping, key, value: int) -> None:
    previous = mapping.get(key)
    if previous is None or value < previous:
        mapping[key] = value


def _route_shanten(
    hand: list[int],
    meld_count: int,
    plan: BigHandPlan,
) -> int:
    if plan.route == BigHandRoute.SEVEN_PAIRS:
        return int(calculate_seven_pairs_shanten(hand, meld_count))
    if plan.route == BigHandRoute.ALL_TRIPLETS:
        return int(calculate_all_triplets_shanten(hand, meld_count))
    if plan.flush_suit is None:
        raise ValueError("flush plan requires a target suit")
    return _flush_shanten(hand, meld_count, plan.flush_suit)


def _seven_pairs_cost(
    raw_shanten: int,
    wall_remaining: int,
    max_penalty: float,
) -> float:
    if raw_shanten <= 0:
        return float(raw_shanten)
    progress = 1.0 - wall_remaining / INITIAL_LIVE_WALL_COUNT
    progress = min(1.0, max(0.0, progress))
    return float(raw_shanten) + max_penalty * progress


def _flush_shanten(hand: list[int], meld_count: int, suit: int) -> int:
    projected = [0] * TILE_TYPE_COUNT
    first_tile = suit * 9
    projected[first_tile:first_tile + 9] = hand[first_tile:first_tile + 9]
    return int(calculate_shanten(projected, meld_count).best)


def _suit_tile_count(hand: list[int], suit: Optional[int]) -> int:
    if suit is None:
        return 0
    first_tile = suit * 9
    return sum(hand[first_tile:first_tile + 9])


def _effective_draw_copies(
    hand: list[int],
    visible: list[int],
    meld_count: int,
    plan: BigHandPlan,
) -> int:
    current = _route_shanten(hand, meld_count, plan)
    copies = 0
    for tile_id in range(TILE_TYPE_COUNT):
        if (
            plan.route == BigHandRoute.FLUSH
            and tile_id // 9 != plan.flush_suit
        ):
            continue
        remaining = 4 - visible[tile_id]
        if remaining <= 0:
            continue
        after_draw = hand.copy()
        after_draw[tile_id] += 1
        if _route_shanten(after_draw, meld_count, plan) < current:
            copies += remaining
    return copies


def _hands_after_each_discard(hand: list[int]) -> Tuple[list[int], ...]:
    hands = tuple(
        _discard(hand, tile_id)
        for tile_id, count in enumerate(hand)
        if count > 0
    )
    if not hands:
        raise ValueError("Pon must leave at least one discardable tile")
    return hands


def _kan_tile(
    action_id: int,
    observation: CompactObservation,
) -> Optional[int]:
    if ANKAN_ACTION_START <= action_id < KAKAN_ACTION_START:
        return action_id - ANKAN_ACTION_START
    if KAKAN_ACTION_START <= action_id < PON_ACTION_ID:
        return action_id - KAKAN_ACTION_START
    if action_id == DAIMINKAN_ACTION_ID:
        tile_id = int(observation.current_discard)
        if not 0 <= tile_id < TILE_TYPE_COUNT:
            raise ValueError("Daiminkan requires a current discard")
        return tile_id
    return None


def _is_kan_action(action_id: int) -> bool:
    return (
        ANKAN_ACTION_START <= action_id < PON_ACTION_ID
        or action_id == DAIMINKAN_ACTION_ID
    )


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
