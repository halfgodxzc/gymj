from dataclasses import dataclass, fields, replace
from typing import Sequence

import numpy as np
import torch

from ._gymj import (
    ACTION_SPACE_SIZE,
    DECISION_FEATURE_VERSION,
    MAX_DISCARD_EVENTS,
    MAX_MELDS,
    PATTERN_ALL_TRIPLETS,
    PATTERN_DI_LONG,
    PATTERN_DRAGON_SEVEN_PAIRS,
    PATTERN_FEATURE_VERSION,
    PATTERN_FLUSH,
    PATTERN_SEVEN_PAIRS,
    PATTERN_SINGLE_WAIT,
    PATTERN_STANDARD,
    TILE_TYPE_COUNT,
    ActionFamily,
    ActionType,
    DecisionDrawKind,
    MeldType,
    ReadyKind,
    RoundPhase,
)
from .kan_adapter import (
    KAN_ADAPTER_FEATURE_DIM,
    tensorize_kan_adapter_features,
)


V4_TENSOR_VERSION = 1
V4_REQUIRED_PATTERN_FEATURE_VERSION = 2
V4_REQUIRED_DECISION_FEATURE_VERSION = 3

V4_PLAYER_COUNT = 4
V4_TILE_PAD = TILE_TYPE_COUNT
V4_PLAYER_PAD = V4_PLAYER_COUNT
V4_ACTION_TYPE_PAD = 10
V4_ACTION_TYPE_COUNT = V4_ACTION_TYPE_PAD + 1
V4_ACTION_FAMILY_COUNT = 4
V4_DRAW_KIND_COUNT = 4
V4_MELD_TYPE_COUNT = 4
V4_PHASE_COUNT = 7

_MAX_TILE_COPIES = 4.0
_MAX_HAND_TILES = 14.0
_MAX_ROUTE_DRAW_KINDS = 27.0
_MAX_ROUTE_DRAW_COPIES = 108.0
_MAX_FLUSH_DRAW_KINDS = 9.0
_MAX_FLUSH_DRAW_COPIES = 36.0
_MAX_DISCARDS = 72.0
_MAX_MELDS = 4.0
_MAX_FIXED_CHICKENS = 8.0
_MAX_REFERENCE_FAN = 9.0
_MAX_REFERENCE_HU_POINTS = 45.0
_MAX_DECISION_SEQUENCE = 256.0

_PATTERN_BITS = (
    PATTERN_STANDARD,
    PATTERN_ALL_TRIPLETS,
    PATTERN_SEVEN_PAIRS,
    PATTERN_DRAGON_SEVEN_PAIRS,
    PATTERN_DI_LONG,
    PATTERN_FLUSH,
    PATTERN_SINGLE_WAIT,
)

V4_HAND_SUMMARY_NAMES = (
    "post_discard_state",
    "normal_progress",
    "seven_pairs_progress",
    "all_triplets_progress",
    "normal_effective_draw_kinds",
    "seven_pairs_effective_draw_kinds",
    "all_triplets_effective_draw_kinds",
    "normal_effective_draw_copies",
    "seven_pairs_effective_draw_copies",
    "all_triplets_effective_draw_copies",
    "wan_flush_progress",
    "wan_concealed_tiles",
    "wan_off_suit_tiles",
    "wan_meld_compatible",
    "wan_effective_draw_kinds",
    "wan_effective_draw_copies",
    "tong_flush_progress",
    "tong_concealed_tiles",
    "tong_off_suit_tiles",
    "tong_meld_compatible",
    "tong_effective_draw_kinds",
    "tong_effective_draw_copies",
    "tiao_flush_progress",
    "tiao_concealed_tiles",
    "tiao_off_suit_tiles",
    "tiao_meld_compatible",
    "tiao_effective_draw_kinds",
    "tiao_effective_draw_copies",
    "best_flush_wan",
    "best_flush_tong",
    "best_flush_tiao",
    "live_wait_kinds",
    "live_wait_copies",
)
V4_HAND_SUMMARY_DIM = len(V4_HAND_SUMMARY_NAMES)

V4_HAND_TILE_NAMES = (
    "concealed_count",
    "visible_count",
    "unseen_count",
    "normal_effective_copies",
    "seven_pairs_effective_copies",
    "all_triplets_effective_copies",
    "wan_flush_effective_copies",
    "tong_flush_effective_copies",
    "tiao_flush_effective_copies",
    "shape_wait",
    "live_wait",
    "wait_standard",
    "wait_all_triplets",
    "wait_seven_pairs",
    "wait_dragon_seven_pairs",
    "wait_di_long",
    "wait_flush",
    "wait_single",
    "wait_base_fan",
    "wait_base_hu_points",
    "wait_remaining_copies",
)
V4_HAND_TILE_DIM = len(V4_HAND_TILE_NAMES)

V4_PLAYER_TILE_NAMES = (
    "river_count",
    "called_discard_count",
    "meld_tile_count",
    "public_owned_tile_count",
)
V4_PLAYER_TILE_DIM = len(V4_PLAYER_TILE_NAMES)

V4_PLAYER_SUMMARY_NAMES = (
    "pon_count",
    "ankan_count",
    "kakan_count",
    "daiminkan_count",
    "ankan_bean_count",
    "kakan_bean_count",
    "daiminkan_bean_count",
    "discard_count",
    "meld_count",
    "fixed_chicken_count",
    "known_tenpai_status",
    "signed_fixed_chicken_count",
    "chongfeng_chicken_count",
    "responsibility_chicken_given_count",
    "responsibility_chicken_received_count",
)
V4_PLAYER_SUMMARY_DIM = len(V4_PLAYER_SUMMARY_NAMES)

V4_PROGRESS_NAMES = (
    "decision_sequence",
    "self_has_normal_drawn",
    "next_has_normal_drawn",
    "opposite_has_normal_drawn",
    "previous_has_normal_drawn",
    "self_has_discarded",
    "next_has_discarded",
    "opposite_has_discarded",
    "previous_has_discarded",
    "self_opening_hand_changed",
    "next_opening_hand_changed",
    "opposite_opening_hand_changed",
    "previous_opening_hand_changed",
    "self_ready_window_closed",
    "next_ready_window_closed",
    "opposite_ready_window_closed",
    "previous_ready_window_closed",
    "must_discard_after_call",
    "pending_discard_is_hot",
    "pending_kakan",
)
V4_PROGRESS_DIM = len(V4_PROGRESS_NAMES)

V4_CONTEXT_NAMES = (
    "wall_remaining",
    "current_shanten_progress",
    "discard_count",
    "meld_count",
)
V4_CONTEXT_DIM = len(V4_CONTEXT_NAMES)

V4_ACTION_SCALAR_NAMES = (
    "post_hand_known",
    "hu_available",
    "hu_is_tsumo",
    "hu_is_qianggang",
    "hu_standard",
    "hu_all_triplets",
    "hu_seven_pairs",
    "hu_dragon_seven_pairs",
    "hu_di_long",
    "hu_flush",
    "hu_single_wait",
    "hu_shape_fan",
    "hu_ready_fan",
    "hu_timing_fan",
    "hu_special_fan",
    "hu_total_fan",
    "hu_points",
    "ready_available",
    "ready_none",
    "ready_tian_ting",
    "ready_bao_ting",
    "ready_wait_kinds",
    "ready_wait_copies",
    "ready_max_base_hu_points",
    "pass_available",
    "pass_declines_pon",
    "pass_declines_daiminkan",
    "pass_declines_ron",
    "pass_declines_ready",
)
V4_ACTION_SCALAR_DIM = len(V4_ACTION_SCALAR_NAMES)


_ACTION_TYPE_INDICES = {
    ActionType.PASS: 0,
    ActionType.DISCARD: 1,
    ActionType.PON: 2,
    ActionType.ANKAN: 3,
    ActionType.KAKAN: 4,
    ActionType.DAIMINKAN: 5,
    ActionType.TSUMO: 6,
    ActionType.RON: 7,
    ActionType.DECLARE_READY: 8,
    ActionType.DRAW: 9,
}
V4_PASS_TYPE = _ACTION_TYPE_INDICES[ActionType.PASS]
V4_DISCARD_TYPE = _ACTION_TYPE_INDICES[ActionType.DISCARD]
V4_PON_TYPE = _ACTION_TYPE_INDICES[ActionType.PON]
V4_ANKAN_TYPE = _ACTION_TYPE_INDICES[ActionType.ANKAN]
V4_KAKAN_TYPE = _ACTION_TYPE_INDICES[ActionType.KAKAN]
V4_DAIMINKAN_TYPE = _ACTION_TYPE_INDICES[ActionType.DAIMINKAN]
V4_TSUMO_TYPE = _ACTION_TYPE_INDICES[ActionType.TSUMO]
V4_RON_TYPE = _ACTION_TYPE_INDICES[ActionType.RON]

_ACTION_FAMILY_INDICES = {
    ActionFamily.TURN: 0,
    ActionFamily.RESPONSE: 1,
    ActionFamily.READY: 2,
    ActionFamily.UNSUPPORTED: 3,
}

_DRAW_KIND_INDICES = {
    DecisionDrawKind.NONE: 0,
    DecisionDrawKind.OPENING: 1,
    DecisionDrawKind.NORMAL: 2,
    DecisionDrawKind.REPLACEMENT: 3,
}

_MELD_TYPE_INDICES = {
    MeldType.PON: 0,
    MeldType.ANKAN: 1,
    MeldType.KAKAN: 2,
    MeldType.DAIMINKAN: 3,
}

_PHASE_INDICES = {
    RoundPhase.NOT_STARTED: 0,
    RoundPhase.AWAIT_TURN_ACTION: 1,
    RoundPhase.AWAIT_READY_DECLARATION: 2,
    RoundPhase.AWAIT_DISCARD_RESPONSES: 3,
    RoundPhase.AWAIT_KAKAN_RESPONSES: 4,
    RoundPhase.REPLACEMENT_DRAW: 5,
    RoundPhase.FINISHED: 6,
}

@dataclass(frozen=True)
class V4TensorBatch:
    hand_summary: torch.Tensor
    hand_tiles: torch.Tensor
    discard_tiles: torch.Tensor
    discard_players: torch.Tensor
    discard_called: torch.Tensor
    discard_valid: torch.Tensor
    meld_tiles: torch.Tensor
    meld_owners: torch.Tensor
    meld_sources: torch.Tensor
    meld_types: torch.Tensor
    meld_valid: torch.Tensor
    player_tiles: torch.Tensor
    player_summary: torch.Tensor
    context: torch.Tensor
    phase: torch.Tensor
    current_actor: torch.Tensor
    current_discard: torch.Tensor
    current_discard_actor: torch.Tensor
    dealer: torch.Tensor
    ready_kinds: torch.Tensor
    current_waits: torch.Tensor
    locked_ready_waits: torch.Tensor
    progress: torch.Tensor
    last_draw_kind: torch.Tensor
    hot_discard_player: torch.Tensor
    action_legal: torch.Tensor
    action_supported: torch.Tensor
    action_type: torch.Tensor
    action_family: torch.Tensor
    action_tile: torch.Tensor
    action_source: torch.Tensor
    action_target: torch.Tensor
    action_scalar: torch.Tensor
    action_ready_waits: torch.Tensor
    discard_post_hand_summary: torch.Tensor
    discard_post_hand_tiles: torch.Tensor
    pon_follow_up_mask: torch.Tensor
    pon_follow_up_summary: torch.Tensor
    pon_follow_up_tiles: torch.Tensor
    kan_adapter_features: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.hand_summary.shape[0])

    @property
    def selectable_actions(self) -> torch.Tensor:
        return self.action_legal & self.action_supported

    def to(self, device: torch.device | str) -> "V4TensorBatch":
        return replace(
            self,
            **{
                field.name: (
                    None
                    if getattr(self, field.name) is None
                    else getattr(self, field.name).to(device)
                )
                for field in fields(self)
            },
        )

    def index_select(self, indices: torch.Tensor) -> "V4TensorBatch":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("V4 batch indices must be a one-dimensional long tensor")
        return replace(
            self,
            **{
                field.name: (
                    None
                    if getattr(self, field.name) is None
                    else getattr(self, field.name).index_select(
                        0,
                        indices.to(getattr(self, field.name).device),
                    )
                )
                for field in fields(self)
            },
        )

    def validate(self) -> None:
        batch = self.batch_size
        expected = {
            "hand_summary": (batch, V4_HAND_SUMMARY_DIM),
            "hand_tiles": (batch, TILE_TYPE_COUNT, V4_HAND_TILE_DIM),
            "discard_tiles": (batch, MAX_DISCARD_EVENTS),
            "discard_players": (batch, MAX_DISCARD_EVENTS),
            "discard_called": (batch, MAX_DISCARD_EVENTS),
            "discard_valid": (batch, MAX_DISCARD_EVENTS),
            "meld_tiles": (batch, MAX_MELDS),
            "meld_owners": (batch, MAX_MELDS),
            "meld_sources": (batch, MAX_MELDS),
            "meld_types": (batch, MAX_MELDS),
            "meld_valid": (batch, MAX_MELDS),
            "player_tiles": (
                batch,
                V4_PLAYER_COUNT,
                TILE_TYPE_COUNT,
                V4_PLAYER_TILE_DIM,
            ),
            "player_summary": (
                batch,
                V4_PLAYER_COUNT,
                V4_PLAYER_SUMMARY_DIM,
            ),
            "context": (batch, V4_CONTEXT_DIM),
            "phase": (batch,),
            "current_actor": (batch,),
            "current_discard": (batch,),
            "current_discard_actor": (batch,),
            "dealer": (batch,),
            "ready_kinds": (batch, V4_PLAYER_COUNT),
            "current_waits": (batch, TILE_TYPE_COUNT),
            "locked_ready_waits": (batch, TILE_TYPE_COUNT),
            "progress": (batch, V4_PROGRESS_DIM),
            "last_draw_kind": (batch,),
            "hot_discard_player": (batch,),
            "action_legal": (batch, ACTION_SPACE_SIZE),
            "action_supported": (batch, ACTION_SPACE_SIZE),
            "action_type": (batch, ACTION_SPACE_SIZE),
            "action_family": (batch, ACTION_SPACE_SIZE),
            "action_tile": (batch, ACTION_SPACE_SIZE),
            "action_source": (batch, ACTION_SPACE_SIZE),
            "action_target": (batch, ACTION_SPACE_SIZE),
            "action_scalar": (
                batch,
                ACTION_SPACE_SIZE,
                V4_ACTION_SCALAR_DIM,
            ),
            "action_ready_waits": (
                batch,
                ACTION_SPACE_SIZE,
                TILE_TYPE_COUNT,
            ),
            "discard_post_hand_summary": (
                batch,
                TILE_TYPE_COUNT,
                V4_HAND_SUMMARY_DIM,
            ),
            "discard_post_hand_tiles": (
                batch,
                TILE_TYPE_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
            "pon_follow_up_mask": (batch, TILE_TYPE_COUNT),
            "pon_follow_up_summary": (
                batch,
                TILE_TYPE_COUNT,
                V4_HAND_SUMMARY_DIM,
            ),
            "pon_follow_up_tiles": (
                batch,
                TILE_TYPE_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
        }
        bool_fields = {
            "action_legal",
            "action_supported",
            "discard_valid",
            "meld_valid",
            "pon_follow_up_mask",
        }
        integer_fields = {
            "discard_tiles",
            "discard_players",
            "meld_tiles",
            "meld_owners",
            "meld_sources",
            "meld_types",
            "phase",
            "current_actor",
            "current_discard",
            "current_discard_actor",
            "dealer",
            "ready_kinds",
            "last_draw_kind",
            "hot_discard_player",
            "action_type",
            "action_family",
            "action_tile",
            "action_source",
            "action_target",
        }
        devices = set()
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name in bool_fields and tensor.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool")
            if name in integer_fields and tensor.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long")
            if name not in bool_fields | integer_fields:
                if tensor.dtype != torch.float32:
                    raise TypeError(f"{name} must use torch.float32")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{name} contains non-finite values")
            devices.add(tensor.device)
        if self.kan_adapter_features is not None:
            expected_kan = (
                batch,
                ACTION_SPACE_SIZE,
                KAN_ADAPTER_FEATURE_DIM,
            )
            if tuple(self.kan_adapter_features.shape) != expected_kan:
                raise ValueError(
                    "kan_adapter_features must have shape "
                    f"{expected_kan}"
                )
            if self.kan_adapter_features.dtype != torch.float32:
                raise TypeError("kan_adapter_features must use torch.float32")
            if not torch.isfinite(self.kan_adapter_features).all():
                raise ValueError("kan_adapter_features contains non-finite values")
            devices.add(self.kan_adapter_features.device)
        if len(devices) != 1:
            raise ValueError("all V4 tensors must be on the same device")


def _enum_index(value, mapping, name: str) -> int:
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unknown {name}: {value}") from error


def _mask_bits(mask: int) -> np.ndarray:
    return np.asarray(
        [1.0 if mask & (1 << tile_id) else 0.0
         for tile_id in range(TILE_TYPE_COUNT)],
        dtype=np.float32,
    )


def _pattern_bits(mask: int) -> np.ndarray:
    return np.asarray(
        [1.0 if mask & bit else 0.0 for bit in _PATTERN_BITS],
        dtype=np.float32,
    )


def _shanten_progress(shanten: int) -> float:
    if shanten < -1 or shanten > 8:
        raise ValueError(f"shanten must be in range -1..8, got {shanten}")
    return (8.0 - float(shanten)) / 9.0


def _hand_arrays(
    hand,
    concealed_counts: np.ndarray,
    visible_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if concealed_counts.shape != (TILE_TYPE_COUNT,):
        raise ValueError("concealed counts must have 27 entries")
    if visible_counts.shape != (TILE_TYPE_COUNT,):
        raise ValueError("visible counts must have 27 entries")
    if np.any(concealed_counts < 0) or np.any(concealed_counts > 4):
        raise ValueError("concealed tile count is outside 0..4")
    if np.any(visible_counts < concealed_counts) or np.any(visible_counts > 4):
        raise ValueError("visible tile count is inconsistent")

    summary = np.zeros(V4_HAND_SUMMARY_DIM, dtype=np.float32)
    summary[0] = float(hand.post_discard_state)
    cursor = 1
    routes = list(hand.routes)
    for route in routes:
        summary[cursor] = _shanten_progress(int(route.shanten))
        cursor += 1
    for route in routes:
        summary[cursor] = float(route.effective_draws.kinds) / _MAX_ROUTE_DRAW_KINDS
        cursor += 1
    for route in routes:
        summary[cursor] = float(route.effective_draws.copies) / _MAX_ROUTE_DRAW_COPIES
        cursor += 1
    flushes = list(hand.flushes)
    for flush in flushes:
        summary[cursor:cursor + 6] = (
            _shanten_progress(int(flush.shanten)),
            float(flush.concealed_tiles) / _MAX_HAND_TILES,
            float(flush.off_suit_tiles) / _MAX_HAND_TILES,
            float(flush.meld_compatible),
            float(flush.effective_draws.kinds) / _MAX_FLUSH_DRAW_KINDS,
            float(flush.effective_draws.copies) / _MAX_FLUSH_DRAW_COPIES,
        )
        cursor += 6
    for suit in range(3):
        summary[cursor + suit] = float(
            bool(int(hand.best_flush_suit_mask) & (1 << suit))
        )
    cursor += 3
    summary[cursor] = float(hand.live_wait_kinds) / _MAX_ROUTE_DRAW_KINDS
    summary[cursor + 1] = float(hand.live_wait_copies) / _MAX_ROUTE_DRAW_COPIES

    tiles = np.zeros(
        (TILE_TYPE_COUNT, V4_HAND_TILE_DIM),
        dtype=np.float32,
    )
    tiles[:, 0] = concealed_counts / _MAX_TILE_COPIES
    tiles[:, 1] = visible_counts / _MAX_TILE_COPIES
    tiles[:, 2] = (_MAX_TILE_COPIES - visible_counts) / _MAX_TILE_COPIES
    for route_index, route in enumerate(routes):
        tiles[:, 3 + route_index] = np.asarray(
            route.effective_draws.remaining_by_tile,
            dtype=np.float32,
        ) / _MAX_TILE_COPIES
    for suit, flush in enumerate(flushes):
        tiles[:, 6 + suit] = np.asarray(
            flush.effective_draws.remaining_by_tile,
            dtype=np.float32,
        ) / _MAX_TILE_COPIES
    tiles[:, 9] = _mask_bits(int(hand.shape_wait_mask))
    tiles[:, 10] = _mask_bits(int(hand.live_wait_mask))
    for tile_id, wait in enumerate(hand.wait_values):
        tiles[tile_id, 11:18] = _pattern_bits(int(wait.pattern_mask))
        tiles[tile_id, 18] = float(wait.base_fan) / _MAX_REFERENCE_FAN
        tiles[tile_id, 19] = (
            float(wait.base_hu_points) / _MAX_REFERENCE_HU_POINTS
        )
        tiles[tile_id, 20] = (
            float(wait.remaining_copies) / _MAX_TILE_COPIES
        )
    return summary, tiles


def v4_hand_pattern_arrays(
    hand,
    concealed_counts: np.ndarray,
    visible_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one engine hand pattern into the stable V4 tensor layout."""
    return _hand_arrays(hand, concealed_counts, visible_counts)


def _public_arrays(public_state):
    player_tiles = np.zeros(
        (V4_PLAYER_COUNT, TILE_TYPE_COUNT, V4_PLAYER_TILE_DIM),
        dtype=np.float32,
    )
    player_summary = np.zeros(
        (V4_PLAYER_COUNT, V4_PLAYER_SUMMARY_DIM),
        dtype=np.float32,
    )
    public_owned = np.zeros(
        (V4_PLAYER_COUNT, TILE_TYPE_COUNT),
        dtype=np.int16,
    )
    river = np.zeros_like(public_owned)
    meld = np.zeros_like(public_owned)

    for player_id, player in enumerate(public_state.players):
        river[player_id] = np.asarray(player.river_counts, dtype=np.int16)
        called = np.asarray(player.called_discard_counts, dtype=np.int16)
        meld[player_id] = np.asarray(player.meld_tile_counts, dtype=np.int16)
        public_owned[player_id] = np.asarray(
            player.public_owned_tile_counts,
            dtype=np.int16,
        )
        player_tiles[player_id, :, 0] = river[player_id] / _MAX_TILE_COPIES
        player_tiles[player_id, :, 1] = called / _MAX_TILE_COPIES
        player_tiles[player_id, :, 2] = meld[player_id] / _MAX_TILE_COPIES
        player_tiles[player_id, :, 3] = (
            public_owned[player_id] / _MAX_TILE_COPIES
        )
        player_summary[player_id] = np.asarray(
            [
                *(float(value) / _MAX_MELDS for value in player.meld_type_counts),
                *(float(value) / _MAX_MELDS for value in player.bean_type_counts),
                float(player.discard_count) / _MAX_DISCARDS,
                float(player.meld_count) / _MAX_MELDS,
                float(player.fixed_chicken_count) / _MAX_FIXED_CHICKENS,
                float(player.known_tenpai_status),
                float(player.signed_fixed_chicken_count) / _MAX_FIXED_CHICKENS,
                float(player.chongfeng_chicken_count),
                float(player.responsibility_chicken_given_count),
                float(player.responsibility_chicken_received_count) / 3.0,
            ],
            dtype=np.float32,
        )

    concealed = public_owned[0] - river[0] - meld[0]
    visible = public_owned.sum(axis=0)
    if np.any(concealed < 0):
        raise ValueError("public state cannot reconstruct the concealed hand")
    if np.any(visible > 4):
        raise ValueError("public state exposes more than four tile copies")
    return player_tiles, player_summary, concealed, visible


def _progress_arrays(progress):
    numeric = np.zeros(V4_PROGRESS_DIM, dtype=np.float32)
    numeric[0] = float(progress.decision_sequence) / _MAX_DECISION_SEQUENCE
    cursor = 1
    for values in (
        progress.has_normal_drawn,
        progress.has_discarded,
        progress.opening_hand_changed,
        progress.ready_window_closed,
    ):
        numeric[cursor:cursor + V4_PLAYER_COUNT] = np.asarray(
            values,
            dtype=np.float32,
        )
        cursor += V4_PLAYER_COUNT
    numeric[cursor:cursor + 3] = (
        float(progress.must_discard_after_call),
        float(progress.pending_discard_is_hot),
        float(progress.pending_kakan),
    )
    draw_kind = _enum_index(
        progress.last_draw_kind,
        _DRAW_KIND_INDICES,
        "draw kind",
    )
    hot_player = int(progress.hot_discard_player)
    if hot_player < 0 or hot_player > V4_PLAYER_PAD:
        hot_player = V4_PLAYER_PAD
    return numeric, draw_kind, hot_player


def _ready_kind_values(kind) -> tuple[float, float, float]:
    return (
        float(kind == ReadyKind.NONE),
        float(kind == ReadyKind.TIAN_TING),
        float(kind == ReadyKind.BAO_TING),
    )


def _action_scalar(candidate) -> np.ndarray:
    scalar = np.zeros(V4_ACTION_SCALAR_DIM, dtype=np.float32)
    scalar[0] = float(candidate.post_hand_known)

    hu = candidate.hu
    scalar[1:4] = (
        float(hu.available),
        float(hu.is_tsumo),
        float(hu.is_qianggang),
    )
    scalar[4:11] = _pattern_bits(int(hu.pattern_mask))
    scalar[11:16] = np.asarray(
        [
            hu.shape_fan,
            hu.ready_fan,
            hu.timing_fan,
            hu.special_fan,
            hu.total_fan,
        ],
        dtype=np.float32,
    ) / _MAX_REFERENCE_FAN
    scalar[16] = float(hu.hu_points) / _MAX_REFERENCE_HU_POINTS

    ready = candidate.ready
    scalar[17] = float(ready.available)
    if ready.available:
        scalar[18:21] = _ready_kind_values(ready.kind)
    scalar[21:24] = (
        float(ready.wait_kinds) / _MAX_ROUTE_DRAW_KINDS,
        float(ready.wait_copies) / _MAX_ROUTE_DRAW_COPIES,
        float(ready.max_base_hu_points) / _MAX_REFERENCE_HU_POINTS,
    )

    passed = candidate.pass_features
    scalar[24:29] = (
        float(passed.available),
        float(passed.declines_pon),
        float(passed.declines_daiminkan),
        float(passed.declines_ron),
        float(passed.declines_ready),
    )
    return scalar


def tensorize_v4_decisions(
    decisions: Sequence,
    device: torch.device | str | None = None,
) -> V4TensorBatch:
    if PATTERN_FEATURE_VERSION != V4_REQUIRED_PATTERN_FEATURE_VERSION:
        raise RuntimeError("V4 tensorizer requires pattern feature version 2")
    if DECISION_FEATURE_VERSION != V4_REQUIRED_DECISION_FEATURE_VERSION:
        raise RuntimeError("V4 tensorizer requires decision feature version 3")
    if not decisions:
        raise ValueError("V4 decision batch cannot be empty")

    batch = len(decisions)
    arrays = {
        "hand_summary": np.zeros((batch, V4_HAND_SUMMARY_DIM), np.float32),
        "hand_tiles": np.zeros(
            (batch, TILE_TYPE_COUNT, V4_HAND_TILE_DIM), np.float32
        ),
        "discard_tiles": np.full(
            (batch, MAX_DISCARD_EVENTS), V4_TILE_PAD, np.int64
        ),
        "discard_players": np.full(
            (batch, MAX_DISCARD_EVENTS), V4_PLAYER_PAD, np.int64
        ),
        "discard_called": np.zeros(
            (batch, MAX_DISCARD_EVENTS), np.float32
        ),
        "discard_valid": np.zeros(
            (batch, MAX_DISCARD_EVENTS), np.bool_
        ),
        "meld_tiles": np.full(
            (batch, MAX_MELDS), V4_TILE_PAD, np.int64
        ),
        "meld_owners": np.full(
            (batch, MAX_MELDS), V4_PLAYER_PAD, np.int64
        ),
        "meld_sources": np.full(
            (batch, MAX_MELDS), V4_PLAYER_PAD, np.int64
        ),
        "meld_types": np.zeros((batch, MAX_MELDS), np.int64),
        "meld_valid": np.zeros((batch, MAX_MELDS), np.bool_),
        "player_tiles": np.zeros(
            (
                batch,
                V4_PLAYER_COUNT,
                TILE_TYPE_COUNT,
                V4_PLAYER_TILE_DIM,
            ),
            np.float32,
        ),
        "player_summary": np.zeros(
            (batch, V4_PLAYER_COUNT, V4_PLAYER_SUMMARY_DIM), np.float32
        ),
        "context": np.zeros((batch, V4_CONTEXT_DIM), np.float32),
        "phase": np.zeros(batch, np.int64),
        "current_actor": np.full(batch, V4_PLAYER_PAD, np.int64),
        "current_discard": np.full(batch, V4_TILE_PAD, np.int64),
        "current_discard_actor": np.full(
            batch, V4_PLAYER_PAD, np.int64
        ),
        "dealer": np.full(batch, V4_PLAYER_PAD, np.int64),
        "ready_kinds": np.zeros(
            (batch, V4_PLAYER_COUNT), np.int64
        ),
        "current_waits": np.zeros(
            (batch, TILE_TYPE_COUNT), np.float32
        ),
        "locked_ready_waits": np.zeros(
            (batch, TILE_TYPE_COUNT), np.float32
        ),
        "progress": np.zeros((batch, V4_PROGRESS_DIM), np.float32),
        "last_draw_kind": np.zeros(batch, np.int64),
        "hot_discard_player": np.full(batch, V4_PLAYER_PAD, np.int64),
        "action_legal": np.zeros((batch, ACTION_SPACE_SIZE), np.bool_),
        "action_supported": np.zeros((batch, ACTION_SPACE_SIZE), np.bool_),
        "action_type": np.full(
            (batch, ACTION_SPACE_SIZE), V4_ACTION_TYPE_PAD, np.int64
        ),
        "action_family": np.full(
            (batch, ACTION_SPACE_SIZE),
            _ACTION_FAMILY_INDICES[ActionFamily.UNSUPPORTED],
            np.int64,
        ),
        "action_tile": np.full(
            (batch, ACTION_SPACE_SIZE), V4_TILE_PAD, np.int64
        ),
        "action_source": np.full(
            (batch, ACTION_SPACE_SIZE), V4_PLAYER_PAD, np.int64
        ),
        "action_target": np.full(
            (batch, ACTION_SPACE_SIZE), V4_PLAYER_PAD, np.int64
        ),
        "action_scalar": np.zeros(
            (batch, ACTION_SPACE_SIZE, V4_ACTION_SCALAR_DIM), np.float32
        ),
        "action_ready_waits": np.zeros(
            (batch, ACTION_SPACE_SIZE, TILE_TYPE_COUNT), np.float32
        ),
        "discard_post_hand_summary": np.zeros(
            (batch, TILE_TYPE_COUNT, V4_HAND_SUMMARY_DIM), np.float32
        ),
        "discard_post_hand_tiles": np.zeros(
            (
                batch,
                TILE_TYPE_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
            np.float32,
        ),
        "pon_follow_up_mask": np.zeros(
            (batch, TILE_TYPE_COUNT), np.bool_
        ),
        "pon_follow_up_summary": np.zeros(
            (batch, TILE_TYPE_COUNT, V4_HAND_SUMMARY_DIM), np.float32
        ),
        "pon_follow_up_tiles": np.zeros(
            (
                batch,
                TILE_TYPE_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
            np.float32,
        ),
    }

    for row, decision in enumerate(decisions):
        observation = decision.observation
        public = decision.public_state
        (
            arrays["player_tiles"][row],
            arrays["player_summary"][row],
            concealed,
            visible,
        ) = _public_arrays(public)
        if not np.array_equal(
            concealed,
            np.asarray(observation.hand_counts, dtype=np.int16),
        ):
            raise ValueError("decision observation and public hand disagree")
        if not np.array_equal(
            visible,
            np.asarray(observation.visible_counts, dtype=np.int16),
        ):
            raise ValueError("decision observation and public visibility disagree")
        (
            arrays["hand_summary"][row],
            arrays["hand_tiles"][row],
        ) = _hand_arrays(public.hand, concealed, visible)
        (
            arrays["progress"][row],
            arrays["last_draw_kind"][row],
            arrays["hot_discard_player"][row],
        ) = _progress_arrays(decision.progress)

        discard_count = int(observation.discard_count)
        if discard_count < 0 or discard_count > MAX_DISCARD_EVENTS:
            raise ValueError("discard count exceeds V4 tensor capacity")
        arrays["discard_valid"][row, :discard_count] = True
        for index in range(discard_count):
            discard = observation.discards[index]
            arrays["discard_tiles"][row, index] = int(discard.tile)
            arrays["discard_players"][row, index] = int(
                discard.relative_actor
            )
            arrays["discard_called"][row, index] = float(
                discard.was_called
            )

        meld_count = int(observation.meld_count)
        if meld_count < 0 or meld_count > MAX_MELDS:
            raise ValueError("meld count exceeds V4 tensor capacity")
        for index in range(meld_count):
            meld = observation.melds[index]
            if not meld.valid:
                continue
            arrays["meld_valid"][row, index] = True
            arrays["meld_tiles"][row, index] = int(meld.tile)
            arrays["meld_owners"][row, index] = int(meld.relative_owner)
            arrays["meld_sources"][row, index] = int(meld.relative_source)
            arrays["meld_types"][row, index] = _enum_index(
                meld.type,
                _MELD_TYPE_INDICES,
                "meld type",
            )

        arrays["context"][row] = (
            float(observation.wall_remaining) / 108.0,
            _shanten_progress(max(-1, min(8, int(observation.shanten)))),
            float(discard_count) / MAX_DISCARD_EVENTS,
            float(meld_count) / MAX_MELDS,
        )
        arrays["phase"][row] = _enum_index(
            observation.phase,
            _PHASE_INDICES,
            "round phase",
        )
        for name, value, pad, maximum in (
            ("current_actor", observation.current_actor, V4_PLAYER_PAD, 4),
            ("current_discard", observation.current_discard, V4_TILE_PAD, 27),
            (
                "current_discard_actor",
                observation.current_discard_actor,
                V4_PLAYER_PAD,
                4,
            ),
            ("dealer", observation.dealer, V4_PLAYER_PAD, 4),
        ):
            value = int(value)
            arrays[name][row] = value if 0 <= value < maximum else pad
        ready_kinds = np.asarray(observation.ready_kinds, dtype=np.int64)
        if np.any(ready_kinds < 0) or np.any(ready_kinds >= 3):
            raise ValueError("observation contains an invalid ready kind")
        arrays["ready_kinds"][row] = ready_kinds
        arrays["current_waits"][row] = _mask_bits(int(observation.waits))
        arrays["locked_ready_waits"][row] = _mask_bits(
            int(observation.locked_ready_waits)
        )

        for action_id, candidate in enumerate(decision.actions):
            if not candidate.legal:
                continue
            arrays["action_legal"][row, action_id] = True
            arrays["action_supported"][row, action_id] = bool(
                candidate.supported_by_v4
            )
            semantics = candidate.semantics
            arrays["action_type"][row, action_id] = _enum_index(
                semantics.type,
                _ACTION_TYPE_INDICES,
                "action type",
            )
            arrays["action_family"][row, action_id] = _enum_index(
                semantics.family,
                _ACTION_FAMILY_INDICES,
                "action family",
            )
            tile = int(semantics.tile)
            arrays["action_tile"][row, action_id] = (
                tile if 0 <= tile < TILE_TYPE_COUNT else V4_TILE_PAD
            )
            for name, value in (
                ("action_source", int(semantics.relative_source)),
                ("action_target", int(semantics.relative_target)),
            ):
                arrays[name][row, action_id] = (
                    value if 0 <= value < V4_PLAYER_COUNT else V4_PLAYER_PAD
                )
            arrays["action_scalar"][row, action_id] = _action_scalar(candidate)
            arrays["action_ready_waits"][row, action_id] = _mask_bits(
                int(candidate.ready.wait_mask)
            )

            if candidate.post_hand_known:
                if tile >= TILE_TYPE_COUNT or concealed[tile] <= 0:
                    raise ValueError("discard post-hand has an invalid tile")
                post_concealed = concealed.copy()
                post_concealed[tile] -= 1
                summary, tile_features = _hand_arrays(
                    candidate.post_hand,
                    post_concealed,
                    visible,
                )
                arrays["discard_post_hand_summary"][row, tile] = summary
                arrays["discard_post_hand_tiles"][row, tile] = tile_features

        follow_up = decision.pon_follow_up
        if follow_up.available:
            pon_candidates = [
                candidate
                for candidate in decision.actions
                if candidate.legal and candidate.semantics.type == ActionType.PON
            ]
            if len(pon_candidates) != 1:
                raise ValueError("pon follow-up requires exactly one legal pon")
            pon_tile = int(pon_candidates[0].semantics.tile)
            after_pon = concealed.copy()
            if pon_tile >= TILE_TYPE_COUNT or after_pon[pon_tile] < 2:
                raise ValueError("pon follow-up cannot consume the called tile")
            after_pon[pon_tile] -= 2
            for discard_id, is_legal in enumerate(follow_up.discard_mask):
                if not is_legal:
                    continue
                if after_pon[discard_id] <= 0:
                    raise ValueError("pon follow-up discard is absent from hand")
                post_concealed = after_pon.copy()
                post_concealed[discard_id] -= 1
                summary, tile_features = _hand_arrays(
                    follow_up.post_discard_hands[discard_id],
                    post_concealed,
                    visible,
                )
                arrays["pon_follow_up_mask"][row, discard_id] = True
                arrays["pon_follow_up_summary"][row, discard_id] = summary
                arrays["pon_follow_up_tiles"][row, discard_id] = tile_features

    result = V4TensorBatch(
        **{
            name: torch.from_numpy(value)
            for name, value in arrays.items()
        },
        kan_adapter_features=tensorize_kan_adapter_features(decisions),
    )
    if device is not None:
        result = result.to(device)
    result.validate()
    return result


def tensorize_v4_decision(
    decision,
    device: torch.device | str | None = None,
) -> V4TensorBatch:
    return tensorize_v4_decisions([decision], device=device)
