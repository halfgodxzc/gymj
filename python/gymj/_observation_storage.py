from typing import Dict

import numpy as np

from ._gymj import MAX_DISCARD_EVENTS, MAX_MELDS, TILE_TYPE_COUNT
from .reward import PLAYER_COUNT


def allocate_observation_arrays(row_count: int) -> Dict[str, np.ndarray]:
    return {
        "hand_counts": np.zeros(
            (row_count, TILE_TYPE_COUNT), dtype=np.uint8
        ),
        "discard_tiles": np.zeros(
            (row_count, MAX_DISCARD_EVENTS), dtype=np.uint8
        ),
        "discard_actors": np.zeros(
            (row_count, MAX_DISCARD_EVENTS), dtype=np.uint8
        ),
        "discard_was_called": np.zeros(
            (row_count, MAX_DISCARD_EVENTS), dtype=np.uint8
        ),
        "discard_count": np.zeros(row_count, dtype=np.uint8),
        "meld_tiles": np.zeros(
            (row_count, MAX_MELDS), dtype=np.uint8
        ),
        "meld_owners": np.zeros(
            (row_count, MAX_MELDS), dtype=np.uint8
        ),
        "meld_sources": np.zeros(
            (row_count, MAX_MELDS), dtype=np.uint8
        ),
        "meld_types": np.zeros(
            (row_count, MAX_MELDS), dtype=np.uint8
        ),
        "meld_valid": np.zeros(
            (row_count, MAX_MELDS), dtype=np.uint8
        ),
        "meld_count": np.zeros(row_count, dtype=np.uint8),
        "visible_counts": np.zeros(
            (row_count, TILE_TYPE_COUNT), dtype=np.uint8
        ),
        "waits": np.zeros(row_count, dtype=np.uint32),
        "shanten": np.zeros(row_count, dtype=np.int8),
        "phase": np.zeros(row_count, dtype=np.uint8),
        "current_actor": np.zeros(row_count, dtype=np.uint8),
        "current_discard": np.zeros(row_count, dtype=np.uint8),
        "current_discard_actor": np.zeros(row_count, dtype=np.uint8),
        "dealer": np.zeros(row_count, dtype=np.uint8),
        "wall_remaining": np.zeros(row_count, dtype=np.uint8),
        "ready_kinds": np.zeros(
            (row_count, PLAYER_COUNT), dtype=np.uint8
        ),
        "locked_ready_waits": np.zeros(row_count, dtype=np.uint32),
    }


def write_observation(
    arrays: Dict[str, np.ndarray],
    index: int,
    observation,
) -> None:
    arrays["hand_counts"][index] = observation.hand_counts

    for discard_index, discard in enumerate(observation.discards):
        arrays["discard_tiles"][index, discard_index] = discard.tile
        arrays["discard_actors"][index, discard_index] = (
            discard.relative_actor
        )
        arrays["discard_was_called"][index, discard_index] = (
            discard.was_called
        )
    arrays["discard_count"][index] = observation.discard_count

    for meld_index, meld in enumerate(observation.melds):
        arrays["meld_tiles"][index, meld_index] = meld.tile
        arrays["meld_owners"][index, meld_index] = meld.relative_owner
        arrays["meld_sources"][index, meld_index] = meld.relative_source
        arrays["meld_types"][index, meld_index] = int(meld.type)
        arrays["meld_valid"][index, meld_index] = meld.valid
    arrays["meld_count"][index] = observation.meld_count

    arrays["visible_counts"][index] = observation.visible_counts
    arrays["waits"][index] = observation.waits
    arrays["shanten"][index] = observation.shanten
    arrays["phase"][index] = int(observation.phase)
    arrays["current_actor"][index] = observation.current_actor
    arrays["current_discard"][index] = observation.current_discard
    arrays["current_discard_actor"][index] = (
        observation.current_discard_actor
    )
    arrays["dealer"][index] = observation.dealer
    arrays["wall_remaining"][index] = observation.wall_remaining
    arrays["ready_kinds"][index] = observation.ready_kinds
    arrays["locked_ready_waits"][index] = observation.locked_ready_waits
