from __future__ import annotations

from dataclasses import dataclass, field


FIXED_CHICKEN_TILE_IDS = frozenset((16, 18))
PLAYER_COUNT = 4


@dataclass
class ChongfengTracker:
    """Tracks each fixed chicken's global first and opening follows."""

    seen_fixed_chickens: set[int] = field(default_factory=set)
    player_has_discarded: list[bool] = field(
        default_factory=lambda: [False] * PLAYER_COUNT
    )

    def can_chongfeng(self, player_id: int, tile_id: int) -> bool:
        if not 0 <= player_id < PLAYER_COUNT:
            raise ValueError("chongfeng player id must be in range 0..3")
        if not 0 <= tile_id < 27:
            raise ValueError("chongfeng tile id must be in range 0..26")
        return (
            tile_id in FIXED_CHICKEN_TILE_IDS
            and (
                tile_id not in self.seen_fixed_chickens
                or not self.player_has_discarded[player_id]
            )
        )

    def observe_discard(self, player_id: int, tile_id: int) -> bool:
        if not 0 <= player_id < PLAYER_COUNT:
            raise ValueError("chongfeng player id must be in range 0..3")
        if not 0 <= tile_id < 27:
            raise ValueError("chongfeng tile id must be in range 0..26")

        is_opening_discard = not self.player_has_discarded[player_id]
        self.player_has_discarded[player_id] = True
        if tile_id not in FIXED_CHICKEN_TILE_IDS:
            return False

        is_global_first = tile_id not in self.seen_fixed_chickens
        self.seen_fixed_chickens.add(tile_id)
        return is_global_first or is_opening_discard


__all__ = ["ChongfengTracker", "FIXED_CHICKEN_TILE_IDS"]
