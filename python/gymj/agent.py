import random
from typing import Optional, Protocol, Sequence

from ._gymj import ACTION_SPACE_SIZE, CompactObservation


class Agent(Protocol):
    def select_action_id(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> int:
        """Select one action id from the legal mask."""
        ...


class RandomAgent:
    """Uniformly sample one legal action id."""

    def __init__(self, seed: Optional[int] = None):
        self._random = random.Random(seed)

    def select_action_id(
        self,
        observation: CompactObservation,
        legal_mask: Sequence[int],
    ) -> int:
        del observation
        if len(legal_mask) != ACTION_SPACE_SIZE:
            raise ValueError(
                f"legal mask must contain {ACTION_SPACE_SIZE} entries"
            )

        legal_action_ids = [
            action_id
            for action_id, is_legal in enumerate(legal_mask)
            if is_legal
        ]
        if not legal_action_ids:
            raise ValueError("legal mask contains no legal actions")
        return self._random.choice(legal_action_ids)
