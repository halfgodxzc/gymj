import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    OBSERVATION_VERSION,
)
from .episode import EpisodeResult
from ._observation_storage import (
    allocate_observation_arrays,
    write_observation,
)
from .reward import PLAYER_COUNT
from .trajectory import Transition, is_action_legal


DATASET_FORMAT_VERSION = 1
PathLike = Union[str, os.PathLike]


class DatasetWriter:
    def __init__(
        self,
        directory: PathLike,
        episodes_per_shard: int = 1000,
        prefix: str = "shard",
        compressed: bool = True,
    ):
        if episodes_per_shard <= 0:
            raise ValueError("episodes per shard must be positive")
        if not prefix or Path(prefix).name != prefix:
            raise ValueError("prefix must be a non-empty file name")

        self.directory = Path(directory)
        self.episodes_per_shard = episodes_per_shard
        self.prefix = prefix
        self.compressed = compressed
        self._episodes: List[EpisodeResult] = []
        self._closed = False

        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_shard_index = self._find_next_shard_index()

    @property
    def buffered_episode_count(self) -> int:
        return len(self._episodes)

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, episode: EpisodeResult) -> Optional[Path]:
        self._ensure_open()
        self._validate_episode(episode)
        self._episodes.append(episode)
        if len(self._episodes) >= self.episodes_per_shard:
            return self._flush_buffer()
        return None

    def flush(self) -> Optional[Path]:
        self._ensure_open()
        return self._flush_buffer()

    def close(self) -> Optional[Path]:
        if self._closed:
            return None
        path = self._flush_buffer()
        self._closed = True
        return path

    def __enter__(self) -> "DatasetWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _flush_buffer(self) -> Optional[Path]:
        if not self._episodes:
            return None

        arrays = _episodes_to_arrays(self._episodes)
        shard_path = self.directory / (
            f"{self.prefix}_{self._next_shard_index:06d}.npz"
        )
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.directory,
                prefix=f".{shard_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                save = np.savez_compressed if self.compressed else np.savez
                save(temporary_file, **arrays)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, shard_path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        self._episodes.clear()
        self._next_shard_index += 1
        return shard_path

    def _find_next_shard_index(self) -> int:
        pattern = re.compile(
            rf"^{re.escape(self.prefix)}_(\d{{6}})\.npz$"
        )
        indices = []
        for path in self.directory.iterdir():
            match = pattern.match(path.name)
            if match:
                indices.append(int(match.group(1)))
        return max(indices, default=-1) + 1

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("dataset writer is closed")

    @staticmethod
    def _validate_episode(episode: EpisodeResult) -> None:
        if len(episode.trajectories) != PLAYER_COUNT:
            raise ValueError("episode must contain four player trajectories")
        if not -(1 << 63) <= episode.seed < (1 << 63):
            raise ValueError("episode seed must fit in int64")

        transition_count = sum(
            len(trajectory) for trajectory in episode.trajectories
        )
        if transition_count != episode.step_count:
            raise ValueError("episode step count does not match trajectories")

        for player_id, trajectory in enumerate(episode.trajectories):
            for transition in trajectory:
                _validate_transition(transition, player_id)

            reward_total = sum(
                transition.reward for transition in trajectory
            )
            if not math.isclose(
                reward_total,
                episode.reward_totals[player_id],
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "episode reward total does not match trajectories"
                )

        if len(set(episode.hu_players)) != len(episode.hu_players):
            raise ValueError("hu player ids must be unique")
        for player_id in episode.hu_players:
            if not 0 <= player_id < PLAYER_COUNT:
                raise ValueError("hu player id must be in range 0..3")


def _validate_transition(transition: Transition, player_id: int) -> None:
    if transition.player_id != player_id:
        raise ValueError("transition is stored in the wrong player trajectory")
    if not 0 <= transition.action_id < ACTION_SPACE_SIZE:
        raise ValueError("transition action id is outside the action space")
    if not is_action_legal(
        transition.legal_mask_words,
        transition.action_id,
    ):
        raise ValueError("transition action is not legal in its mask")


def _episodes_to_arrays(
    episodes: Sequence[EpisodeResult],
) -> Dict[str, np.ndarray]:
    episode_count = len(episodes)
    transition_count = sum(
        episode.step_count for episode in episodes
    )

    arrays = _allocate_arrays(episode_count, transition_count)
    transition_index = 0
    arrays["episode_transition_offsets"][0] = 0

    for episode_index, episode in enumerate(episodes):
        arrays["episode_seed"][episode_index] = episode.seed
        arrays["episode_reward_totals"][episode_index] = (
            episode.reward_totals
        )
        arrays["episode_end_reason"][episode_index] = int(
            episode.end_reason
        )
        hu_mask = 0
        for player_id in episode.hu_players:
            hu_mask |= 1 << player_id
        arrays["episode_hu_mask"][episode_index] = hu_mask

        for player_id, trajectory in enumerate(episode.trajectories):
            for player_step, transition in enumerate(trajectory):
                _write_transition(
                    arrays,
                    transition_index,
                    episode_index,
                    player_step,
                    transition,
                )
                transition_index += 1

        arrays["episode_transition_offsets"][episode_index + 1] = (
            transition_index
        )

    return arrays


def _allocate_arrays(
    episode_count: int,
    transition_count: int,
) -> Dict[str, np.ndarray]:
    arrays = allocate_observation_arrays(transition_count)
    arrays.update({
        "dataset_format_version": np.asarray(
            DATASET_FORMAT_VERSION, dtype=np.uint32
        ),
        "observation_version": np.asarray(
            OBSERVATION_VERSION, dtype=np.uint32
        ),
        "action_encoding_version": np.asarray(
            ACTION_ENCODING_VERSION, dtype=np.uint32
        ),
        "episode_count": np.asarray(episode_count, dtype=np.uint32),
        "transition_count": np.asarray(
            transition_count, dtype=np.uint64
        ),
        "episode_seed": np.empty(episode_count, dtype=np.int64),
        "episode_transition_offsets": np.empty(
            episode_count + 1, dtype=np.uint64
        ),
        "episode_reward_totals": np.empty(
            (episode_count, PLAYER_COUNT), dtype=np.float32
        ),
        "episode_end_reason": np.empty(episode_count, dtype=np.uint8),
        "episode_hu_mask": np.empty(episode_count, dtype=np.uint8),
        "episode_index": np.empty(transition_count, dtype=np.uint32),
        "player_id": np.empty(transition_count, dtype=np.uint8),
        "player_step": np.empty(transition_count, dtype=np.uint32),
        "legal_mask_words": np.empty(
            (transition_count, 2), dtype=np.uint64
        ),
        "action_id": np.empty(transition_count, dtype=np.uint8),
        "reward": np.empty(transition_count, dtype=np.float32),
        "done": np.empty(transition_count, dtype=np.bool_),
    })
    return arrays


def _write_transition(
    arrays: Dict[str, np.ndarray],
    index: int,
    episode_index: int,
    player_step: int,
    transition: Transition,
) -> None:
    arrays["episode_index"][index] = episode_index
    arrays["player_id"][index] = transition.player_id
    arrays["player_step"][index] = player_step
    write_observation(arrays, index, transition.observation)
    arrays["legal_mask_words"][index] = transition.legal_mask_words
    arrays["action_id"][index] = transition.action_id
    arrays["reward"][index] = transition.reward
    arrays["done"][index] = transition.done
