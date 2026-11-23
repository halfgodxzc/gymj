import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ._gymj import ACTION_ENCODING_VERSION, OBSERVATION_VERSION
from ._observation_storage import (
    allocate_observation_arrays,
    write_observation,
)
from .demonstration import DemonstrationSample


DEMONSTRATION_FORMAT_VERSION = 1
PathLike = Union[str, os.PathLike]


class DemonstrationWriter:
    def __init__(
        self,
        directory: PathLike,
        samples_per_shard: int = 10_000,
        prefix: str = "demonstration",
        compressed: bool = True,
    ):
        if samples_per_shard <= 0:
            raise ValueError("samples per shard must be positive")
        if not prefix or Path(prefix).name != prefix:
            raise ValueError("prefix must be a non-empty file name")

        self.directory = Path(directory)
        self.samples_per_shard = samples_per_shard
        self.prefix = prefix
        self.compressed = compressed
        self._samples: List[DemonstrationSample] = []
        self._closed = False

        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_shard_index = self._find_next_shard_index()

    @property
    def buffered_sample_count(self) -> int:
        return len(self._samples)

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, sample: DemonstrationSample) -> Optional[Path]:
        self._ensure_open()
        if not isinstance(sample, DemonstrationSample):
            raise TypeError("writer accepts DemonstrationSample objects")
        self._samples.append(sample)
        if len(self._samples) >= self.samples_per_shard:
            return self._flush_buffer()
        return None

    def append_many(
        self,
        samples: Iterable[DemonstrationSample],
    ) -> Tuple[Path, ...]:
        paths = []
        for sample in samples:
            path = self.append(sample)
            if path is not None:
                paths.append(path)
        return tuple(paths)

    def flush(self) -> Optional[Path]:
        self._ensure_open()
        return self._flush_buffer()

    def close(self) -> Optional[Path]:
        if self._closed:
            return None
        path = self._flush_buffer()
        self._closed = True
        return path

    def __enter__(self) -> "DemonstrationWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _flush_buffer(self) -> Optional[Path]:
        if not self._samples:
            return None

        arrays = _samples_to_arrays(self._samples)
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

        self._samples.clear()
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
            raise RuntimeError("demonstration writer is closed")


def _samples_to_arrays(
    samples: List[DemonstrationSample],
) -> Dict[str, np.ndarray]:
    sample_count = len(samples)
    arrays = allocate_observation_arrays(sample_count)
    arrays.update({
        "demonstration_format_version": np.asarray(
            DEMONSTRATION_FORMAT_VERSION,
            dtype=np.uint32,
        ),
        "observation_version": np.asarray(
            OBSERVATION_VERSION,
            dtype=np.uint32,
        ),
        "action_encoding_version": np.asarray(
            ACTION_ENCODING_VERSION,
            dtype=np.uint32,
        ),
        "sample_count": np.asarray(sample_count, dtype=np.uint64),
        "episode_seed": np.empty(sample_count, dtype=np.int64),
        "player_id": np.empty(sample_count, dtype=np.uint8),
        "legal_mask_words": np.empty(
            (sample_count, 2), dtype=np.uint64
        ),
        "supervision_mask_words": np.empty(
            (sample_count, 2), dtype=np.uint64
        ),
        "target_action_mask_words": np.empty(
            (sample_count, 2), dtype=np.uint64
        ),
        "selected_action_id": np.empty(sample_count, dtype=np.uint8),
    })

    for index, sample in enumerate(samples):
        write_observation(arrays, index, sample.observation)
        arrays["episode_seed"][index] = sample.episode_seed
        arrays["player_id"][index] = sample.player_id
        arrays["legal_mask_words"][index] = sample.legal_mask_words
        arrays["supervision_mask_words"][index] = (
            sample.supervision_mask_words
        )
        arrays["target_action_mask_words"][index] = (
            sample.target_action_mask_words
        )
        arrays["selected_action_id"][index] = sample.selected_action_id

    return arrays
