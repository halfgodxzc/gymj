from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, fields, replace
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ._gymj import (
    ACTION_ENCODING_VERSION,
    DECISION_FEATURE_VERSION,
    PRIVILEGED_PATTERN_VERSION,
    PRIVILEGED_STATE_VERSION,
)
from .v4_critic import (
    V4_PRIVILEGED_TENSOR_VERSION,
    V4CriticPublicTensorBatch,
    V4PrivilegedTensorBatch,
    tensorize_v4_critic_public_states,
    tensorize_v4_privileged_states,
)
from .v4_critic_rollout import V4CriticSample
from .v4_features import V4_TENSOR_VERSION


V4_CRITIC_DATASET_FORMAT_VERSION = 1
V4_CRITIC_DATASET_PREFIX = "v4_critic"

_PUBLIC_PREFIX = "public__"
_PRIVILEGED_PREFIX = "privileged__"
_PUBLIC_FIELDS = tuple(
    item.name for item in fields(V4CriticPublicTensorBatch)
)
_PRIVILEGED_FIELDS = tuple(
    item.name for item in fields(V4PrivilegedTensorBatch)
)
_SAMPLE_FIELDS = (
    "episode_seed",
    "player_id",
    "decision_index",
    "action_id",
    "score_return",
    "shanten_return",
)
_PUBLIC_BOOL_FIELDS = {"discard_valid", "meld_valid", "action_legal"}
_PUBLIC_INTEGER_FIELDS = {
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
}
_PRIVILEGED_BOOL_FIELDS = {
    "future_draw_valid",
    "replacement_draw_valid",
}
_PRIVILEGED_INTEGER_FIELDS = {
    "future_draw_tiles",
    "replacement_draw_tiles",
}
_VERSION_FIELDS = {
    "v4_critic_dataset_format_version": V4_CRITIC_DATASET_FORMAT_VERSION,
    "v4_tensor_version": V4_TENSOR_VERSION,
    "v4_privileged_tensor_version": V4_PRIVILEGED_TENSOR_VERSION,
    "decision_feature_version": DECISION_FEATURE_VERSION,
    "privileged_state_version": PRIVILEGED_STATE_VERSION,
    "privileged_pattern_version": PRIVILEGED_PATTERN_VERSION,
    "action_encoding_version": ACTION_ENCODING_VERSION,
}


@dataclass(frozen=True)
class V4CriticBatch:
    public: V4CriticPublicTensorBatch
    privileged: V4PrivilegedTensorBatch
    episode_seed: torch.Tensor
    player_id: torch.Tensor
    decision_index: torch.Tensor
    action_id: torch.Tensor
    score_return: torch.Tensor
    shanten_return: torch.Tensor

    def __len__(self) -> int:
        return self.public.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "V4CriticBatch":
        return replace(
            self,
            public=self.public.to(device, non_blocking=non_blocking),
            privileged=self.privileged.to(
                device,
                non_blocking=non_blocking,
            ),
            **{
                name: getattr(self, name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for name in _SAMPLE_FIELDS
            },
        )

    def pin_memory(self) -> "V4CriticBatch":
        return replace(
            self,
            public=replace(self.public, **{
                item.name: getattr(self.public, item.name).pin_memory()
                for item in fields(self.public)
            }),
            privileged=replace(self.privileged, **{
                item.name: getattr(self.privileged, item.name).pin_memory()
                for item in fields(self.privileged)
            }),
            **{
                name: getattr(self, name).pin_memory()
                for name in _SAMPLE_FIELDS
            },
        )


class V4CriticWriter:
    def __init__(
        self,
        directory: Path | str,
        *,
        samples_per_shard: int = 256,
        prefix: str = V4_CRITIC_DATASET_PREFIX,
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
        self._samples: list[V4CriticSample] = []
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_shard_index = self._find_next_shard_index()

    @property
    def buffered_sample_count(self) -> int:
        return len(self._samples)

    def append(self, sample: V4CriticSample) -> Path | None:
        self._ensure_open()
        if not isinstance(sample, V4CriticSample):
            raise TypeError("writer accepts V4CriticSample objects")
        self._samples.append(sample)
        if len(self._samples) >= self.samples_per_shard:
            return self._flush_buffer()
        return None

    def append_many(
        self,
        samples: Iterable[V4CriticSample],
    ) -> tuple[Path, ...]:
        paths = []
        for sample in samples:
            path = self.append(sample)
            if path is not None:
                paths.append(path)
        return tuple(paths)

    def flush(self) -> Path | None:
        self._ensure_open()
        return self._flush_buffer()

    def close(self) -> Path | None:
        if self._closed:
            return None
        path = self._flush_buffer()
        self._closed = True
        return path

    def __enter__(self) -> "V4CriticWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _flush_buffer(self) -> Path | None:
        if not self._samples:
            return None
        destination = self.directory / (
            f"{self.prefix}_{self._next_shard_index:06d}.npz"
        )
        _save_npz_atomic(
            destination,
            _samples_to_arrays(self._samples),
            self.compressed,
        )
        self._samples.clear()
        self._next_shard_index += 1
        return destination

    def _find_next_shard_index(self) -> int:
        pattern = re.compile(rf"^{re.escape(self.prefix)}_(\d{{6}})\.npz$")
        indices = [
            int(match.group(1))
            for path in self.directory.iterdir()
            if (match := pattern.match(path.name))
        ]
        return max(indices, default=-1) + 1

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("V4 Critic writer is closed")


@dataclass(frozen=True)
class _ShardInfo:
    path: Path
    start: int
    count: int


class V4CriticDataset(Dataset):
    """Lazy sharded V4 Critic dataset with episode-level indices."""

    def __init__(
        self,
        directory: Path | str,
        *,
        prefix: str = V4_CRITIC_DATASET_PREFIX,
        cache_shards: int = 1,
    ):
        if cache_shards <= 0:
            raise ValueError("cache shards must be positive")
        self.directory = Path(directory)
        paths = sorted(self.directory.glob(f"{prefix}_*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"no V4 Critic shards found in {self.directory}"
            )
        self._cache_limit = cache_shards
        self._cache: OrderedDict[int, Mapping[str, np.ndarray]] = OrderedDict()
        self._shards: list[_ShardInfo] = []
        episodes = []
        players = []
        decisions = []
        start = 0
        for path in paths:
            count, shard_episodes, shard_players, shard_decisions = (
                _read_shard_index(path)
            )
            self._shards.append(_ShardInfo(path, start, count))
            start += count
            episodes.append(shard_episodes)
            players.append(shard_players)
            decisions.append(shard_decisions)
        self._length = start
        self._starts = [shard.start for shard in self._shards]
        self._episode_seeds = np.concatenate(episodes)
        self._player_ids = np.concatenate(players)
        self._decision_indices = np.concatenate(decisions)

    def __len__(self) -> int:
        return self._length

    @property
    def episode_seeds(self) -> np.ndarray:
        return self._episode_seeds

    @property
    def player_ids(self) -> np.ndarray:
        return self._player_ids

    @property
    def decision_indices(self) -> np.ndarray:
        return self._decision_indices

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def shard_index_for_sample(self, index: int) -> int:
        index = self._normalize_index(index)
        return bisect_right(self._starts, index) - 1

    def __getitem__(self, index: int) -> dict[str, object]:
        index = self._normalize_index(index)
        shard_index = self.shard_index_for_sample(index)
        info = self._shards[shard_index]
        local_index = index - info.start
        arrays = self._load_shard(shard_index)
        return {
            "public": {
                name: torch.as_tensor(arrays[_PUBLIC_PREFIX + name][local_index])
                for name in _PUBLIC_FIELDS
            },
            "privileged": {
                name: torch.as_tensor(
                    arrays[_PRIVILEGED_PREFIX + name][local_index]
                )
                for name in _PRIVILEGED_FIELDS
            },
            **{
                name: torch.as_tensor(arrays[name][local_index])
                for name in _SAMPLE_FIELDS
            },
        }

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("dataset index out of range")
        return index

    def _load_shard(self, shard_index: int) -> Mapping[str, np.ndarray]:
        if shard_index in self._cache:
            arrays = self._cache.pop(shard_index)
            self._cache[shard_index] = arrays
            return arrays
        info = self._shards[shard_index]
        with np.load(info.path, allow_pickle=False) as shard:
            _validate_shard_header(info.path, shard)
            arrays = {
                name: np.ascontiguousarray(shard[name])
                for name in _stored_fields()
            }
        _validate_loaded_arrays(info.path, arrays, info.count)
        self._cache[shard_index] = arrays
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return arrays


class V4CriticShardBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: V4CriticDataset,
        indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if np.any((self.indices < 0) | (self.indices >= len(dataset))):
            raise ValueError("batch sampler index is outside the dataset")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        grouped: dict[int, list[int]] = {}
        for index in self.indices:
            grouped.setdefault(
                dataset.shard_index_for_sample(int(index)),
                [],
            ).append(int(index))
        self._groups = grouped

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        shard_order = list(self._groups)
        if self.shuffle:
            rng.shuffle(shard_order)
        for shard_index in shard_order:
            indices = np.asarray(self._groups[shard_index], dtype=np.int64)
            if self.shuffle:
                indices = rng.permutation(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._groups.values()
        )


def make_v4_critic_data_loader(
    dataset: V4CriticDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=V4CriticShardBatchSampler(
            dataset,
            indices,
            batch_size,
            shuffle=shuffle,
            seed=seed,
        ),
        collate_fn=collate_v4_critic_samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def collate_v4_critic_samples(
    samples: Sequence[Mapping[str, object]],
) -> V4CriticBatch:
    if not samples:
        raise ValueError("cannot collate an empty V4 Critic batch")
    public = V4CriticPublicTensorBatch(**{
        name: _to_runtime_dtype(
            name,
            torch.stack([sample["public"][name] for sample in samples]),
            privileged=False,
        )
        for name in _PUBLIC_FIELDS
    })
    privileged = V4PrivilegedTensorBatch(**{
        name: _to_runtime_dtype(
            name,
            torch.stack([sample["privileged"][name] for sample in samples]),
            privileged=True,
        )
        for name in _PRIVILEGED_FIELDS
    })
    public.validate()
    privileged.validate()
    return V4CriticBatch(
        public=public,
        privileged=privileged,
        episode_seed=torch.stack([
            sample["episode_seed"] for sample in samples
        ]).long(),
        player_id=torch.stack([
            sample["player_id"] for sample in samples
        ]).long(),
        decision_index=torch.stack([
            sample["decision_index"] for sample in samples
        ]).long(),
        action_id=torch.stack([
            sample["action_id"] for sample in samples
        ]).long(),
        score_return=torch.stack([
            sample["score_return"] for sample in samples
        ]).float(),
        shanten_return=torch.stack([
            sample["shanten_return"] for sample in samples
        ]).float(),
    )


def _samples_to_arrays(
    samples: Sequence[V4CriticSample],
) -> dict[str, np.ndarray]:
    decisions = [sample.decision_features for sample in samples]
    public = tensorize_v4_critic_public_states(decisions)
    privileged = tensorize_v4_privileged_states(
        decisions,
        [sample.privileged_state for sample in samples],
        [sample.privileged_patterns for sample in samples],
    )
    arrays = {
        name: np.asarray(value)
        for name, value in _VERSION_FIELDS.items()
    }
    arrays["sample_count"] = np.asarray(len(samples), dtype=np.uint64)
    for item in fields(public):
        value = getattr(public, item.name).detach().cpu().numpy()
        arrays[_PUBLIC_PREFIX + item.name] = _to_storage_array(
            item.name,
            value,
            privileged=False,
        )
    for item in fields(privileged):
        value = getattr(privileged, item.name).detach().cpu().numpy()
        arrays[_PRIVILEGED_PREFIX + item.name] = _to_storage_array(
            item.name,
            value,
            privileged=True,
        )
    arrays["episode_seed"] = np.asarray(
        [sample.episode_seed for sample in samples],
        dtype=np.int64,
    )
    arrays["player_id"] = np.asarray(
        [sample.player_id for sample in samples],
        dtype=np.uint8,
    )
    arrays["decision_index"] = np.asarray(
        [sample.decision_index for sample in samples],
        dtype=np.uint32,
    )
    arrays["action_id"] = np.asarray(
        [sample.action_id for sample in samples],
        dtype=np.uint8,
    )
    arrays["score_return"] = np.asarray(
        [sample.score_return for sample in samples],
        dtype=np.float32,
    )
    arrays["shanten_return"] = np.asarray(
        [sample.shanten_return for sample in samples],
        dtype=np.float32,
    )
    return arrays


def _read_shard_index(
    path: Path,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as shard:
        _validate_shard_header(path, shard)
        count = int(shard["sample_count"])
        episodes = np.asarray(shard["episode_seed"], dtype=np.int64).copy()
        players = np.asarray(shard["player_id"], dtype=np.uint8).copy()
        decisions = np.asarray(shard["decision_index"], dtype=np.uint32).copy()
    if (
        episodes.shape != (count,)
        or players.shape != (count,)
        or decisions.shape != (count,)
    ):
        raise ValueError(f"{path} has invalid V4 Critic index arrays")
    return count, episodes, players, decisions


def _stored_fields() -> tuple[str, ...]:
    return tuple(
        [_PUBLIC_PREFIX + name for name in _PUBLIC_FIELDS]
        + [_PRIVILEGED_PREFIX + name for name in _PRIVILEGED_FIELDS]
        + list(_SAMPLE_FIELDS)
    )


def _validate_shard_header(path: Path, shard) -> None:
    for name, expected in _VERSION_FIELDS.items():
        if name not in shard or int(shard[name]) != expected:
            raise ValueError(f"{path} has an incompatible {name}")
    if "sample_count" not in shard:
        raise ValueError(f"{path} has no sample count")
    missing = set(_stored_fields()) - set(shard.files)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")


def _validate_loaded_arrays(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    count: int,
) -> None:
    for name, array in arrays.items():
        if array.shape[0] != count:
            raise ValueError(
                f"{path}:{name} has {array.shape[0]} rows, expected {count}"
            )


def _to_storage_array(
    name: str,
    array: np.ndarray,
    *,
    privileged: bool,
) -> np.ndarray:
    bool_fields = (
        _PRIVILEGED_BOOL_FIELDS if privileged else _PUBLIC_BOOL_FIELDS
    )
    integer_fields = (
        _PRIVILEGED_INTEGER_FIELDS if privileged else _PUBLIC_INTEGER_FIELDS
    )
    if name in bool_fields:
        dtype = np.bool_
    elif name in integer_fields:
        dtype = np.int16
    else:
        dtype = np.float16
    return np.ascontiguousarray(array, dtype=dtype)


def _to_runtime_dtype(
    name: str,
    tensor: torch.Tensor,
    *,
    privileged: bool,
) -> torch.Tensor:
    bool_fields = (
        _PRIVILEGED_BOOL_FIELDS if privileged else _PUBLIC_BOOL_FIELDS
    )
    integer_fields = (
        _PRIVILEGED_INTEGER_FIELDS if privileged else _PUBLIC_INTEGER_FIELDS
    )
    if name in bool_fields:
        return tensor.bool()
    if name in integer_fields:
        return tensor.long()
    return tensor.float()


def _save_npz_atomic(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    compressed: bool,
) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.stem}_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            save = np.savez_compressed if compressed else np.savez
            save(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "V4_CRITIC_DATASET_FORMAT_VERSION",
    "V4_CRITIC_DATASET_PREFIX",
    "V4CriticBatch",
    "V4CriticDataset",
    "V4CriticShardBatchSampler",
    "V4CriticWriter",
    "collate_v4_critic_samples",
    "make_v4_critic_data_loader",
]
