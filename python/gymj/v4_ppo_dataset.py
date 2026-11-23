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
from .kan_adapter import KAN_ADAPTER_FEATURE_DIM
from .v4_critic import (
    V4_PRIVILEGED_TENSOR_VERSION,
    V4PrivilegedTensorBatch,
    tensorize_v4_privileged_states,
)
from .v4_features import (
    V4_TENSOR_VERSION,
    V4TensorBatch,
    tensorize_v4_decisions,
)
from .v4_ppo_rollout import V4PPOSample


V4_PPO_DATASET_FORMAT_VERSION = 1
V4_PPO_DATASET_PREFIX = "v4_ppo"

_ACTOR_PREFIX = "actor__"
_PRIVILEGED_PREFIX = "privileged__"
_ACTOR_FIELDS = tuple(item.name for item in fields(V4TensorBatch))
_OPTIONAL_ACTOR_FIELDS = ("kan_adapter_features",)
_PRIVILEGED_FIELDS = tuple(
    item.name for item in fields(V4PrivilegedTensorBatch)
)
_SAMPLE_FIELDS = (
    "episode_seed",
    "player_id",
    "decision_index",
    "action_id",
    "old_log_probability",
    "old_entropy",
    "actor_trainable",
    "score_return",
    "shanten_return",
    "score_returns_all",
    "chicken_attribution_points",
)
_ACTOR_BOOL_FIELDS = {
    "action_legal",
    "action_supported",
    "discard_valid",
    "meld_valid",
    "pon_follow_up_mask",
}
_ACTOR_INTEGER_FIELDS = {
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
_PRIVILEGED_BOOL_FIELDS = {
    "future_draw_valid",
    "replacement_draw_valid",
}
_PRIVILEGED_INTEGER_FIELDS = {
    "future_draw_tiles",
    "replacement_draw_tiles",
}
_VERSION_FIELDS = {
    "v4_ppo_dataset_format_version": V4_PPO_DATASET_FORMAT_VERSION,
    "v4_tensor_version": V4_TENSOR_VERSION,
    "v4_privileged_tensor_version": V4_PRIVILEGED_TENSOR_VERSION,
    "decision_feature_version": DECISION_FEATURE_VERSION,
    "privileged_state_version": PRIVILEGED_STATE_VERSION,
    "privileged_pattern_version": PRIVILEGED_PATTERN_VERSION,
    "action_encoding_version": ACTION_ENCODING_VERSION,
}


@dataclass(frozen=True)
class V4PPOBatch:
    features: V4TensorBatch
    privileged: V4PrivilegedTensorBatch
    sample_index: torch.Tensor
    episode_seed: torch.Tensor
    player_id: torch.Tensor
    decision_index: torch.Tensor
    action_id: torch.Tensor
    old_log_probability: torch.Tensor
    old_entropy: torch.Tensor
    actor_trainable: torch.Tensor
    score_return: torch.Tensor
    shanten_return: torch.Tensor
    score_returns_all: torch.Tensor
    chicken_attribution_points: torch.Tensor

    def __len__(self) -> int:
        return self.features.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "V4PPOBatch":
        return replace(
            self,
            features=self.features.to(device),
            privileged=self.privileged.to(
                device,
                non_blocking=non_blocking,
            ),
            **{
                name: getattr(self, name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for name in ("sample_index",) + _SAMPLE_FIELDS
            },
        )

    def pin_memory(self) -> "V4PPOBatch":
        return replace(
            self,
            features=replace(self.features, **{
                item.name: getattr(self.features, item.name).pin_memory()
                for item in fields(self.features)
            }),
            privileged=replace(self.privileged, **{
                item.name: getattr(self.privileged, item.name).pin_memory()
                for item in fields(self.privileged)
            }),
            **{
                name: getattr(self, name).pin_memory()
                for name in ("sample_index",) + _SAMPLE_FIELDS
            },
        )


class V4PPOWriter:
    def __init__(
        self,
        directory: Path | str,
        *,
        samples_per_shard: int = 128,
        prefix: str = V4_PPO_DATASET_PREFIX,
        compressed: bool = True,
        actor_float32: bool = False,
    ):
        if samples_per_shard <= 0:
            raise ValueError("samples per shard must be positive")
        if not prefix or Path(prefix).name != prefix:
            raise ValueError("prefix must be a non-empty file name")
        self.directory = Path(directory)
        self.samples_per_shard = samples_per_shard
        self.prefix = prefix
        self.compressed = compressed
        self.actor_float32 = actor_float32
        self._samples: list[V4PPOSample] = []
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_shard_index = self._find_next_shard_index()

    def append(self, sample: V4PPOSample) -> Path | None:
        self._ensure_open()
        if not isinstance(sample, V4PPOSample):
            raise TypeError("writer accepts V4PPOSample objects")
        self._samples.append(sample)
        if len(self._samples) >= self.samples_per_shard:
            return self._flush_buffer()
        return None

    def append_many(
        self,
        samples: Iterable[V4PPOSample],
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

    def __enter__(self) -> "V4PPOWriter":
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
            _samples_to_arrays(self._samples, actor_float32=self.actor_float32),
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
            raise RuntimeError("V4 PPO writer is closed")


@dataclass(frozen=True)
class _ShardInfo:
    path: Path
    start: int
    count: int


class V4PPODataset(Dataset):
    def __init__(
        self,
        directory: Path | str,
        *,
        prefix: str = V4_PPO_DATASET_PREFIX,
        cache_shards: int = 1,
    ):
        if cache_shards <= 0:
            raise ValueError("cache shards must be positive")
        self.directory = Path(directory)
        paths = sorted(self.directory.glob(f"{prefix}_*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"no V4 PPO shards found in {self.directory}"
            )
        self._cache_limit = cache_shards
        self._cache: OrderedDict[int, Mapping[str, np.ndarray]] = OrderedDict()
        self._shards = []
        episodes = []
        players = []
        decisions = []
        start = 0
        for path in paths:
            count, episode, player, decision = _read_shard_index(path)
            self._shards.append(_ShardInfo(path, start, count))
            start += count
            episodes.append(episode)
            players.append(player)
            decisions.append(decision)
        self._length = start
        self._starts = [item.start for item in self._shards]
        self._episode_seeds = np.concatenate(episodes)
        self._player_ids = np.concatenate(players)
        self._decision_indices = np.concatenate(decisions)

    def __len__(self) -> int:
        return self._length

    @property
    def episode_seeds(self) -> np.ndarray:
        return self._episode_seeds

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
        arrays = self._load_shard(shard_index)
        local = index - info.start
        return {
            "sample_index": torch.as_tensor(index),
            "features": {
                name: torch.as_tensor(
                    arrays[_ACTOR_PREFIX + name][local]
                )
                for name in _ACTOR_FIELDS
            },
            "privileged": {
                name: torch.as_tensor(
                    arrays[_PRIVILEGED_PREFIX + name][local]
                )
                for name in _PRIVILEGED_FIELDS
            },
            **{
                name: torch.as_tensor(np.asarray(arrays[name][local]))
                for name in _SAMPLE_FIELDS
            },
        }

    def _normalize_index(self, index: int) -> int:
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
                name: (
                    np.ascontiguousarray(shard[name])
                    if name in shard
                    else np.full((info.count, 4), np.nan, dtype=np.float32)
                    if name == "score_returns_all"
                    else np.full(info.count, np.nan, dtype=np.float32)
                    if name == "chicken_attribution_points"
                    else np.zeros(
                        (
                            info.count,
                            ACTION_SPACE_SIZE,
                            KAN_ADAPTER_FEATURE_DIM,
                        ),
                        dtype=np.float16,
                    )
                )
                for name in _stored_fields()
            }
        _validate_loaded_arrays(info.path, arrays, info.count)
        self._cache[shard_index] = arrays
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return arrays


class V4PPOShardBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: V4PPODataset,
        indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        groups: dict[int, list[int]] = {}
        for raw_index in indices:
            index = int(raw_index)
            groups.setdefault(
                dataset.shard_index_for_sample(index),
                [],
            ).append(index)
        self._groups = groups

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed)
        shard_ids = np.asarray(list(self._groups), dtype=np.int64)
        if self.shuffle:
            generator.shuffle(shard_ids)
        for shard_id in shard_ids:
            indices = np.asarray(self._groups[int(shard_id)], dtype=np.int64)
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._groups.values()
        )


def make_v4_ppo_data_loader(
    dataset: V4PPODataset,
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
        batch_sampler=V4PPOShardBatchSampler(
            dataset,
            indices,
            batch_size,
            shuffle=shuffle,
            seed=seed,
        ),
        collate_fn=collate_v4_ppo_samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def collate_v4_ppo_samples(
    samples: Sequence[Mapping[str, object]],
) -> V4PPOBatch:
    if not samples:
        raise ValueError("cannot collate an empty V4 PPO batch")
    actor_values = {
        name: torch.stack([sample["features"][name] for sample in samples])
        for name in _ACTOR_FIELDS
    }
    privileged_values = {
        name: torch.stack([sample["privileged"][name] for sample in samples])
        for name in _PRIVILEGED_FIELDS
    }
    features = V4TensorBatch(**{
        name: _runtime_dtype(name, tensor, privileged=False)
        for name, tensor in actor_values.items()
    })
    privileged = V4PrivilegedTensorBatch(**{
        name: _runtime_dtype(name, tensor, privileged=True)
        for name, tensor in privileged_values.items()
    })
    features.validate()
    privileged.validate()
    return V4PPOBatch(
        features=features,
        privileged=privileged,
        sample_index=torch.stack([
            sample["sample_index"] for sample in samples
        ]).long(),
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
        old_log_probability=torch.stack([
            sample["old_log_probability"] for sample in samples
        ]).float(),
        old_entropy=torch.stack([
            sample["old_entropy"] for sample in samples
        ]).float(),
        actor_trainable=torch.stack([
            sample["actor_trainable"] for sample in samples
        ]).bool(),
        score_return=torch.stack([
            sample["score_return"] for sample in samples
        ]).float(),
        shanten_return=torch.stack([
            sample["shanten_return"] for sample in samples
        ]).float(),
        score_returns_all=torch.stack([sample["score_returns_all"] for sample in samples]).float(),
        chicken_attribution_points=torch.stack([
            sample["chicken_attribution_points"] for sample in samples
        ]).float(),
    )


def _samples_to_arrays(
    samples: Sequence[V4PPOSample],
    *,
    actor_float32: bool = False,
) -> dict[str, np.ndarray]:
    actor = tensorize_v4_decisions([
        sample.decision_features for sample in samples
    ])
    privileged = tensorize_v4_privileged_states(
        [sample.decision_features for sample in samples],
        [sample.privileged_state for sample in samples],
        [sample.privileged_patterns for sample in samples],
    )
    arrays = {
        name: np.asarray(value)
        for name, value in _VERSION_FIELDS.items()
    }
    arrays["sample_count"] = np.asarray(len(samples), dtype=np.uint64)
    arrays["chicken_attribution_version"] = np.asarray(1, dtype=np.uint32)
    for item in fields(actor):
        value = getattr(actor, item.name).detach().cpu().numpy()
        # Preserve the behavior-policy inputs when strict likelihood matching
        # matters. Existing collectors retain their compact default storage.
        arrays[_ACTOR_PREFIX + item.name] = (
            np.ascontiguousarray(value, dtype=np.float32)
            if actor_float32 and value.dtype.kind == "f"
            else _storage_array(item.name, value, privileged=False)
        )
    for item in fields(privileged):
        value = getattr(privileged, item.name).detach().cpu().numpy()
        arrays[_PRIVILEGED_PREFIX + item.name] = _storage_array(
            item.name,
            value,
            privileged=True,
        )
    arrays.update({
        "episode_seed": np.asarray(
            [sample.episode_seed for sample in samples], dtype=np.int64
        ),
        "player_id": np.asarray(
            [sample.player_id for sample in samples], dtype=np.uint8
        ),
        "decision_index": np.asarray(
            [sample.decision_index for sample in samples], dtype=np.uint32
        ),
        "action_id": np.asarray(
            [sample.action_id for sample in samples], dtype=np.uint8
        ),
        "old_log_probability": np.asarray(
            [sample.old_log_probability for sample in samples],
            dtype=np.float32,
        ),
        "old_entropy": np.asarray(
            [sample.old_entropy for sample in samples], dtype=np.float32
        ),
        "actor_trainable": np.asarray(
            [sample.actor_trainable for sample in samples], dtype=np.bool_
        ),
        "score_return": np.asarray(
            [sample.score_return for sample in samples], dtype=np.float32
        ),
        "shanten_return": np.asarray(
            [sample.shanten_return for sample in samples], dtype=np.float32
        ),
        "score_returns_all": np.asarray([
            sample.score_returns_all if sample.score_returns_all is not None else [np.nan]*4
            for sample in samples], dtype=np.float32),
        "chicken_attribution_points": np.asarray([
            sample.chicken_attribution_points if sample.chicken_attribution_points is not None else np.nan
            for sample in samples
        ], dtype=np.float32),
    })
    return arrays


def _read_shard_index(
    path: Path,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as shard:
        _validate_shard_header(path, shard)
        count = int(shard["sample_count"])
        episode = np.asarray(shard["episode_seed"], dtype=np.int64).copy()
        player = np.asarray(shard["player_id"], dtype=np.uint8).copy()
        decision = np.asarray(shard["decision_index"], dtype=np.uint32).copy()
    if any(array.shape != (count,) for array in (episode, player, decision)):
        raise ValueError(f"{path} has invalid V4 PPO index arrays")
    return count, episode, player, decision


def _stored_fields() -> tuple[str, ...]:
    return tuple(
        [_ACTOR_PREFIX + name for name in _ACTOR_FIELDS]
        + [_PRIVILEGED_PREFIX + name for name in _PRIVILEGED_FIELDS]
        + list(_SAMPLE_FIELDS)
    )


def _validate_shard_header(path: Path, shard) -> None:
    if "chicken_attribution_points" in shard:
        if "chicken_attribution_version" not in shard or int(shard["chicken_attribution_version"]) != 1:
            raise ValueError(f"{path} has incompatible chicken attribution version")
    for name, expected in _VERSION_FIELDS.items():
        if name not in shard or int(shard[name]) != expected:
            raise ValueError(f"{path} has an incompatible {name}")
    if "sample_count" not in shard:
        raise ValueError(f"{path} has no sample count")
    optional = {
        _ACTOR_PREFIX + name
        for name in _OPTIONAL_ACTOR_FIELDS
    } | {"score_returns_all", "chicken_attribution_points"}
    missing = (set(_stored_fields()) - optional) - set(shard.files)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")


def _validate_loaded_arrays(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    count: int,
) -> None:
    points = arrays["chicken_attribution_points"]
    if points.shape != (count,) or np.isinf(points).any():
        raise ValueError(f"{path} has invalid chicken attribution values")
    for name, array in arrays.items():
        if array.shape[0] != count:
            raise ValueError(
                f"{path}:{name} has {array.shape[0]} rows, expected {count}"
            )


def _storage_array(
    name: str,
    array: np.ndarray,
    *,
    privileged: bool,
) -> np.ndarray:
    bool_fields = (
        _PRIVILEGED_BOOL_FIELDS if privileged else _ACTOR_BOOL_FIELDS
    )
    integer_fields = (
        _PRIVILEGED_INTEGER_FIELDS if privileged else _ACTOR_INTEGER_FIELDS
    )
    if name in bool_fields:
        dtype = np.bool_
    elif name in integer_fields:
        dtype = np.int16
    else:
        dtype = np.float16
    return np.ascontiguousarray(array, dtype=dtype)


def _runtime_dtype(
    name: str,
    tensor: torch.Tensor,
    *,
    privileged: bool,
) -> torch.Tensor:
    bool_fields = (
        _PRIVILEGED_BOOL_FIELDS if privileged else _ACTOR_BOOL_FIELDS
    )
    integer_fields = (
        _PRIVILEGED_INTEGER_FIELDS if privileged else _ACTOR_INTEGER_FIELDS
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
    "V4PPOBatch",
    "V4PPODataset",
    "V4PPOWriter",
    "V4_PPO_DATASET_FORMAT_VERSION",
    "V4_PPO_DATASET_PREFIX",
    "collate_v4_ppo_samples",
    "make_v4_ppo_data_loader",
]
