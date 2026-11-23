from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from bisect import bisect_right
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    DECISION_FEATURE_VERSION,
)
from .demonstration import (
    TeacherDecision,
    action_ids_from_mask,
)
from .teacher import (
    DECLARE_READY_ACTION_ID,
    PASS_ACTION_ID,
    PON_ACTION_ID,
    RON_ACTION_ID,
    TSUMO_ACTION_ID,
)
from .kan_adapter import KAN_ADAPTER_FEATURE_DIM
from .v4_features import (
    V4_TENSOR_VERSION,
    V4TensorBatch,
    tensorize_v4_decisions,
)


V4_DEMONSTRATION_FORMAT_VERSION = 1
V4_DEMONSTRATION_PREFIX = "v4_demonstration"

V4_DISCARD_CATEGORY = 0
V4_PON_PASS_CATEGORY = 1
V4_HU_CATEGORY = 2
V4_READY_CATEGORY = 3
V4_CATEGORY_NAMES = ("discard", "pon_pass", "hu", "ready")

_BOOL_FIELDS = {
    "action_legal",
    "action_supported",
    "discard_valid",
    "meld_valid",
    "pon_follow_up_mask",
}
_INTEGER_FIELDS = {
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
_TENSOR_FIELDS = tuple(field.name for field in fields(V4TensorBatch))
_OPTIONAL_TENSOR_FIELDS = ("kan_adapter_features",)
_SAMPLE_FIELDS = (
    "episode_seed",
    "player_id",
    "decision_index",
    "selected_action_id",
    "category",
    "supervision_mask",
    "target_mask",
)
_VERSION_FIELDS = {
    "v4_demonstration_format_version": V4_DEMONSTRATION_FORMAT_VERSION,
    "v4_tensor_version": V4_TENSOR_VERSION,
    "decision_feature_version": DECISION_FEATURE_VERSION,
    "action_encoding_version": ACTION_ENCODING_VERSION,
}


@dataclass(frozen=True)
class V4DemonstrationSample:
    episode_seed: int
    player_id: int
    decision_index: int
    features: object
    selected_action_id: int
    supervision_action_ids: tuple[int, ...]
    target_action_ids: tuple[int, ...]
    category: int

    def __post_init__(self) -> None:
        if not -(1 << 63) <= self.episode_seed < (1 << 63):
            raise ValueError("episode seed must fit in int64")
        if not 0 <= self.player_id < 4:
            raise ValueError("player id must be in range 0..3")
        if not 0 <= self.decision_index < (1 << 32):
            raise ValueError("decision index must fit in uint32")
        _validate_action_id(self.selected_action_id)
        supervision = _validated_action_set(
            "supervision actions",
            self.supervision_action_ids,
        )
        targets = _validated_action_set(
            "target actions",
            self.target_action_ids,
        )
        if not targets:
            raise ValueError("demonstration sample must contain a target")
        if not targets.issubset(supervision):
            raise ValueError("target actions must be supervised")

        legal = {
            action_id
            for action_id, action in enumerate(self.features.actions)
            if action.legal
        }
        supported = {
            action_id
            for action_id, action in enumerate(self.features.actions)
            if action.legal and action.supported_by_v4
        }
        if self.selected_action_id not in legal:
            raise ValueError("selected action must be legal")
        if not supervision.issubset(supported):
            raise ValueError("supervised actions must be supported by V4")
        observation_legal = set(action_ids_from_mask(tuple(
            int(word) for word in self.features.observation.legal_mask
        )))
        if observation_legal != legal:
            raise ValueError("decision and observation legal actions disagree")
        expected_category = classify_v4_demonstration(
            supervision,
            targets,
        )
        if self.category != expected_category:
            raise ValueError("sample category does not match its masks")

    @classmethod
    def from_teacher_decision(
        cls,
        *,
        episode_seed: int,
        player_id: int,
        decision_index: int,
        features,
        decision: TeacherDecision,
    ) -> "V4DemonstrationSample":
        if not decision.trainable:
            raise ValueError("skipped teacher decisions are not samples")
        supported = {
            action_id
            for action_id, action in enumerate(features.actions)
            if action.legal and action.supported_by_v4
        }
        supervision = tuple(
            action_id
            for action_id in decision.supervised_action_ids
            if action_id in supported
        )
        targets = tuple(
            action_id
            for action_id in decision.target_action_ids
            if action_id in supported
        )
        if not targets:
            raise ValueError("teacher targets contain no V4-supported action")
        return cls(
            episode_seed=episode_seed,
            player_id=player_id,
            decision_index=decision_index,
            features=features,
            selected_action_id=decision.selected_action_id,
            supervision_action_ids=supervision,
            target_action_ids=targets,
            category=classify_v4_demonstration(supervision, targets),
        )


@dataclass(frozen=True)
class V4DemonstrationBatch:
    features: V4TensorBatch
    episode_seed: torch.Tensor
    player_id: torch.Tensor
    decision_index: torch.Tensor
    selected_action_id: torch.Tensor
    category: torch.Tensor
    supervision_mask: torch.Tensor
    target_mask: torch.Tensor

    def __len__(self) -> int:
        return self.features.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "V4DemonstrationBatch":
        return replace(
            self,
            features=self.features.to(device),
            **{
                name: getattr(self, name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for name in _SAMPLE_FIELDS
                if name not in {"supervision_mask", "target_mask"}
            },
            supervision_mask=self.supervision_mask.to(
                device,
                non_blocking=non_blocking,
            ),
            target_mask=self.target_mask.to(
                device,
                non_blocking=non_blocking,
            ),
        )

    def pin_memory(self) -> "V4DemonstrationBatch":
        return replace(
            self,
            features=replace(
                self.features,
                **{
                    field.name: getattr(self.features, field.name).pin_memory()
                    for field in fields(self.features)
                },
            ),
            **{
                name: getattr(self, name).pin_memory()
                for name in _SAMPLE_FIELDS
            },
        )


class V4DemonstrationWriter:
    def __init__(
        self,
        directory: Path | str,
        samples_per_shard: int = 256,
        prefix: str = V4_DEMONSTRATION_PREFIX,
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
        self._samples: list[V4DemonstrationSample] = []
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_shard_index = self._find_next_shard_index()

    @property
    def buffered_sample_count(self) -> int:
        return len(self._samples)

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, sample: V4DemonstrationSample) -> Optional[Path]:
        self._ensure_open()
        if not isinstance(sample, V4DemonstrationSample):
            raise TypeError("writer accepts V4DemonstrationSample objects")
        self._samples.append(sample)
        if len(self._samples) >= self.samples_per_shard:
            return self._flush_buffer()
        return None

    def append_many(
        self,
        samples: Iterable[V4DemonstrationSample],
    ) -> tuple[Path, ...]:
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

    def __enter__(self) -> "V4DemonstrationWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _flush_buffer(self) -> Optional[Path]:
        if not self._samples:
            return None
        arrays = _samples_to_arrays(self._samples)
        destination = self.directory / (
            f"{self.prefix}_{self._next_shard_index:06d}.npz"
        )
        _save_npz_atomic(destination, arrays, self.compressed)
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
            raise RuntimeError("demonstration writer is closed")


@dataclass(frozen=True)
class _ShardInfo:
    path: Path
    start: int
    count: int


class V4DemonstrationDataset(Dataset):
    """Lazy sharded dataset; only a small shard cache resides in RAM."""

    def __init__(
        self,
        directory: Path | str,
        *,
        prefix: str = V4_DEMONSTRATION_PREFIX,
        cache_shards: int = 1,
    ):
        if cache_shards <= 0:
            raise ValueError("cache_shards must be positive")
        self.directory = Path(directory)
        paths = sorted(self.directory.glob(f"{prefix}_*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"no V4 demonstration shards found in {self.directory}"
            )
        self._cache_limit = cache_shards
        self._cache: OrderedDict[int, Mapping[str, np.ndarray]] = OrderedDict()
        self._shards: list[_ShardInfo] = []
        episode_parts = []
        player_parts = []
        decision_parts = []
        category_parts = []
        target_parts = []
        start = 0
        for path in paths:
            (
                count,
                episodes,
                players,
                decisions,
                categories,
                targets,
            ) = _read_shard_index(path)
            self._shards.append(_ShardInfo(path, start, count))
            start += count
            episode_parts.append(episodes)
            player_parts.append(players)
            decision_parts.append(decisions)
            category_parts.append(categories)
            target_parts.append(targets)
        self._length = start
        self._starts = [info.start for info in self._shards]
        self._episode_seeds = np.concatenate(episode_parts)
        self._player_ids = np.concatenate(player_parts)
        self._decision_indices = np.concatenate(decision_parts)
        self._categories = np.concatenate(category_parts)
        self._target_masks = np.concatenate(target_parts)

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
    def categories(self) -> np.ndarray:
        return self._categories

    @property
    def target_masks(self) -> np.ndarray:
        return self._target_masks

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def shard_index_for_sample(self, index: int) -> int:
        index = self._normalize_index(index)
        return bisect_right(self._starts, index) - 1

    def __getitem__(self, index: int) -> dict[str, object]:
        index = self._normalize_index(index)
        shard_index = self.shard_index_for_sample(index)
        shard_info = self._shards[shard_index]
        local_index = index - shard_info.start
        arrays = self._load_shard(shard_index)
        return {
            "features": {
                name: torch.as_tensor(arrays[name][local_index])
                for name in _TENSOR_FIELDS
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
                name: (
                    np.ascontiguousarray(shard[name])
                    if name in shard
                    else np.zeros(
                        (
                            info.count,
                            ACTION_SPACE_SIZE,
                            KAN_ADAPTER_FEATURE_DIM,
                        ),
                        dtype=np.float16,
                    )
                )
                for name in _TENSOR_FIELDS + _SAMPLE_FIELDS
            }
        _validate_loaded_arrays(info.path, arrays, info.count)
        self._cache[shard_index] = arrays
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return arrays


class V4ShardBatchSampler(Sampler[list[int]]):
    """Shuffle within and between shards without random shard thrashing."""

    def __init__(
        self,
        dataset: V4DemonstrationDataset,
        indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if np.any((self.indices < 0) | (self.indices >= len(dataset))):
            raise ValueError("batch sampler indices are outside the dataset")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
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
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch.tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self._groups.values()
            )
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._groups.values()
        )


def make_v4_data_loader(
    dataset: V4DemonstrationDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    sampler = V4ShardBatchSampler(
        dataset,
        indices,
        batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_v4_demonstrations,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def collate_v4_demonstrations(
    samples: Sequence[Mapping[str, object]],
) -> V4DemonstrationBatch:
    if not samples:
        raise ValueError("cannot collate an empty V4 batch")
    feature_values = {
        name: torch.stack([
            sample["features"][name] for sample in samples
        ])
        for name in _TENSOR_FIELDS
    }
    features = V4TensorBatch(**{
        name: _to_runtime_dtype(name, tensor)
        for name, tensor in feature_values.items()
    })
    features.validate()
    batch = V4DemonstrationBatch(
        features=features,
        episode_seed=torch.stack([
            sample["episode_seed"] for sample in samples
        ]).long(),
        player_id=torch.stack([
            sample["player_id"] for sample in samples
        ]).long(),
        decision_index=torch.stack([
            sample["decision_index"] for sample in samples
        ]).long(),
        selected_action_id=torch.stack([
            sample["selected_action_id"] for sample in samples
        ]).long(),
        category=torch.stack([
            sample["category"] for sample in samples
        ]).long(),
        supervision_mask=torch.stack([
            sample["supervision_mask"] for sample in samples
        ]).bool(),
        target_mask=torch.stack([
            sample["target_mask"] for sample in samples
        ]).bool(),
    )
    _validate_training_batch(batch)
    return batch


def classify_v4_demonstration(
    supervision_action_ids: Iterable[int],
    target_action_ids: Iterable[int],
) -> int:
    supervision = _validated_action_set(
        "supervision actions",
        supervision_action_ids,
    )
    targets = _validated_action_set("target actions", target_action_ids)
    if targets & {TSUMO_ACTION_ID, RON_ACTION_ID}:
        return V4_HU_CATEGORY
    if supervision == {PON_ACTION_ID, PASS_ACTION_ID}:
        return V4_PON_PASS_CATEGORY
    if supervision == {PASS_ACTION_ID, DECLARE_READY_ACTION_ID}:
        return V4_READY_CATEGORY
    if supervision and all(action_id < 27 for action_id in supervision):
        return V4_DISCARD_CATEGORY
    raise ValueError("demonstration masks do not match a V4 category")


def _samples_to_arrays(
    samples: Sequence[V4DemonstrationSample],
) -> dict[str, np.ndarray]:
    tensor_batch = tensorize_v4_decisions([
        sample.features for sample in samples
    ])
    arrays = {
        name: np.asarray(value)
        for name, value in _VERSION_FIELDS.items()
    }
    arrays["sample_count"] = np.asarray(len(samples), dtype=np.uint64)
    for field in fields(tensor_batch):
        tensor = getattr(tensor_batch, field.name).detach().cpu()
        arrays[field.name] = _to_storage_array(field.name, tensor.numpy())

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
    arrays["selected_action_id"] = np.asarray(
        [sample.selected_action_id for sample in samples],
        dtype=np.uint8,
    )
    arrays["category"] = np.asarray(
        [sample.category for sample in samples],
        dtype=np.uint8,
    )
    arrays["supervision_mask"] = np.zeros(
        (len(samples), ACTION_SPACE_SIZE),
        dtype=np.bool_,
    )
    arrays["target_mask"] = np.zeros_like(arrays["supervision_mask"])
    for row, sample in enumerate(samples):
        arrays["supervision_mask"][row, list(
            sample.supervision_action_ids
        )] = True
        arrays["target_mask"][row, list(sample.target_action_ids)] = True
    return arrays


def _read_shard_index(
    path: Path,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    with np.load(path, allow_pickle=False) as shard:
        _validate_shard_header(path, shard)
        count = int(shard["sample_count"])
        episodes = np.asarray(shard["episode_seed"], dtype=np.int64).copy()
        players = np.asarray(shard["player_id"], dtype=np.uint8).copy()
        decisions = np.asarray(shard["decision_index"], dtype=np.uint32).copy()
        categories = np.asarray(shard["category"], dtype=np.uint8).copy()
        targets = np.asarray(shard["target_mask"], dtype=np.bool_).copy()
    if (
        episodes.shape != (count,)
        or players.shape != (count,)
        or decisions.shape != (count,)
        or categories.shape != (count,)
        or targets.shape != (count, ACTION_SPACE_SIZE)
    ):
        raise ValueError(f"{path} has invalid index arrays")
    return count, episodes, players, decisions, categories, targets


def _validate_shard_header(path: Path, shard) -> None:
    for name, expected in _VERSION_FIELDS.items():
        if name not in shard or int(shard[name]) != expected:
            raise ValueError(f"{path} has an incompatible {name}")
    if "sample_count" not in shard:
        raise ValueError(f"{path} has no sample count")
    required = set(_TENSOR_FIELDS + _SAMPLE_FIELDS) - set(
        _OPTIONAL_TENSOR_FIELDS
    )
    missing = required - set(shard.files)
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
    if arrays["supervision_mask"].shape != (count, ACTION_SPACE_SIZE):
        raise ValueError(f"{path} has an invalid supervision mask")
    if arrays["target_mask"].shape != (count, ACTION_SPACE_SIZE):
        raise ValueError(f"{path} has an invalid target mask")
    if np.any(arrays["target_mask"] & ~arrays["supervision_mask"]):
        raise ValueError(f"{path} contains unsupervised targets")
    if np.any(~arrays["target_mask"].any(axis=1)):
        raise ValueError(f"{path} contains targetless samples")


def _validate_training_batch(batch: V4DemonstrationBatch) -> None:
    expected = (len(batch), ACTION_SPACE_SIZE)
    if batch.supervision_mask.shape != expected:
        raise ValueError("supervision mask has the wrong shape")
    if batch.target_mask.shape != expected:
        raise ValueError("target mask has the wrong shape")
    if batch.supervision_mask.dtype != torch.bool:
        raise TypeError("supervision mask must be bool")
    if batch.target_mask.dtype != torch.bool:
        raise TypeError("target mask must be bool")
    if torch.any(batch.target_mask & ~batch.supervision_mask):
        raise ValueError("target actions must be supervised")
    if torch.any(batch.supervision_mask & ~batch.features.selectable_actions):
        raise ValueError("supervised actions must be selectable by V4")


def _to_storage_array(name: str, array: np.ndarray) -> np.ndarray:
    if name in _BOOL_FIELDS:
        dtype = np.bool_
    elif name in _INTEGER_FIELDS:
        dtype = np.int16
    else:
        dtype = np.float16
    return np.ascontiguousarray(array, dtype=dtype)


def _to_runtime_dtype(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name in _BOOL_FIELDS:
        return tensor.to(dtype=torch.bool)
    if name in _INTEGER_FIELDS:
        return tensor.to(dtype=torch.long)
    return tensor.to(dtype=torch.float32)


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


def _validated_action_set(
    name: str,
    action_ids: Iterable[int],
) -> set[int]:
    values = tuple(action_ids)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contain duplicates")
    for action_id in values:
        _validate_action_id(action_id)
    return set(values)


def _validate_action_id(action_id: int) -> None:
    if not isinstance(action_id, int):
        raise TypeError("action id must be an integer")
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError("action id is outside the action space")


__all__ = [
    "V4_CATEGORY_NAMES",
    "V4_DEMONSTRATION_FORMAT_VERSION",
    "V4_DEMONSTRATION_PREFIX",
    "V4_DISCARD_CATEGORY",
    "V4_HU_CATEGORY",
    "V4_PON_PASS_CATEGORY",
    "V4_READY_CATEGORY",
    "V4DemonstrationBatch",
    "V4DemonstrationDataset",
    "V4DemonstrationSample",
    "V4DemonstrationWriter",
    "V4ShardBatchSampler",
    "classify_v4_demonstration",
    "collate_v4_demonstrations",
    "make_v4_data_loader",
]
