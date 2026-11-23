from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from bisect import bisect_right
import math
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
    ACTION_SPACE_SIZE,
    DECISION_FEATURE_VERSION,
    ActionType,
)
from .kan_adapter import (
    KAN_ADAPTER_FEATURE_DIM,
    KAN_ADAPTER_FEATURE_VERSION,
)
from .kan_teacher import KanConfidence, KanDecision
from .v4_features import (
    V4_TENSOR_VERSION,
    V4TensorBatch,
    tensorize_v4_decisions,
)


KAN_ADAPTER_DATASET_FORMAT_VERSION = 1
KAN_ADAPTER_DATASET_PREFIX = "kan_adapter"
KAN_CONFIDENCE_NAMES = ("high", "exploratory", "low")

_CONFIDENCE_TO_ID = {
    KanConfidence.HIGH: 0,
    KanConfidence.EXPLORATORY: 1,
    KanConfidence.LOW: 2,
}
_TENSOR_FIELDS = tuple(field.name for field in fields(V4TensorBatch))
_SAMPLE_FIELDS = (
    "episode_seed",
    "player_id",
    "decision_index",
    "fallback_action_id",
    "kan_action_id",
    "selected_action_id",
    "target_probability",
    "behavior_kan_probability",
    "confidence",
    "terminal_score",
)
_VERSION_FIELDS = {
    "kan_adapter_dataset_format_version": KAN_ADAPTER_DATASET_FORMAT_VERSION,
    "kan_adapter_feature_version": KAN_ADAPTER_FEATURE_VERSION,
    "v4_tensor_version": V4_TENSOR_VERSION,
    "decision_feature_version": DECISION_FEATURE_VERSION,
    "action_encoding_version": ACTION_ENCODING_VERSION,
}


@dataclass(frozen=True)
class KanAdapterSample:
    episode_seed: int
    player_id: int
    decision_index: int
    features: object
    fallback_action_id: int
    kan_action_id: int
    selected_action_id: int
    target_probability: float
    behavior_kan_probability: float
    confidence: int
    terminal_score: float

    def __post_init__(self) -> None:
        if not -(1 << 63) <= self.episode_seed < (1 << 63):
            raise ValueError("episode seed must fit in int64")
        if not 0 <= self.player_id < 4:
            raise ValueError("player id must be in range 0..3")
        if not 0 <= self.decision_index < (1 << 32):
            raise ValueError("decision index must fit in uint32")
        for name, action_id in (
            ("fallback", self.fallback_action_id),
            ("kan", self.kan_action_id),
            ("selected", self.selected_action_id),
        ):
            if not 0 <= action_id < ACTION_SPACE_SIZE:
                raise ValueError(f"{name} action id is outside the action space")
        if self.selected_action_id not in (
            self.fallback_action_id,
            self.kan_action_id,
        ):
            raise ValueError("selected action must be the fallback or candidate kan")
        _validate_probability("target probability", self.target_probability)
        _validate_probability(
            "behavior kan probability",
            self.behavior_kan_probability,
        )
        if not 0 <= self.confidence < len(KAN_CONFIDENCE_NAMES):
            raise ValueError("unknown kan confidence id")
        if not math.isfinite(self.terminal_score):
            raise ValueError("terminal score must be finite")

        if len(self.features.actions) != ACTION_SPACE_SIZE:
            raise ValueError("decision must contain the complete action space")
        fallback = self.features.actions[self.fallback_action_id]
        candidate = self.features.actions[self.kan_action_id]
        if not fallback.legal or not fallback.supported_by_v4:
            raise ValueError("fallback action must be legal and supported by V4")
        if fallback.semantics.type in (
            ActionType.ANKAN,
            ActionType.KAKAN,
            ActionType.DAIMINKAN,
        ):
            raise ValueError("fallback action cannot be a kan")
        if not candidate.legal:
            raise ValueError("candidate kan must be legal")
        if candidate.semantics.type not in (
            ActionType.ANKAN,
            ActionType.KAKAN,
            ActionType.DAIMINKAN,
        ):
            raise ValueError("candidate action is not a kan")

    @classmethod
    def from_decision(
        cls,
        *,
        episode_seed: int,
        player_id: int,
        decision_index: int,
        features,
        decision: KanDecision,
        target_probability: float,
        terminal_score: float,
    ) -> "KanAdapterSample":
        if not decision.evaluations:
            raise ValueError("kan decision has no evaluated candidate")
        best = max(
            decision.evaluations,
            key=lambda item: (
                item.expected_shanten_delta,
                -item.worse_probability,
                item.improve_probability,
                -item.action_id,
            ),
        )
        return cls(
            episode_seed=episode_seed,
            player_id=player_id,
            decision_index=decision_index,
            features=features,
            fallback_action_id=decision.fallback_action_id,
            kan_action_id=best.action_id,
            selected_action_id=decision.selected_action_id,
            target_probability=target_probability,
            behavior_kan_probability=decision.kan_selection_probability,
            confidence=_CONFIDENCE_TO_ID[best.confidence],
            terminal_score=terminal_score,
        )


@dataclass(frozen=True)
class KanAdapterBatch:
    features: V4TensorBatch
    episode_seed: torch.Tensor
    player_id: torch.Tensor
    decision_index: torch.Tensor
    fallback_action_id: torch.Tensor
    kan_action_id: torch.Tensor
    selected_action_id: torch.Tensor
    target_probability: torch.Tensor
    behavior_kan_probability: torch.Tensor
    confidence: torch.Tensor
    terminal_score: torch.Tensor

    def __len__(self) -> int:
        return self.features.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "KanAdapterBatch":
        return replace(
            self,
            features=self.features.to(device),
            **{
                name: getattr(self, name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for name in _SAMPLE_FIELDS
            },
        )

    def pin_memory(self) -> "KanAdapterBatch":
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


class KanAdapterWriter:
    def __init__(
        self,
        directory: Path | str,
        samples_per_shard: int = 256,
        prefix: str = KAN_ADAPTER_DATASET_PREFIX,
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
        self._samples: list[KanAdapterSample] = []
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_shard_index = self._find_next_shard_index()

    def append(self, sample: KanAdapterSample) -> Path | None:
        self._ensure_open()
        if not isinstance(sample, KanAdapterSample):
            raise TypeError("writer accepts KanAdapterSample objects")
        self._samples.append(sample)
        if len(self._samples) >= self.samples_per_shard:
            return self._flush_buffer()
        return None

    def append_many(
        self,
        samples: Iterable[KanAdapterSample],
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

    def __enter__(self) -> "KanAdapterWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _flush_buffer(self) -> Path | None:
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
            raise RuntimeError("writer is closed")


@dataclass(frozen=True)
class _ShardInfo:
    path: Path
    start: int
    count: int


class KanAdapterDataset(Dataset):
    def __init__(
        self,
        directory: Path | str,
        *,
        prefix: str = KAN_ADAPTER_DATASET_PREFIX,
        cache_size: int = 2,
    ):
        if cache_size <= 0:
            raise ValueError("cache size must be positive")
        self.directory = Path(directory)
        paths = sorted(self.directory.glob(f"{prefix}_*.npz"))
        if not paths:
            raise ValueError("kan adapter dataset contains no shards")

        self._shards: list[_ShardInfo] = []
        self._starts = []
        episodes = []
        players = []
        decisions = []
        confidences = []
        start = 0
        for path in paths:
            count, index_arrays = _read_shard_index(path)
            self._starts.append(start)
            self._shards.append(_ShardInfo(path, start, count))
            start += count
            episodes.append(index_arrays["episode_seed"])
            players.append(index_arrays["player_id"])
            decisions.append(index_arrays["decision_index"])
            confidences.append(index_arrays["confidence"])
        self._length = start
        self._episode_seeds = np.concatenate(episodes)
        self._player_ids = np.concatenate(players)
        self._decision_indices = np.concatenate(decisions)
        self._confidences = np.concatenate(confidences)
        self._cache_limit = cache_size
        self._cache: OrderedDict[int, Mapping[str, np.ndarray]] = OrderedDict()

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
    def confidences(self) -> np.ndarray:
        return self._confidences

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
            "features": {
                name: torch.as_tensor(arrays[name][local])
                for name in _TENSOR_FIELDS
            },
            **{
                name: torch.as_tensor(arrays[name][local])
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
                for name in _TENSOR_FIELDS + _SAMPLE_FIELDS
            }
        _validate_loaded_arrays(info.path, arrays, info.count)
        self._cache[shard_index] = arrays
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return arrays


class KanAdapterShardBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: KanAdapterDataset,
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
            raise ValueError("batch sampler indices are outside the dataset")
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
            values = np.asarray(self._groups[shard_index], dtype=np.int64)
            if self.shuffle:
                values = rng.permutation(values)
            for start in range(0, len(values), self.batch_size):
                yield values[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(
            (len(values) + self.batch_size - 1) // self.batch_size
            for values in self._groups.values()
        )


def make_kan_adapter_data_loader(
    dataset: KanAdapterDataset,
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
        batch_sampler=KanAdapterShardBatchSampler(
            dataset,
            indices,
            batch_size,
            shuffle=shuffle,
            seed=seed,
        ),
        collate_fn=collate_kan_adapter_samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def collate_kan_adapter_samples(
    samples: Sequence[Mapping[str, object]],
) -> KanAdapterBatch:
    if not samples:
        raise ValueError("cannot collate an empty kan adapter batch")
    features = V4TensorBatch(**{
        name: _to_runtime_tensor(name, torch.stack([
            sample["features"][name] for sample in samples
        ]))
        for name in _TENSOR_FIELDS
    })
    features.validate()
    batch = KanAdapterBatch(
        features=features,
        episode_seed=_stack(samples, "episode_seed", torch.long),
        player_id=_stack(samples, "player_id", torch.long),
        decision_index=_stack(samples, "decision_index", torch.long),
        fallback_action_id=_stack(samples, "fallback_action_id", torch.long),
        kan_action_id=_stack(samples, "kan_action_id", torch.long),
        selected_action_id=_stack(samples, "selected_action_id", torch.long),
        target_probability=_stack(
            samples,
            "target_probability",
            torch.float32,
        ),
        behavior_kan_probability=_stack(
            samples,
            "behavior_kan_probability",
            torch.float32,
        ),
        confidence=_stack(samples, "confidence", torch.long),
        terminal_score=_stack(samples, "terminal_score", torch.float32),
    )
    _validate_batch(batch)
    return batch


def _samples_to_arrays(
    samples: Sequence[KanAdapterSample],
) -> dict[str, np.ndarray]:
    tensor_batch = tensorize_v4_decisions([
        sample.features for sample in samples
    ])
    if tensor_batch.kan_adapter_features is None:
        raise ValueError("tensorized samples have no kan adapter features")
    arrays = {
        name: np.asarray(value)
        for name, value in _VERSION_FIELDS.items()
    }
    arrays["sample_count"] = np.asarray(len(samples), dtype=np.uint64)
    for field in fields(tensor_batch):
        tensor = getattr(tensor_batch, field.name)
        if tensor is None:
            raise ValueError(f"tensor field {field.name} cannot be absent")
        arrays[field.name] = _to_storage_array(tensor.detach().cpu().numpy())
    arrays.update({
        "episode_seed": np.asarray(
            [sample.episode_seed for sample in samples],
            dtype=np.int64,
        ),
        "player_id": np.asarray(
            [sample.player_id for sample in samples],
            dtype=np.uint8,
        ),
        "decision_index": np.asarray(
            [sample.decision_index for sample in samples],
            dtype=np.uint32,
        ),
        "fallback_action_id": np.asarray(
            [sample.fallback_action_id for sample in samples],
            dtype=np.uint8,
        ),
        "kan_action_id": np.asarray(
            [sample.kan_action_id for sample in samples],
            dtype=np.uint8,
        ),
        "selected_action_id": np.asarray(
            [sample.selected_action_id for sample in samples],
            dtype=np.uint8,
        ),
        "target_probability": np.asarray(
            [sample.target_probability for sample in samples],
            dtype=np.float32,
        ),
        "behavior_kan_probability": np.asarray(
            [sample.behavior_kan_probability for sample in samples],
            dtype=np.float32,
        ),
        "confidence": np.asarray(
            [sample.confidence for sample in samples],
            dtype=np.uint8,
        ),
        "terminal_score": np.asarray(
            [sample.terminal_score for sample in samples],
            dtype=np.float32,
        ),
    })
    return arrays


def _read_shard_index(
    path: Path,
) -> tuple[int, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as shard:
        _validate_shard_header(path, shard)
        count = int(shard["sample_count"])
        arrays = {
            name: np.asarray(shard[name]).copy()
            for name in (
                "episode_seed",
                "player_id",
                "decision_index",
                "confidence",
            )
        }
    if any(array.shape != (count,) for array in arrays.values()):
        raise ValueError(f"{path} has invalid index arrays")
    return count, arrays


def _validate_shard_header(path: Path, shard) -> None:
    for name, expected in _VERSION_FIELDS.items():
        if name not in shard or int(shard[name]) != expected:
            raise ValueError(f"{path} has an incompatible {name}")
    if "sample_count" not in shard:
        raise ValueError(f"{path} has no sample count")
    missing = set(_TENSOR_FIELDS + _SAMPLE_FIELDS) - set(shard.files)
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
    if arrays["kan_adapter_features"].shape != (
        count,
        ACTION_SPACE_SIZE,
        KAN_ADAPTER_FEATURE_DIM,
    ):
        raise ValueError(f"{path} has invalid kan adapter features")
    for name in ("target_probability", "behavior_kan_probability"):
        values = arrays[name]
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{path} has invalid {name}")
    if np.any(~np.isfinite(arrays["terminal_score"])):
        raise ValueError(f"{path} has invalid terminal scores")


def _validate_batch(batch: KanAdapterBatch) -> None:
    row = torch.arange(len(batch), device=batch.kan_action_id.device)
    if torch.any(
        batch.features.kan_adapter_features[
            row,
            batch.kan_action_id,
            0,
        ] <= 0.5
    ):
        raise ValueError("candidate kan has no known adapter features")
    if torch.any(~batch.features.selectable_actions[
        row,
        batch.fallback_action_id,
    ]):
        raise ValueError("fallback action is not selectable")
    if torch.any(~batch.features.action_legal[row, batch.kan_action_id]):
        raise ValueError("candidate kan is not legal")
    for name in ("target_probability", "behavior_kan_probability"):
        values = getattr(batch, name)
        if torch.any(~torch.isfinite(values)) or torch.any(
            (values < 0.0) | (values > 1.0)
        ):
            raise ValueError(f"invalid {name}")


def _stack(
    samples: Sequence[Mapping[str, object]],
    name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack([sample[name] for sample in samples]).to(dtype=dtype)


def _to_storage_array(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.bool_):
        dtype = np.bool_
    elif np.issubdtype(array.dtype, np.integer):
        dtype = np.int16
    else:
        dtype = np.float16
    return np.ascontiguousarray(array, dtype=dtype)


def _to_runtime_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        return tensor
    if tensor.is_floating_point():
        return tensor.to(dtype=torch.float32)
    return tensor.to(dtype=torch.long)


def _save_npz_atomic(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    compressed: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.stem + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        save = np.savez_compressed if compressed else np.savez
        save(handle, **arrays)
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in range 0..1")


__all__ = [
    "KAN_ADAPTER_DATASET_FORMAT_VERSION",
    "KAN_ADAPTER_DATASET_PREFIX",
    "KAN_CONFIDENCE_NAMES",
    "KanAdapterBatch",
    "KanAdapterDataset",
    "KanAdapterSample",
    "KanAdapterShardBatchSampler",
    "KanAdapterWriter",
    "collate_kan_adapter_samples",
    "make_kan_adapter_data_loader",
]
