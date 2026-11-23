from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_FEATURE_COUNT,
    ACTION_FEATURE_VERSION,
    ACTION_SPACE_SIZE,
    MAX_DISCARD_EVENTS,
    MAX_MELDS,
    OBSERVATION_VERSION,
    TILE_TYPE_COUNT,
)
from ._observation_storage import (
    allocate_observation_arrays,
    write_observation,
)
from .action_features import tensorize_action_features
from .demonstration_dataset import DEMONSTRATION_FORMAT_VERSION


DISCARD_CATEGORY = 0
PON_PASS_CATEGORY = 1
HU_CATEGORY = 2
CATEGORY_NAMES = ("discard", "pon_pass", "hu")

PON_ACTION_ID = TILE_TYPE_COUNT * 3
TSUMO_ACTION_ID = PON_ACTION_ID + 2
RON_ACTION_ID = PON_ACTION_ID + 3
PASS_ACTION_ID = PON_ACTION_ID + 4

TILE_PAD = TILE_TYPE_COUNT
PLAYER_PAD = 4
ROUND_PHASE_COUNT = 7
READY_KIND_COUNT = 3
MELD_TYPE_COUNT = 4
POLICY_ARCHITECTURE_VERSION = 4

_ARRAY_SHAPES = {
    "hand_counts": (TILE_TYPE_COUNT,),
    "discard_tiles": (MAX_DISCARD_EVENTS,),
    "discard_actors": (MAX_DISCARD_EVENTS,),
    "discard_was_called": (MAX_DISCARD_EVENTS,),
    "discard_count": (),
    "meld_tiles": (MAX_MELDS,),
    "meld_owners": (MAX_MELDS,),
    "meld_sources": (MAX_MELDS,),
    "meld_types": (MAX_MELDS,),
    "meld_valid": (MAX_MELDS,),
    "meld_count": (),
    "visible_counts": (TILE_TYPE_COUNT,),
    "waits": (),
    "shanten": (),
    "phase": (),
    "current_actor": (),
    "current_discard": (),
    "current_discard_actor": (),
    "dealer": (),
    "wall_remaining": (),
    "ready_kinds": (4,),
    "locked_ready_waits": (),
    "episode_seed": (),
    "player_id": (),
    "legal_mask_words": (2,),
    "supervision_mask_words": (2,),
    "target_action_mask_words": (2,),
    "selected_action_id": (),
}
_OBSERVATION_ARRAY_NAMES = tuple(_ARRAY_SHAPES)[:22]


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_seeds: Tuple[int, ...]
    validation_seeds: Tuple[int, ...]
    test_seeds: Tuple[int, ...]


@dataclass(frozen=True)
class PolicyConfig:
    tile_embedding_dim: int = 8
    player_embedding_dim: int = 4
    meld_type_embedding_dim: int = 4
    ready_embedding_dim: int = 4
    phase_embedding_dim: int = 4
    shanten_embedding_dim: int = 4
    discard_hidden_dim: int = 48
    tile_conv_channels: int = 0
    explicit_meld_counts: bool = False
    public_profile_enabled: bool = False
    player_discard_profile_enabled: bool = True
    player_meld_profile_enabled: bool = True
    player_discard_hidden_dim: int = 32
    player_meld_hidden_dim: int = 32
    player_token_dim: int = 64
    profile_hidden_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1
    action_feature_enabled: bool = False
    action_feature_hidden_dim: int = 64
    action_feature_intermediate_dim: int = 32


class DemonstrationTensorDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        if not tensors:
            raise ValueError("demonstration tensors cannot be empty")
        lengths = {len(tensor) for tensor in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("all demonstration tensors must have equal length")
        self.tensors = dict(tensors)
        self._length = lengths.pop()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            name: tensor[index]
            for name, tensor in self.tensors.items()
        }

    @property
    def episode_seeds(self) -> torch.Tensor:
        return self.tensors["episode_seed"]

    @property
    def categories(self) -> torch.Tensor:
        return self.tensors["category"]


class ActionFeatureHead(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        if hidden_dim <= 0 or intermediate_dim <= 0:
            raise ValueError("action feature dimensions must be positive")
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.action_projection = nn.Linear(ACTION_FEATURE_COUNT, hidden_dim)
        self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, hidden_dim)
        self.activation = nn.GELU()
        self.score_hidden = nn.Linear(hidden_dim, intermediate_dim)
        self.score_output = nn.Linear(intermediate_dim, 1)
        nn.init.zeros_(self.score_output.weight)
        nn.init.zeros_(self.score_output.bias)
        self.register_buffer(
            "supported_actions",
            torch.arange(ACTION_SPACE_SIZE) <= PON_ACTION_ID + 1,
            persistent=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        expected = (state.shape[0], ACTION_SPACE_SIZE, ACTION_FEATURE_COUNT)
        if action_features.shape != expected:
            raise ValueError(f"action features must have shape {expected}")
        action_ids = torch.arange(ACTION_SPACE_SIZE, device=state.device)
        combined = (
            self.state_projection(state).unsqueeze(1)
            + self.action_projection(action_features.to(dtype=state.dtype))
            + self.action_embedding(action_ids).unsqueeze(0)
        )
        scores = self.score_output(
            self.activation(self.score_hidden(self.activation(combined)))
        ).squeeze(-1)
        return scores * self.supported_actions.to(dtype=scores.dtype)


class BehaviorClonePolicy(nn.Module):
    def __init__(self, config: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.config = config

        self.tile_embedding = nn.Embedding(
            TILE_TYPE_COUNT + 1,
            config.tile_embedding_dim,
            padding_idx=TILE_PAD,
        )
        self.player_embedding = nn.Embedding(
            PLAYER_PAD + 1,
            config.player_embedding_dim,
            padding_idx=PLAYER_PAD,
        )
        self.meld_type_embedding = nn.Embedding(
            MELD_TYPE_COUNT,
            config.meld_type_embedding_dim,
        )
        self.ready_embedding = nn.Embedding(
            READY_KIND_COUNT,
            config.ready_embedding_dim,
        )
        self.phase_embedding = nn.Embedding(
            ROUND_PHASE_COUNT,
            config.phase_embedding_dim,
        )
        self.shanten_embedding = nn.Embedding(
            10,
            config.shanten_embedding_dim,
        )

        discard_input_dim = (
            config.tile_embedding_dim
            + config.player_embedding_dim
            + 1
        )
        self.discard_encoder = nn.GRU(
            input_size=discard_input_dim,
            hidden_size=config.discard_hidden_dim,
            batch_first=True,
        )
        self.suit_encoder = None
        if config.tile_conv_channels:
            self.suit_encoder = nn.Sequential(
                nn.Conv1d(
                    4,
                    config.tile_conv_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv1d(
                    config.tile_conv_channels,
                    config.tile_conv_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
            )

        meld_dim = (
            config.tile_embedding_dim
            + config.player_embedding_dim * 2
            + config.meld_type_embedding_dim
        )
        fixed_numeric_dim = (
            TILE_TYPE_COUNT * 4
            + TILE_TYPE_COUNT * config.tile_conv_channels
            + (
                PLAYER_PAD + MELD_TYPE_COUNT
                if config.explicit_meld_counts
                else 0
            )
            + 3
        )
        fixed_embedding_dim = (
            config.shanten_embedding_dim
            + config.phase_embedding_dim
            + config.player_embedding_dim * 3
            + config.tile_embedding_dim
            + config.ready_embedding_dim * 4
        )
        encoder_dim = (
            fixed_numeric_dim
            + fixed_embedding_dim
            + config.discard_hidden_dim
            + meld_dim
        )
        self.encoder_dim = encoder_dim

        self.policy_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, ACTION_SPACE_SIZE),
        )

        self.player_discard_encoder = None
        self.player_meld_encoder = None
        self.player_token_encoder = None
        self.profile_head = None
        self.profile_dim = 0
        if config.public_profile_enabled:
            player_discard_input_dim = config.tile_embedding_dim + 2
            self.player_discard_encoder = nn.GRU(
                input_size=player_discard_input_dim,
                hidden_size=config.player_discard_hidden_dim,
                batch_first=True,
            )
            player_meld_input_dim = (
                config.tile_embedding_dim
                + config.player_embedding_dim
                + config.meld_type_embedding_dim
            )
            self.player_meld_encoder = nn.GRU(
                input_size=player_meld_input_dim,
                hidden_size=config.player_meld_hidden_dim,
                batch_first=True,
            )
            player_token_input_dim = (
                config.player_discard_hidden_dim
                + config.player_meld_hidden_dim
                + config.ready_embedding_dim
                + 2
            )
            self.player_token_encoder = nn.Sequential(
                nn.LayerNorm(player_token_input_dim),
                nn.Linear(player_token_input_dim, config.player_token_dim),
                nn.GELU(),
                nn.Linear(config.player_token_dim, config.player_token_dim),
                nn.GELU(),
            )
            self.profile_dim = PLAYER_PAD * config.player_token_dim
            self.profile_head = nn.Sequential(
                nn.LayerNorm(encoder_dim + self.profile_dim),
                nn.Linear(
                    encoder_dim + self.profile_dim,
                    config.profile_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.profile_hidden_dim, ACTION_SPACE_SIZE),
            )
            nn.init.zeros_(self.profile_head[-1].weight)
            nn.init.zeros_(self.profile_head[-1].bias)

        self.action_feature_head = None
        if config.action_feature_enabled:
            self.action_feature_head = ActionFeatureHead(
                encoder_dim,
                config.action_feature_hidden_dim,
                config.action_feature_intermediate_dim,
            )

    def encode_observation(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        hand_counts = batch["hand_counts"].float() / 4.0
        visible_counts = batch["visible_counts"].float() / 4.0
        waits = batch["wait_bits"].float()
        locked_waits = batch["locked_ready_wait_bits"].float()
        tile_planes = torch.stack(
            (hand_counts, visible_counts, waits, locked_waits),
            dim=1,
        )
        if self.suit_encoder is None:
            suit_features = hand_counts.new_empty((len(hand_counts), 0))
        else:
            suit_inputs = tile_planes.reshape(-1, 4, 3, 9).permute(
                0, 2, 1, 3
            )
            suit_features = self.suit_encoder(
                suit_inputs.reshape(-1, 4, 9)
            ).reshape(len(hand_counts), -1)

        discard_count = batch["discard_count"].long()
        discard_positions = torch.arange(
            MAX_DISCARD_EVENTS,
            device=discard_count.device,
        ).unsqueeze(0)
        discard_valid = discard_positions < discard_count.unsqueeze(1)
        discard_inputs = torch.cat(
            (
                self.tile_embedding(batch["discard_tiles"].long()),
                self.player_embedding(batch["discard_actors"].long()),
                batch["discard_was_called"].float().unsqueeze(-1),
            ),
            dim=-1,
        )
        discard_inputs = discard_inputs * discard_valid.unsqueeze(-1)
        packed_discards = pack_padded_sequence(
            discard_inputs,
            discard_count.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, discard_hidden = self.discard_encoder(packed_discards)
        discard_summary = discard_hidden[-1]
        discard_summary = discard_summary * (
            discard_count > 0
        ).unsqueeze(1)

        meld_valid = batch["meld_valid"].bool()
        meld_features = torch.cat(
            (
                self.tile_embedding(batch["meld_tiles"].long()),
                self.player_embedding(batch["meld_owners"].long()),
                self.player_embedding(batch["meld_sources"].long()),
                self.meld_type_embedding(batch["meld_types"].long()),
            ),
            dim=-1,
        )
        meld_features = meld_features * meld_valid.unsqueeze(-1)
        meld_summary = meld_features.sum(dim=1) / meld_valid.sum(
            dim=1,
            keepdim=True,
        ).clamp(min=1)
        if self.config.explicit_meld_counts:
            meld_owners = functional.one_hot(
                batch["meld_owners"].long(),
                num_classes=PLAYER_PAD + 1,
            )[:, :, :PLAYER_PAD]
            meld_owner_counts = (
                meld_owners * meld_valid.unsqueeze(-1)
            ).sum(dim=1).float() / 4.0
            meld_types = functional.one_hot(
                batch["meld_types"].long(),
                num_classes=MELD_TYPE_COUNT,
            )
            meld_type_counts = (
                meld_types * meld_valid.unsqueeze(-1)
            ).sum(dim=1).float() / 4.0
        else:
            meld_owner_counts = hand_counts.new_empty((len(hand_counts), 0))
            meld_type_counts = hand_counts.new_empty((len(hand_counts), 0))

        shanten_index = (batch["shanten"].long() + 1).clamp(0, 9)
        ready_features = self.ready_embedding(
            batch["ready_kinds"].long()
        ).flatten(start_dim=1)
        scalars = torch.stack(
            (
                batch["wall_remaining"].float() / 56.0,
                discard_count.float() / MAX_DISCARD_EVENTS,
                batch["meld_count"].float() / MAX_MELDS,
            ),
            dim=1,
        )

        encoded = torch.cat(
            (
                hand_counts,
                visible_counts,
                waits,
                locked_waits,
                suit_features,
                meld_owner_counts,
                meld_type_counts,
                scalars,
                self.shanten_embedding(shanten_index),
                self.phase_embedding(batch["phase"].long()),
                self.player_embedding(batch["current_actor"].long()),
                self.tile_embedding(batch["current_discard"].long()),
                self.player_embedding(
                    batch["current_discard_actor"].long()
                ),
                self.player_embedding(batch["dealer"].long()),
                ready_features,
                discard_summary,
                meld_summary,
            ),
            dim=1,
        )
        return encoded

    def encode_player_discards(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.player_discard_encoder is None:
            raise RuntimeError("public player profiles are disabled")

        discard_count = batch["discard_count"].long()
        positions = torch.arange(
            MAX_DISCARD_EVENTS,
            device=discard_count.device,
        ).unsqueeze(0)
        valid = positions < discard_count.unsqueeze(1)
        normalized_positions = positions.expand(
            len(discard_count), -1
        ).float() / max(MAX_DISCARD_EVENTS - 1, 1)
        inputs = torch.cat(
            (
                self.tile_embedding(batch["discard_tiles"].long()),
                batch["discard_was_called"].float().unsqueeze(-1),
                normalized_positions.unsqueeze(-1),
            ),
            dim=-1,
        )
        return self._encode_grouped_sequences(
            inputs,
            batch["discard_actors"].long(),
            valid,
            self.player_discard_encoder,
        )

    def encode_player_melds(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.player_meld_encoder is None:
            raise RuntimeError("public player profiles are disabled")

        inputs = torch.cat(
            (
                self.tile_embedding(batch["meld_tiles"].long()),
                self.player_embedding(batch["meld_sources"].long()),
                self.meld_type_embedding(batch["meld_types"].long()),
            ),
            dim=-1,
        )
        return self._encode_grouped_sequences(
            inputs,
            batch["meld_owners"].long(),
            batch["meld_valid"].bool(),
            self.player_meld_encoder,
        )

    @staticmethod
    def _encode_grouped_sequences(
        inputs: torch.Tensor,
        owners: torch.Tensor,
        valid: torch.Tensor,
        encoder: nn.GRU,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, event_count, feature_dim = inputs.shape
        player_ids = torch.arange(
            PLAYER_PAD,
            device=owners.device,
        ).view(1, PLAYER_PAD, 1)
        grouped_valid = (
            valid.unsqueeze(1)
            & (owners.unsqueeze(1) == player_ids)
        )
        positions = torch.arange(
            event_count,
            device=owners.device,
        ).view(1, 1, event_count)
        sort_keys = torch.where(
            grouped_valid,
            positions,
            positions + event_count,
        )
        order = sort_keys.argsort(dim=2)
        grouped_inputs = inputs.unsqueeze(1).expand(
            -1, PLAYER_PAD, -1, -1
        )
        grouped_inputs = torch.gather(
            grouped_inputs,
            2,
            order.unsqueeze(-1).expand(-1, -1, -1, feature_dim),
        )
        lengths = grouped_valid.sum(dim=2)
        packed = pack_padded_sequence(
            grouped_inputs.reshape(
                batch_size * PLAYER_PAD,
                event_count,
                feature_dim,
            ),
            lengths.reshape(-1).clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = encoder(packed)
        summaries = hidden[-1].reshape(batch_size, PLAYER_PAD, -1)
        summaries = summaries * (lengths > 0).unsqueeze(-1)
        return summaries, lengths

    def encode_public_profiles(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.player_token_encoder is None:
            raise RuntimeError("public player profiles are disabled")

        batch_size = len(batch["hand_counts"])
        reference = self.tile_embedding.weight
        if self.config.player_discard_profile_enabled:
            discard_summaries, discard_counts = (
                self.encode_player_discards(batch)
            )
        else:
            discard_summaries = reference.new_zeros((
                batch_size,
                PLAYER_PAD,
                self.config.player_discard_hidden_dim,
            ))
            discard_counts = batch["discard_count"].new_zeros((
                batch_size,
                PLAYER_PAD,
            ))
        if self.config.player_meld_profile_enabled:
            meld_summaries, meld_counts = self.encode_player_melds(batch)
        else:
            meld_summaries = reference.new_zeros((
                batch_size,
                PLAYER_PAD,
                self.config.player_meld_hidden_dim,
            ))
            meld_counts = batch["meld_count"].new_zeros((
                batch_size,
                PLAYER_PAD,
            ))

        counts = torch.stack(
            (
                discard_counts.float() / MAX_DISCARD_EVENTS,
                meld_counts.float() / MAX_MELDS,
            ),
            dim=-1,
        )
        player_inputs = torch.cat(
            (
                discard_summaries,
                meld_summaries,
                self.ready_embedding(batch["ready_kinds"].long()),
                counts,
            ),
            dim=-1,
        )
        return self.player_token_encoder(player_inputs).flatten(
            start_dim=1
        )

    def base_parameters(self) -> Iterable[nn.Parameter]:
        extension_prefixes = (
            "player_discard_encoder.",
            "player_meld_encoder.",
            "player_token_encoder.",
            "profile_head.",
            "action_feature_head.",
        )
        for name, parameter in self.named_parameters():
            if not name.startswith(extension_prefixes):
                yield parameter

    def profile_parameters(self) -> Iterable[nn.Parameter]:
        profile_prefixes = (
            "player_discard_encoder.",
            "player_meld_encoder.",
            "player_token_encoder.",
            "profile_head.",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(profile_prefixes):
                yield parameter

    def set_base_trainable(self, trainable: bool) -> None:
        for parameter in self.base_parameters():
            parameter.requires_grad_(trainable)

    def action_feature_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name.startswith("action_feature_head."):
                yield parameter

    def encode_policy_features(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        observation = self.encode_observation(batch)
        if self.player_token_encoder is None:
            return observation
        return torch.cat(
            (observation, self.encode_public_profiles(batch)),
            dim=1,
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        action_feature_scale: float = 1.0,
    ) -> torch.Tensor:
        observation = self.encode_observation(batch)
        logits = self.policy_head(observation)
        if self.profile_head is not None:
            profile = self.encode_public_profiles(batch)
            logits = logits + self.profile_head(
                torch.cat((observation, profile), dim=1)
            )
        if self.action_feature_head is not None:
            logits = logits + action_feature_scale * self.action_feature_head(
                observation,
                batch["action_features"],
            )
        return logits


def load_demonstration_dataset(
    directory: Path,
) -> DemonstrationTensorDataset:
    directory = Path(directory)
    shard_paths = sorted(directory.glob("demonstration_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(
            f"no demonstration shards found in {directory}"
        )

    pieces: Dict[str, list[np.ndarray]] = {
        name: [] for name in _ARRAY_SHAPES
    }
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            _validate_shard(shard_path, shard)
            for name in _ARRAY_SHAPES:
                pieces[name].append(np.asarray(shard[name]))

    arrays = {
        name: np.ascontiguousarray(np.concatenate(parts, axis=0))
        for name, parts in pieces.items()
    }
    _validate_feature_ranges(arrays)
    legal_mask = unpack_action_masks(arrays["legal_mask_words"])
    supervision_mask = unpack_action_masks(
        arrays["supervision_mask_words"]
    )
    target_mask = unpack_action_masks(
        arrays["target_action_mask_words"]
    )
    _validate_masks(
        arrays["selected_action_id"],
        legal_mask,
        supervision_mask,
        target_mask,
    )
    categories = classify_samples(supervision_mask, target_mask)

    tensors = _observation_arrays_to_tensors(arrays)
    for name in ("episode_seed", "player_id", "selected_action_id"):
        tensors[name] = torch.from_numpy(arrays[name])
    tensors["legal_mask"] = torch.from_numpy(legal_mask)
    tensors["supervision_mask"] = torch.from_numpy(supervision_mask)
    tensors["target_mask"] = torch.from_numpy(target_mask)
    tensors["wait_bits"] = torch.from_numpy(
        unpack_tile_bits(arrays["waits"])
    )
    tensors["locked_ready_wait_bits"] = torch.from_numpy(
        unpack_tile_bits(arrays["locked_ready_waits"])
    )
    tensors["category"] = torch.from_numpy(categories)
    return DemonstrationTensorDataset(tensors)


def tensorize_observations(
    observations: Sequence[object],
    *,
    include_action_features: bool = False,
    legal_mask_words: Sequence[Sequence[int]] | None = None,
) -> Dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("at least one observation is required")
    arrays = allocate_observation_arrays(len(observations))
    for index, observation in enumerate(observations):
        write_observation(arrays, index, observation)
    _validate_feature_ranges(arrays)
    tensors = _observation_arrays_to_tensors(arrays)
    if include_action_features:
        tensors["action_features"] = tensorize_action_features(
            observations,
            legal_mask_words,
        )
    return tensors


def split_by_episode(
    episode_seeds: Sequence[int],
    seed: int = 0,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> DatasetSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave test data")

    seeds_array = np.asarray(episode_seeds, dtype=np.int64)
    unique_seeds = np.unique(seeds_array)
    if len(unique_seeds) < 3:
        raise ValueError("at least three episodes are required for splitting")

    shuffled = unique_seeds.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = min(
        len(shuffled) - 2,
        max(1, int(len(shuffled) * train_fraction)),
    )
    validation_count = max(
        1,
        int(len(shuffled) * validation_fraction),
    )
    validation_count = min(
        len(shuffled) - train_end - 1,
        validation_count,
    )
    validation_end = train_end + validation_count

    train_seeds = shuffled[:train_end]
    validation_seeds = shuffled[train_end:validation_end]
    test_seeds = shuffled[validation_end:]
    return DatasetSplit(
        train_indices=np.flatnonzero(np.isin(seeds_array, train_seeds)),
        validation_indices=np.flatnonzero(
            np.isin(seeds_array, validation_seeds)
        ),
        test_indices=np.flatnonzero(np.isin(seeds_array, test_seeds)),
        train_seeds=tuple(map(int, train_seeds)),
        validation_seeds=tuple(map(int, validation_seeds)),
        test_seeds=tuple(map(int, test_seeds)),
    )


def set_valued_nll(
    logits: torch.Tensor,
    supervision_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    _validate_logit_masks(logits, supervision_mask, target_mask)
    supervised_logits = logits.masked_fill(~supervision_mask, -torch.inf)
    target_logits = logits.masked_fill(~target_mask, -torch.inf)
    return (
        torch.logsumexp(supervised_logits, dim=1)
        - torch.logsumexp(target_logits, dim=1)
    )


def evaluate_logits(
    logits: torch.Tensor,
    supervision_mask: torch.Tensor,
    target_mask: torch.Tensor,
    categories: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> Dict[str, object]:
    losses = set_valued_nll(logits, supervision_mask, target_mask)
    supervised_logits = logits.masked_fill(~supervision_mask, -torch.inf)
    predicted = supervised_logits.argmax(dim=1)
    correct = target_mask.gather(1, predicted.unsqueeze(1)).squeeze(1)

    report: Dict[str, object] = {
        "sample_count": int(len(logits)),
        "loss_sum": float(losses.sum().item()),
        "correct_count": int(correct.sum().item()),
        "categories": {},
    }
    category_reports = report["categories"]
    assert isinstance(category_reports, dict)
    for category_id, name in enumerate(CATEGORY_NAMES):
        selected = categories == category_id
        category_reports[name] = {
            "sample_count": int(selected.sum().item()),
            "correct_count": int(correct[selected].sum().item()),
        }

    for name, action_id in (
        ("pon", PON_ACTION_ID),
        ("pass", PASS_ACTION_ID),
        ("tsumo", TSUMO_ACTION_ID),
        ("ron", RON_ACTION_ID),
    ):
        expected = target_mask[:, action_id]
        report[name] = {
            "sample_count": int(expected.sum().item()),
            "correct_count": int(
                ((predicted == action_id) & expected).sum().item()
            ),
        }

    if legal_mask is not None:
        if legal_mask.shape != logits.shape or legal_mask.dtype != torch.bool:
            raise ValueError("legal mask must be a bool tensor matching logits")
        legal_predicted = logits.masked_fill(
            ~legal_mask,
            -torch.inf,
        ).argmax(dim=1)
        legal_target_hit = target_mask.gather(
            1,
            legal_predicted.unsqueeze(1),
        ).squeeze(1)
        legal_is_supervised = supervision_mask.gather(
            1,
            legal_predicted.unsqueeze(1),
        ).squeeze(1)
        report["legal_policy"] = {
            "target_count": int(legal_target_hit.sum().item()),
            "outside_supervision_count": int(
                (~legal_is_supervised).sum().item()
            ),
        }
    return report


def merge_evaluation_reports(
    reports: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    merged: Dict[str, object] = {
        "sample_count": 0,
        "loss_sum": 0.0,
        "correct_count": 0,
        "categories": {
            name: {"sample_count": 0, "correct_count": 0}
            for name in CATEGORY_NAMES
        },
        "pon": {"sample_count": 0, "correct_count": 0},
        "pass": {"sample_count": 0, "correct_count": 0},
        "tsumo": {"sample_count": 0, "correct_count": 0},
        "ron": {"sample_count": 0, "correct_count": 0},
        "legal_policy": {
            "target_count": 0,
            "outside_supervision_count": 0,
        },
    }
    for report in reports:
        for key in ("sample_count", "correct_count"):
            merged[key] = int(merged[key]) + int(report[key])
        merged["loss_sum"] = float(merged["loss_sum"]) + float(
            report["loss_sum"]
        )
        for name in CATEGORY_NAMES:
            _merge_count_pair(
                merged["categories"][name],
                report["categories"][name],
            )
        for name in ("pon", "pass", "tsumo", "ron"):
            _merge_count_pair(merged[name], report[name])
        if "legal_policy" in report:
            for key in ("target_count", "outside_supervision_count"):
                merged["legal_policy"][key] += int(
                    report["legal_policy"][key]
                )
    return finalize_evaluation_report(merged)


def finalize_evaluation_report(report: Mapping[str, object]) -> Dict[str, object]:
    sample_count = int(report["sample_count"])
    result = dict(report)
    result["loss"] = (
        float(report["loss_sum"]) / sample_count
        if sample_count
        else 0.0
    )
    result["accuracy"] = _ratio(
        int(report["correct_count"]),
        sample_count,
    )

    categories = {}
    category_accuracies = []
    for name in CATEGORY_NAMES:
        counts = report["categories"][name]
        count = int(counts["sample_count"])
        correct = int(counts["correct_count"])
        categories[name] = {
            "sample_count": count,
            "correct_count": correct,
            "accuracy": _ratio(correct, count),
        }
        if count:
            category_accuracies.append(_ratio(correct, count))
    result["categories"] = categories
    result["macro_accuracy"] = (
        sum(category_accuracies) / len(category_accuracies)
        if category_accuracies
        else 0.0
    )

    for name in ("pon", "pass", "tsumo", "ron"):
        counts = report[name]
        count = int(counts["sample_count"])
        correct = int(counts["correct_count"])
        result[name] = {
            "sample_count": count,
            "correct_count": correct,
            "accuracy": _ratio(correct, count),
        }

    decision_accuracies = [categories["discard"]["accuracy"]]
    for name in ("pon", "pass"):
        if result[name]["sample_count"]:
            decision_accuracies.append(result[name]["accuracy"])
    if categories["hu"]["sample_count"]:
        decision_accuracies.append(categories["hu"]["accuracy"])
    result["decision_balanced_accuracy"] = (
        sum(decision_accuracies) / len(decision_accuracies)
    )

    legal = report["legal_policy"]
    result["legal_policy"] = {
        "target_count": int(legal["target_count"]),
        "target_rate": _ratio(int(legal["target_count"]), sample_count),
        "outside_supervision_count": int(
            legal["outside_supervision_count"]
        ),
        "outside_supervision_rate": _ratio(
            int(legal["outside_supervision_count"]),
            sample_count,
        ),
    }
    return result


def policy_config_dict(config: PolicyConfig) -> Dict[str, object]:
    return asdict(config)


def policy_architecture_version(config: PolicyConfig) -> int:
    if config.action_feature_enabled:
        return 4
    if config.public_profile_enabled:
        return 3
    if config.tile_conv_channels or config.explicit_meld_counts:
        return 2
    return 1


def policy_config_from_dict(values: Mapping[str, object]) -> PolicyConfig:
    migrated = dict(values)
    if "explicit_meld_counts" not in migrated:
        migrated["explicit_meld_counts"] = bool(
            migrated.get("tile_conv_channels", 0)
        )
    return PolicyConfig(**migrated)


def upgrade_policy_with_public_profiles(
    policy: BehaviorClonePolicy,
    *,
    discard_enabled: bool = True,
    meld_enabled: bool = True,
    player_discard_hidden_dim: int = 32,
    player_meld_hidden_dim: int = 32,
    player_token_dim: int = 64,
    profile_hidden_dim: int = 256,
) -> BehaviorClonePolicy:
    if policy.config.public_profile_enabled:
        raise ValueError("policy already has public player profiles")

    config = replace(
        policy.config,
        public_profile_enabled=True,
        player_discard_profile_enabled=discard_enabled,
        player_meld_profile_enabled=meld_enabled,
        player_discard_hidden_dim=player_discard_hidden_dim,
        player_meld_hidden_dim=player_meld_hidden_dim,
        player_token_dim=player_token_dim,
        profile_hidden_dim=profile_hidden_dim,
    )
    reference = next(policy.parameters())
    upgraded = BehaviorClonePolicy(config).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    upgraded_state = upgraded.state_dict()
    for name, value in policy.state_dict().items():
        if name in upgraded_state and upgraded_state[name].shape == value.shape:
            upgraded_state[name] = value.detach().clone()
    upgraded.load_state_dict(upgraded_state, strict=True)
    upgraded.train(policy.training)
    return upgraded


def upgrade_policy_with_action_features(
    policy: BehaviorClonePolicy,
    *,
    hidden_dim: int = 64,
    intermediate_dim: int = 32,
) -> BehaviorClonePolicy:
    if policy.config.action_feature_enabled:
        raise ValueError("policy already has action features")

    config = replace(
        policy.config,
        action_feature_enabled=True,
        action_feature_hidden_dim=hidden_dim,
        action_feature_intermediate_dim=intermediate_dim,
    )
    reference = next(policy.parameters())
    upgraded = BehaviorClonePolicy(config).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    upgraded_state = upgraded.state_dict()
    for name, value in policy.state_dict().items():
        if name not in upgraded_state or upgraded_state[name].shape != value.shape:
            raise ValueError(f"cannot preserve policy parameter {name}")
        upgraded_state[name] = value.detach().clone()
    upgraded.load_state_dict(upgraded_state, strict=True)
    upgraded.train(policy.training)
    return upgraded


def load_policy_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> Tuple[BehaviorClonePolicy, Mapping[str, object]]:
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected_versions = {
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "demonstration_format_version": DEMONSTRATION_FORMAT_VERSION,
    }
    for name, expected in expected_versions.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"checkpoint has an incompatible {name}")

    config = policy_config_from_dict(checkpoint["policy_config"])
    if config.action_feature_enabled and checkpoint.get(
        "action_feature_version"
    ) != ACTION_FEATURE_VERSION:
        raise ValueError("checkpoint has an incompatible action_feature_version")
    expected_architecture = policy_architecture_version(config)
    if checkpoint.get(
        "policy_architecture_version",
        expected_architecture,
    ) != expected_architecture:
        raise ValueError("checkpoint policy architecture is inconsistent")

    model = BehaviorClonePolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def unpack_action_masks(mask_words: np.ndarray) -> np.ndarray:
    if mask_words.ndim != 2 or mask_words.shape[1] != 2:
        raise ValueError("packed action masks must have shape (N, 2)")
    action_ids = np.arange(ACTION_SPACE_SIZE, dtype=np.uint64)
    word_indices = (action_ids // 64).astype(np.intp)
    bit_indices = action_ids % 64
    return np.ascontiguousarray(
        ((mask_words[:, word_indices] >> bit_indices) & 1).astype(
            np.bool_
        )
    )


def unpack_tile_bits(values: np.ndarray) -> np.ndarray:
    bit_indices = np.arange(TILE_TYPE_COUNT, dtype=np.uint32)
    return np.ascontiguousarray(
        ((values[:, None].astype(np.uint32) >> bit_indices) & 1).astype(
            np.bool_
        )
    )


def classify_samples(
    supervision_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    hu = target_mask[:, TSUMO_ACTION_ID] | target_mask[:, RON_ACTION_ID]
    pon_pass = (
        supervision_mask[:, PON_ACTION_ID]
        & supervision_mask[:, PASS_ACTION_ID]
        & ~hu
    )
    discard = (
        supervision_mask[:, :TILE_TYPE_COUNT].any(axis=1)
        & ~supervision_mask[:, TILE_TYPE_COUNT:].any(axis=1)
        & ~hu
    )
    membership_count = (
        discard.astype(np.uint8)
        + pon_pass.astype(np.uint8)
        + hu.astype(np.uint8)
    )
    if np.any(membership_count != 1):
        bad_count = int(np.count_nonzero(membership_count != 1))
        raise ValueError(f"{bad_count} samples have an unknown category")

    result = np.empty(len(supervision_mask), dtype=np.int64)
    result[discard] = DISCARD_CATEGORY
    result[pon_pass] = PON_PASS_CATEGORY
    result[hu] = HU_CATEGORY
    return result


def _observation_arrays_to_tensors(
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(arrays[name]))
        for name in _OBSERVATION_ARRAY_NAMES
        if name not in {"waits", "locked_ready_waits"}
    }
    tensors["wait_bits"] = torch.from_numpy(
        unpack_tile_bits(arrays["waits"])
    )
    tensors["locked_ready_wait_bits"] = torch.from_numpy(
        unpack_tile_bits(arrays["locked_ready_waits"])
    )
    return tensors


def _validate_shard(shard_path: Path, shard) -> None:
    versions = (
        ("demonstration_format_version", DEMONSTRATION_FORMAT_VERSION),
        ("observation_version", OBSERVATION_VERSION),
        ("action_encoding_version", ACTION_ENCODING_VERSION),
    )
    for name, expected in versions:
        if name not in shard or int(shard[name]) != expected:
            raise ValueError(f"{shard_path} has an incompatible {name}")

    if "sample_count" not in shard:
        raise ValueError(f"{shard_path} has no sample count")
    sample_count = int(shard["sample_count"])
    for name, trailing_shape in _ARRAY_SHAPES.items():
        if name not in shard:
            raise ValueError(f"{shard_path} is missing {name}")
        expected_shape = (sample_count,) + trailing_shape
        if shard[name].shape != expected_shape:
            raise ValueError(
                f"{shard_path}:{name} has shape {shard[name].shape}, "
                f"expected {expected_shape}"
            )


def _validate_masks(
    selected_action_ids: np.ndarray,
    legal_mask: np.ndarray,
    supervision_mask: np.ndarray,
    target_mask: np.ndarray,
) -> None:
    if np.any(supervision_mask & ~legal_mask):
        raise ValueError("supervised actions must be legal")
    if np.any(target_mask & ~supervision_mask):
        raise ValueError("target actions must be supervised")
    if np.any(~target_mask.any(axis=1)):
        raise ValueError("every demonstration must have a target action")
    selected = selected_action_ids.astype(np.int64)
    if np.any((selected < 0) | (selected >= ACTION_SPACE_SIZE)):
        raise ValueError("selected action is outside the action space")
    rows = np.arange(len(selected))
    if np.any(~legal_mask[rows, selected]):
        raise ValueError("selected action must be legal")


def _validate_feature_ranges(arrays: Mapping[str, np.ndarray]) -> None:
    bounded = {
        "hand_counts": (0, 4),
        "visible_counts": (0, 4),
        "discard_tiles": (0, TILE_PAD),
        "discard_actors": (0, PLAYER_PAD),
        "discard_was_called": (0, 1),
        "discard_count": (0, MAX_DISCARD_EVENTS),
        "meld_tiles": (0, TILE_PAD),
        "meld_owners": (0, PLAYER_PAD),
        "meld_sources": (0, PLAYER_PAD),
        "meld_types": (0, MELD_TYPE_COUNT - 1),
        "meld_valid": (0, 1),
        "meld_count": (0, MAX_MELDS),
        "shanten": (-1, 8),
        "phase": (0, ROUND_PHASE_COUNT - 1),
        "current_actor": (0, PLAYER_PAD),
        "current_discard": (0, TILE_PAD),
        "current_discard_actor": (0, PLAYER_PAD),
        "dealer": (0, 3),
        "wall_remaining": (0, 56),
        "ready_kinds": (0, READY_KIND_COUNT - 1),
        "player_id": (0, 3),
    }
    for name, (minimum, maximum) in bounded.items():
        if name not in arrays:
            continue
        values = arrays[name]
        if np.any((values < minimum) | (values > maximum)):
            raise ValueError(
                f"{name} contains values outside {minimum}..{maximum}"
            )

    for name in ("waits", "locked_ready_waits"):
        if name not in arrays:
            continue
        if np.any(arrays[name].astype(np.uint32) >> TILE_TYPE_COUNT):
            raise ValueError(f"{name} contains bits outside the tile space")
    used_high_bits = ACTION_SPACE_SIZE - 64
    for name in (
        "legal_mask_words",
        "supervision_mask_words",
        "target_action_mask_words",
    ):
        if name not in arrays:
            continue
        if np.any(arrays[name][:, 1] >> used_high_bits):
            raise ValueError(f"{name} contains bits outside the action space")


def _validate_logit_masks(
    logits: torch.Tensor,
    supervision_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    if logits.ndim != 2 or logits.shape[1] != ACTION_SPACE_SIZE:
        raise ValueError("logits must have shape (N, ACTION_SPACE_SIZE)")
    if supervision_mask.shape != logits.shape:
        raise ValueError("supervision mask must match logits")
    if target_mask.shape != logits.shape:
        raise ValueError("target mask must match logits")
    if supervision_mask.dtype != torch.bool or target_mask.dtype != torch.bool:
        raise ValueError("action masks must be bool tensors")
    if torch.any(target_mask & ~supervision_mask):
        raise ValueError("target mask must be a subset of supervision mask")
    if torch.any(~target_mask.any(dim=1)):
        raise ValueError("every row must contain a target action")


def _merge_count_pair(target: dict, source: Mapping[str, object]) -> None:
    target["sample_count"] += int(source["sample_count"])
    target["correct_count"] += int(source["correct_count"])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
