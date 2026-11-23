from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ._gymj import (
    ACTION_SPACE_SIZE,
    MAX_TILE_COPIES,
    PRIVILEGED_FUTURE_DRAW_COUNT,
    PRIVILEGED_PATTERN_VERSION,
    PRIVILEGED_STATE_VERSION,
    PRIVILEGED_TILE_PAD,
    TILE_TYPE_COUNT,
)
from .critic import CriticValues
from .v4_actor import V4Actor, V4ActorConfig, V4PublicStateEncoder
from .v4_features import (
    V4_HAND_SUMMARY_DIM,
    V4_HAND_TILE_DIM,
    V4_PLAYER_COUNT,
    V4TensorBatch,
    tensorize_v4_decisions,
    v4_hand_pattern_arrays,
)


V4_PRIVILEGED_CRITIC_ARCHITECTURE = "v4_structured_privileged_v1"
V4_PRIVILEGED_TENSOR_VERSION = 1
V4_REQUIRED_PRIVILEGED_STATE_VERSION = 3
V4_REQUIRED_PRIVILEGED_PATTERN_VERSION = 2
_MAX_FIXED_CHICKENS = 8.0


@dataclass(frozen=True)
class V4CriticPublicTensorBatch:
    """Compact V4 public state used by the Critic.

    The Actor's action-specific simulation tensors are intentionally omitted.
    V4PublicStateEncoder only consumes the fields below.
    """

    hand_summary: torch.Tensor
    hand_tiles: torch.Tensor
    discard_tiles: torch.Tensor
    discard_players: torch.Tensor
    discard_called: torch.Tensor
    discard_valid: torch.Tensor
    meld_tiles: torch.Tensor
    meld_owners: torch.Tensor
    meld_sources: torch.Tensor
    meld_types: torch.Tensor
    meld_valid: torch.Tensor
    player_tiles: torch.Tensor
    player_summary: torch.Tensor
    context: torch.Tensor
    phase: torch.Tensor
    current_actor: torch.Tensor
    current_discard: torch.Tensor
    current_discard_actor: torch.Tensor
    dealer: torch.Tensor
    ready_kinds: torch.Tensor
    current_waits: torch.Tensor
    locked_ready_waits: torch.Tensor
    progress: torch.Tensor
    last_draw_kind: torch.Tensor
    hot_discard_player: torch.Tensor
    action_legal: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.hand_summary.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "V4CriticPublicTensorBatch":
        return replace(
            self,
            **{
                item.name: getattr(self, item.name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for item in fields(self)
            },
        )

    def validate(self) -> None:
        # Reuse the established V4 state dimensions without duplicating them.
        batch = self.batch_size
        if self.action_legal.shape != (batch, ACTION_SPACE_SIZE):
            raise ValueError("V4 Critic legal mask has the wrong shape")
        if self.action_legal.dtype != torch.bool:
            raise TypeError("V4 Critic legal mask must use torch.bool")
        if not self.action_legal.any(dim=1).all():
            raise ValueError("V4 Critic state must contain a legal action")
        devices = {
            getattr(self, item.name).device
            for item in fields(self)
        }
        if len(devices) != 1:
            raise ValueError("all V4 Critic public tensors must share a device")


def tensorize_v4_critic_public_states(
    decisions: Sequence,
    device: torch.device | str | None = None,
) -> V4CriticPublicTensorBatch:
    """Tensorize only the public fields consumed by the V4 Critic."""

    full = tensorize_v4_decisions(decisions)
    result = V4CriticPublicTensorBatch(**{
        item.name: getattr(full, item.name)
        for item in fields(V4CriticPublicTensorBatch)
    })
    if device is not None:
        result = result.to(device)
    result.validate()
    return result


def v4_critic_public_from_actor_batch(
    batch: V4TensorBatch,
) -> V4CriticPublicTensorBatch:
    """Select the Critic's public fields from an already tensorized V4 state."""

    result = V4CriticPublicTensorBatch(**{
        item.name: getattr(batch, item.name)
        for item in fields(V4CriticPublicTensorBatch)
    })
    result.validate()
    return result


@dataclass(frozen=True)
class V4PrivilegedTensorBatch:
    player_hand_summary: torch.Tensor
    player_hand_tiles: torch.Tensor
    future_draw_tiles: torch.Tensor
    future_draw_valid: torch.Tensor
    replacement_draw_tiles: torch.Tensor
    replacement_draw_valid: torch.Tensor
    wall_counts: torch.Tensor
    player_tenpai: torch.Tensor
    signed_fixed_chickens: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.player_hand_summary.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "V4PrivilegedTensorBatch":
        return replace(
            self,
            **{
                item.name: getattr(self, item.name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for item in fields(self)
            },
        )

    def index_select(
        self,
        indices: torch.Tensor,
    ) -> "V4PrivilegedTensorBatch":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError(
                "V4 privileged indices must be a one-dimensional long tensor"
            )
        return replace(
            self,
            **{
                item.name: getattr(self, item.name).index_select(
                    0,
                    indices.to(getattr(self, item.name).device),
                )
                for item in fields(self)
            },
        )

    def validate(self) -> None:
        batch = self.batch_size
        draw_count = PRIVILEGED_FUTURE_DRAW_COUNT
        expected = {
            "player_hand_summary": (
                batch,
                V4_PLAYER_COUNT,
                V4_HAND_SUMMARY_DIM,
            ),
            "player_hand_tiles": (
                batch,
                V4_PLAYER_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
            "future_draw_tiles": (
                batch,
                V4_PLAYER_COUNT,
                draw_count,
            ),
            "future_draw_valid": (
                batch,
                V4_PLAYER_COUNT,
                draw_count,
            ),
            "replacement_draw_tiles": (batch, draw_count),
            "replacement_draw_valid": (batch, draw_count),
            "wall_counts": (batch, TILE_TYPE_COUNT),
            "player_tenpai": (batch, V4_PLAYER_COUNT),
            "signed_fixed_chickens": (batch, V4_PLAYER_COUNT),
        }
        integer_fields = {"future_draw_tiles", "replacement_draw_tiles"}
        bool_fields = {"future_draw_valid", "replacement_draw_valid"}
        devices = set()
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name in integer_fields and tensor.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long")
            if name in bool_fields and tensor.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool")
            if name not in integer_fields | bool_fields:
                if tensor.dtype != torch.float32:
                    raise TypeError(f"{name} must use torch.float32")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{name} contains non-finite values")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise ValueError("all V4 privileged tensors must share a device")
        for tiles, valid, name in (
            (
                self.future_draw_tiles,
                self.future_draw_valid,
                "future draw",
            ),
            (
                self.replacement_draw_tiles,
                self.replacement_draw_valid,
                "replacement draw",
            ),
        ):
            if torch.any(valid & ((tiles < 0) | (tiles >= TILE_TYPE_COUNT))):
                raise ValueError(f"active {name} tile is invalid")
            if torch.any(~valid & (tiles != PRIVILEGED_TILE_PAD)):
                raise ValueError(f"inactive {name} tile must be padded")
            if torch.any(valid[..., 1:] & ~valid[..., :-1]):
                raise ValueError(f"{name} validity must be a prefix")
        if torch.any((self.wall_counts < 0.0) | (self.wall_counts > 1.0)):
            raise ValueError("normalized wall counts must be in range 0..1")
        if torch.any(
            (self.player_tenpai < 0.0) | (self.player_tenpai > 1.0)
        ):
            raise ValueError("tenpai flags must be in range 0..1")
        if torch.any(
            (self.signed_fixed_chickens < -1.0)
            | (self.signed_fixed_chickens > 1.0)
        ):
            raise ValueError("normalized chicken counts must be in range -1..1")


def tensorize_v4_privileged_states(
    decisions: Sequence,
    states: Sequence,
    patterns: Sequence,
    device: torch.device | str | None = None,
) -> V4PrivilegedTensorBatch:
    if PRIVILEGED_STATE_VERSION != V4_REQUIRED_PRIVILEGED_STATE_VERSION:
        raise RuntimeError("V4 Critic requires privileged state version 3")
    if PRIVILEGED_PATTERN_VERSION != V4_REQUIRED_PRIVILEGED_PATTERN_VERSION:
        raise RuntimeError("V4 Critic requires privileged pattern version 2")
    if not decisions:
        raise ValueError("V4 privileged batch cannot be empty")
    if len(decisions) != len(states) or len(states) != len(patterns):
        raise ValueError("V4 privileged inputs must have equal lengths")

    batch = len(decisions)
    draw_count = PRIVILEGED_FUTURE_DRAW_COUNT
    arrays = {
        "player_hand_summary": np.zeros(
            (batch, V4_PLAYER_COUNT, V4_HAND_SUMMARY_DIM),
            np.float32,
        ),
        "player_hand_tiles": np.zeros(
            (
                batch,
                V4_PLAYER_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
            np.float32,
        ),
        "future_draw_tiles": np.full(
            (batch, V4_PLAYER_COUNT, draw_count),
            PRIVILEGED_TILE_PAD,
            np.int64,
        ),
        "future_draw_valid": np.zeros(
            (batch, V4_PLAYER_COUNT, draw_count),
            np.bool_,
        ),
        "replacement_draw_tiles": np.full(
            (batch, draw_count),
            PRIVILEGED_TILE_PAD,
            np.int64,
        ),
        "replacement_draw_valid": np.zeros(
            (batch, draw_count),
            np.bool_,
        ),
        "wall_counts": np.zeros(
            (batch, TILE_TYPE_COUNT),
            np.float32,
        ),
        "player_tenpai": np.zeros(
            (batch, V4_PLAYER_COUNT),
            np.float32,
        ),
        "signed_fixed_chickens": np.zeros(
            (batch, V4_PLAYER_COUNT),
            np.float32,
        ),
    }

    for row, (decision, state, pattern) in enumerate(
        zip(decisions, states, patterns)
    ):
        opponent_hands = np.asarray(state.opponent_hands, dtype=np.int16)
        if opponent_hands.shape != (V4_PLAYER_COUNT - 1, TILE_TYPE_COUNT):
            raise ValueError("opponent hands must have shape [3, 27]")
        self_hand = np.asarray(
            decision.observation.hand_counts,
            dtype=np.int16,
        )
        concealed = np.concatenate((self_hand[None, :], opponent_hands), axis=0)
        wall_counts = np.asarray(
            pattern.remaining_wall_counts,
            dtype=np.int16,
        )
        if wall_counts.shape != (TILE_TYPE_COUNT,):
            raise ValueError("remaining wall counts must have 27 entries")
        if np.any(wall_counts < 0) or np.any(wall_counts > MAX_TILE_COPIES):
            raise ValueError("remaining wall count is outside 0..4")
        visible = MAX_TILE_COPIES - wall_counts
        if np.any(visible[None, :] < concealed):
            raise ValueError("privileged visibility excludes a concealed tile")
        player_patterns = list(pattern.player_patterns)
        if len(player_patterns) != V4_PLAYER_COUNT:
            raise ValueError("privileged patterns must contain four players")
        for player in range(V4_PLAYER_COUNT):
            summary, tile_features = v4_hand_pattern_arrays(
                player_patterns[player],
                concealed[player],
                visible,
            )
            arrays["player_hand_summary"][row, player] = summary
            arrays["player_hand_tiles"][row, player] = tile_features

        future = np.asarray(state.future_draw_tiles, dtype=np.int64)
        replacement = np.asarray(
            state.replacement_draw_tiles,
            dtype=np.int64,
        )
        if future.shape != (V4_PLAYER_COUNT, draw_count):
            raise ValueError("future draw tiles must have shape [4, 3]")
        if replacement.shape != (draw_count,):
            raise ValueError("replacement draw tiles must have shape [3]")
        arrays["future_draw_tiles"][row] = future
        arrays["future_draw_valid"][row] = future != PRIVILEGED_TILE_PAD
        arrays["replacement_draw_tiles"][row] = replacement
        arrays["replacement_draw_valid"][row] = (
            replacement != PRIVILEGED_TILE_PAD
        )
        arrays["wall_counts"][row] = wall_counts / float(MAX_TILE_COPIES)

        raw_tenpai = np.asarray(state.player_tenpai, dtype=np.float32)
        economy_tenpai = np.asarray(
            pattern.economy.player_tenpai,
            dtype=np.float32,
        )
        if not np.array_equal(raw_tenpai, economy_tenpai):
            raise ValueError("privileged tenpai encoders disagree")
        arrays["player_tenpai"][row] = raw_tenpai
        arrays["signed_fixed_chickens"][row] = np.asarray(
            pattern.economy.signed_fixed_chicken_counts,
            dtype=np.float32,
        ) / _MAX_FIXED_CHICKENS

    result = V4PrivilegedTensorBatch(**{
        name: torch.from_numpy(value)
        for name, value in arrays.items()
    })
    if device is not None:
        result = result.to(device)
    result.validate()
    return result


@dataclass(frozen=True)
class V4PrivilegedCriticConfig:
    public_config: V4ActorConfig = field(default_factory=V4ActorConfig)
    legal_mask_hidden_dim: int = 32
    future_draw_hidden_dim: int = 32
    player_hidden_dim: int = 128
    wall_tile_hidden_dim: int = 16
    wall_hidden_dim: int = 64
    fusion_hidden_dim: int = 256


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class ShortDrawSequenceEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tile_ids: torch.Tensor,
        valid: torch.Tensor,
        tile_embedding: nn.Embedding,
    ) -> torch.Tensor:
        if tile_ids.shape != valid.shape or tile_ids.ndim < 2:
            raise ValueError("draw tiles and validity must have equal sequence shapes")
        prefix_shape = tile_ids.shape[:-1]
        sequence_length = tile_ids.shape[-1]
        flat_tiles = tile_ids.reshape(-1, sequence_length)
        flat_valid = valid.reshape(-1, sequence_length)
        embedded = tile_embedding(flat_tiles)
        outputs, _ = self.gru(embedded)
        lengths = flat_valid.sum(dim=1)
        last_indices = (lengths - 1).clamp_min(0)
        selected = outputs[
            torch.arange(outputs.shape[0], device=outputs.device),
            last_indices,
        ]
        selected = self.output_norm(selected)
        selected = selected * (lengths > 0).to(selected.dtype).unsqueeze(1)
        return selected.reshape(*prefix_shape, self.hidden_dim)


class V4PrivilegedGuiyangCritic(nn.Module):
    """Structured training-only Critic with V4 public and hidden features."""

    def __init__(
        self,
        config: V4PrivilegedCriticConfig = V4PrivilegedCriticConfig(),
    ):
        super().__init__()
        dimensions = (
            config.legal_mask_hidden_dim,
            config.future_draw_hidden_dim,
            config.player_hidden_dim,
            config.wall_tile_hidden_dim,
            config.wall_hidden_dim,
            config.fusion_hidden_dim,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("V4 privileged critic dimensions must be positive")

        self.config = config
        public = config.public_config
        self.public_encoder = V4PublicStateEncoder(public)
        self.legal_mask_encoder = nn.Sequential(
            nn.Linear(ACTION_SPACE_SIZE, config.legal_mask_hidden_dim),
            nn.GELU(),
        )
        self.future_draw_encoder = ShortDrawSequenceEncoder(
            public.tile_embedding_dim,
            config.future_draw_hidden_dim,
        )
        self.replacement_draw_encoder = ShortDrawSequenceEncoder(
            public.tile_embedding_dim,
            config.future_draw_hidden_dim,
        )
        player_input_dim = (
            public.player_output_dim
            + public.hand_output_dim
            + config.future_draw_hidden_dim
            + 2
        )
        self.player_fusion = _mlp(
            player_input_dim,
            config.player_hidden_dim,
            config.player_hidden_dim,
            public.dropout,
        )
        self.wall_tile_encoder = _mlp(
            public.tile_embedding_dim + 1,
            config.wall_tile_hidden_dim,
            config.wall_tile_hidden_dim,
            public.dropout,
        )
        self.wall_encoder = _mlp(
            TILE_TYPE_COUNT * config.wall_tile_hidden_dim,
            config.wall_hidden_dim,
            config.wall_hidden_dim,
            public.dropout,
        )
        fusion_input_dim = (
            public.state_hidden_dim
            + config.legal_mask_hidden_dim
            + V4_PLAYER_COUNT * config.player_hidden_dim
            + config.wall_hidden_dim
            + config.future_draw_hidden_dim
        )
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(public.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_hidden_dim),
        )
        self.score_head = nn.Linear(config.fusion_hidden_dim, 1)
        self.shanten_head = nn.Linear(config.fusion_hidden_dim, 1)
        self._initialize_value_heads()

    def _initialize_value_heads(self) -> None:
        for head in (self.score_head, self.shanten_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def value_feature_dim(self) -> int:
        return self.config.fusion_hidden_dim

    def initialize_public_encoder(self, actor: V4Actor) -> None:
        if actor.config != self.config.public_config:
            raise ValueError("V4 actor config does not match public encoder")
        actor_state = actor.state_dict()
        public_state = {
            name: actor_state[name]
            for name in self.public_encoder.state_dict()
        }
        self.public_encoder.load_state_dict(public_state, strict=True)

    def forward(
        self,
        batch: V4TensorBatch | V4CriticPublicTensorBatch,
        privileged: V4PrivilegedTensorBatch,
    ) -> CriticValues:
        return self.values_from_features(
            self.encode_value_features(batch, privileged)
        )

    def values_from_features(self, features: torch.Tensor) -> CriticValues:
        if features.ndim != 2 or features.shape[1] != self.value_feature_dim:
            raise ValueError(
                "value features must have shape [batch, value_feature_dim]"
            )
        return CriticValues(
            score_value=self.score_head(features).squeeze(-1),
            shanten_value=self.shanten_head(features).squeeze(-1),
        )

    def encode_value_features(
        self,
        batch: V4TensorBatch | V4CriticPublicTensorBatch,
        privileged: V4PrivilegedTensorBatch,
    ) -> torch.Tensor:
        batch_size = batch.batch_size
        if privileged.batch_size != batch_size:
            raise ValueError("public and privileged batch sizes differ")
        if batch.action_legal.shape != (batch_size, ACTION_SPACE_SIZE):
            raise ValueError("V4 legal mask must have shape [batch, actions]")

        public_state, public_players = self.public_encoder.encode_state(batch)
        device = public_state.device
        dtype = public_state.dtype
        if any(
            getattr(privileged, item.name).device != device
            for item in fields(privileged)
        ):
            raise ValueError("public and privileged tensors must share a device")

        tile_ids = torch.arange(TILE_TYPE_COUNT, device=device)
        tile_table = self.public_encoder.tile_embedding(tile_ids)
        true_hands = self.public_encoder.hand_encoder(
            privileged.player_hand_summary.reshape(
                batch_size * V4_PLAYER_COUNT,
                V4_HAND_SUMMARY_DIM,
            ),
            privileged.player_hand_tiles.reshape(
                batch_size * V4_PLAYER_COUNT,
                TILE_TYPE_COUNT,
                V4_HAND_TILE_DIM,
            ),
            tile_table,
        ).reshape(batch_size, V4_PLAYER_COUNT, -1)
        future_draws = self.future_draw_encoder(
            privileged.future_draw_tiles,
            privileged.future_draw_valid,
            self.public_encoder.tile_embedding,
        )
        economy = torch.stack(
            (
                privileged.player_tenpai,
                privileged.signed_fixed_chickens,
            ),
            dim=-1,
        ).to(dtype=dtype)
        player_tokens = self.player_fusion(torch.cat(
            (public_players, true_hands, future_draws, economy),
            dim=-1,
        ))

        expanded_tiles = tile_table.unsqueeze(0).expand(batch_size, -1, -1)
        wall_tile_tokens = self.wall_tile_encoder(torch.cat(
            (
                expanded_tiles,
                privileged.wall_counts.to(dtype=dtype).unsqueeze(-1),
            ),
            dim=-1,
        ))
        wall_token = self.wall_encoder(
            wall_tile_tokens.flatten(start_dim=1)
        )
        replacement_token = self.replacement_draw_encoder(
            privileged.replacement_draw_tiles,
            privileged.replacement_draw_valid,
            self.public_encoder.tile_embedding,
        )
        legal_token = self.legal_mask_encoder(
            batch.action_legal.to(dtype=dtype)
        )
        return self.value_trunk(torch.cat(
            (
                public_state,
                legal_token,
                player_tokens.flatten(start_dim=1),
                wall_token,
                replacement_token,
            ),
            dim=1,
        ))


def v4_privileged_critic_config_from_dict(
    values: Mapping[str, object],
) -> V4PrivilegedCriticConfig:
    config_values = dict(values)
    public_config = V4ActorConfig(**dict(config_values.pop("public_config")))
    return V4PrivilegedCriticConfig(
        public_config=public_config,
        **config_values,
    )


class V4FourSeatCritic(V4PrivilegedGuiyangCritic):
    """Experimental four-seat readouts, relative to the current observer."""
    def __init__(self, config, zero_sum=True):
        super().__init__(config)
        self.score_head = nn.Linear(self.value_feature_dim, 4)
        del self.shanten_head
        self.zero_sum = zero_sum

    def values_from_features(self, features):
        from types import SimpleNamespace
        raw = self.score_head(features)
        scores = raw - raw.mean(dim=1, keepdim=True) if self.zero_sum else raw
        return SimpleNamespace(score_values_all=scores, score_value=scores[:, 0],
                               shanten_value=torch.zeros_like(scores[:, 0]))


def save_v4_privileged_critic_checkpoint(
    path: Path | str,
    critic: V4PrivilegedGuiyangCritic,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "critic_architecture": V4_PRIVILEGED_CRITIC_ARCHITECTURE,
        "four_seat": isinstance(critic, V4FourSeatCritic),
        "zero_sum": getattr(critic, "zero_sum", False),
        "critic_tensor_version": V4_PRIVILEGED_TENSOR_VERSION,
        "critic_config": asdict(critic.config),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in critic.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }, destination)


def load_v4_privileged_critic_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[V4PrivilegedGuiyangCritic, Mapping[str, object]]:
    device = torch.device(device)
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("architecture") == "v4_four_scalar_projection_ab_v1":
        config = v4_privileged_critic_config_from_dict(checkpoint["config"])
        critic = V4FourSeatCritic(config, checkpoint["zero_sum"])
        critic.load_state_dict(checkpoint["model"], strict=True)
        return critic.to(device).eval(), checkpoint
    if (
        checkpoint.get("critic_architecture")
        != V4_PRIVILEGED_CRITIC_ARCHITECTURE
    ):
        raise ValueError("checkpoint is not a structured V4 privileged critic")
    if checkpoint.get("critic_tensor_version") != V4_PRIVILEGED_TENSOR_VERSION:
        raise ValueError("checkpoint uses an incompatible V4 Critic tensor format")
    config = v4_privileged_critic_config_from_dict(
        checkpoint["critic_config"]
    )
    critic = (V4FourSeatCritic(config, checkpoint["zero_sum"])
              if checkpoint.get("four_seat") else V4PrivilegedGuiyangCritic(config))
    critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    critic.to(device).eval()
    return critic, checkpoint


__all__ = [
    "ShortDrawSequenceEncoder",
    "V4_PRIVILEGED_CRITIC_ARCHITECTURE",
    "V4_PRIVILEGED_TENSOR_VERSION",
    "V4CriticPublicTensorBatch",
    "V4PrivilegedCriticConfig",
    "V4PrivilegedGuiyangCritic",
    "V4PrivilegedTensorBatch",
    "load_v4_privileged_critic_checkpoint",
    "save_v4_privileged_critic_checkpoint",
    "tensorize_v4_critic_public_states",
    "tensorize_v4_privileged_states",
    "v4_critic_public_from_actor_batch",
    "v4_privileged_critic_config_from_dict",
]
