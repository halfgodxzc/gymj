from dataclasses import dataclass, replace

import torch
from torch import nn

from ._gymj import ACTION_SPACE_SIZE, MAX_DISCARD_EVENTS, TILE_TYPE_COUNT
from .kan_adapter import KAN_ADAPTER_FEATURE_DIM
from .v4_features import (
    V4_ANKAN_TYPE,
    V4_ACTION_FAMILY_COUNT,
    V4_ACTION_SCALAR_DIM,
    V4_ACTION_TYPE_COUNT,
    V4_DAIMINKAN_TYPE,
    V4_DISCARD_TYPE,
    V4_DRAW_KIND_COUNT,
    V4_CONTEXT_DIM,
    V4_HAND_SUMMARY_DIM,
    V4_HAND_TILE_DIM,
    V4_KAKAN_TYPE,
    V4_MELD_TYPE_COUNT,
    V4_PHASE_COUNT,
    V4_PLAYER_COUNT,
    V4_PLAYER_PAD,
    V4_PLAYER_SUMMARY_DIM,
    V4_PLAYER_TILE_DIM,
    V4_PON_TYPE,
    V4_PROGRESS_DIM,
    V4_RON_TYPE,
    V4_TILE_PAD,
    V4_TSUMO_TYPE,
    V4TensorBatch,
)


V4_BASE_ACTOR_ARCHITECTURE_VERSION = 1
V4_KAN_ADAPTER_ARCHITECTURE_VERSION = 2
V4_ACTOR_ARCHITECTURE_VERSION = V4_KAN_ADAPTER_ARCHITECTURE_VERSION


@dataclass(frozen=True)
class V4ActorConfig:
    tile_embedding_dim: int = 16
    player_embedding_dim: int = 8
    action_type_embedding_dim: int = 16
    draw_embedding_dim: int = 8
    phase_embedding_dim: int = 8
    ready_embedding_dim: int = 8
    meld_type_embedding_dim: int = 8
    discard_hidden_dim: int = 48
    player_discard_hidden_dim: int = 32
    meld_event_hidden_dim: int = 32
    hand_tile_hidden_dim: int = 64
    hand_summary_hidden_dim: int = 64
    hand_output_dim: int = 128
    player_tile_hidden_dim: int = 32
    player_summary_hidden_dim: int = 32
    player_output_dim: int = 64
    progress_hidden_dim: int = 64
    state_hidden_dim: int = 256
    action_scalar_hidden_dim: int = 64
    action_hidden_dim: int = 128
    family_hidden_dim: int = 128
    kan_adapter_enabled: bool = False
    kan_adapter_feature_hidden_dim: int = 32
    kan_adapter_hidden_dim: int = 128
    kan_adapter_initial_delta: float = -4.0
    dropout: float = 0.1


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class HandPatternEncoder(nn.Module):
    def __init__(self, config: V4ActorConfig):
        super().__init__()
        tile_input = V4_HAND_TILE_DIM + config.tile_embedding_dim
        self.tile_encoder = _mlp(
            tile_input,
            config.hand_tile_hidden_dim,
            config.hand_tile_hidden_dim,
            config.dropout,
        )
        self.summary_encoder = _mlp(
            V4_HAND_SUMMARY_DIM,
            config.hand_summary_hidden_dim,
            config.hand_summary_hidden_dim,
            config.dropout,
        )
        combined_dim = (
            2 * config.hand_tile_hidden_dim
            + config.hand_summary_hidden_dim
        )
        self.output = nn.Sequential(
            nn.Linear(combined_dim, config.hand_output_dim),
            nn.GELU(),
            nn.LayerNorm(config.hand_output_dim),
        )

    def forward(
        self,
        summary: torch.Tensor,
        tiles: torch.Tensor,
        tile_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        prefix = tiles.shape[:-2]
        expected_summary = (*prefix, V4_HAND_SUMMARY_DIM)
        expected_tiles = (*prefix, TILE_TYPE_COUNT, V4_HAND_TILE_DIM)
        if tuple(summary.shape) != expected_summary:
            raise ValueError(f"hand summary must have shape {expected_summary}")
        if tuple(tiles.shape) != expected_tiles:
            raise ValueError(f"hand tiles must have shape {expected_tiles}")
        embedding_shape = (
            *((1,) * len(prefix)),
            TILE_TYPE_COUNT,
            tile_embeddings.shape[-1],
        )
        expanded_embeddings = tile_embeddings.view(embedding_shape).expand(
            *prefix,
            TILE_TYPE_COUNT,
            tile_embeddings.shape[-1],
        )
        tile_tokens = self.tile_encoder(
            torch.cat((tiles, expanded_embeddings), dim=-1)
        )
        pooled = torch.cat(
            (tile_tokens.mean(dim=-2), tile_tokens.amax(dim=-2)),
            dim=-1,
        )
        summary_token = self.summary_encoder(summary)
        return self.output(torch.cat((pooled, summary_token), dim=-1))


class PublicPlayerEncoder(nn.Module):
    def __init__(self, config: V4ActorConfig):
        super().__init__()
        self.tile_encoder = _mlp(
            V4_PLAYER_TILE_DIM + config.tile_embedding_dim,
            config.player_tile_hidden_dim,
            config.player_tile_hidden_dim,
            config.dropout,
        )
        self.summary_encoder = _mlp(
            V4_PLAYER_SUMMARY_DIM,
            config.player_summary_hidden_dim,
            config.player_summary_hidden_dim,
            config.dropout,
        )
        combined_dim = (
            2 * config.player_tile_hidden_dim
            + config.player_summary_hidden_dim
            + config.player_embedding_dim
            + config.player_discard_hidden_dim
            + config.meld_event_hidden_dim
            + config.ready_embedding_dim
        )
        self.output = nn.Sequential(
            nn.Linear(combined_dim, config.player_output_dim),
            nn.GELU(),
            nn.LayerNorm(config.player_output_dim),
        )

    def forward(
        self,
        tiles: torch.Tensor,
        summary: torch.Tensor,
        tile_embeddings: torch.Tensor,
        player_embeddings: torch.Tensor,
        discard_profiles: torch.Tensor,
        meld_profiles: torch.Tensor,
        ready_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch = tiles.shape[0]
        expected_tiles = (
            batch,
            V4_PLAYER_COUNT,
            TILE_TYPE_COUNT,
            V4_PLAYER_TILE_DIM,
        )
        expected_summary = (
            batch,
            V4_PLAYER_COUNT,
            V4_PLAYER_SUMMARY_DIM,
        )
        if tuple(tiles.shape) != expected_tiles:
            raise ValueError(f"player tiles must have shape {expected_tiles}")
        if tuple(summary.shape) != expected_summary:
            raise ValueError(f"player summary must have shape {expected_summary}")
        tile_ids = tile_embeddings.view(
            1,
            1,
            TILE_TYPE_COUNT,
            -1,
        ).expand(batch, V4_PLAYER_COUNT, TILE_TYPE_COUNT, -1)
        tile_tokens = self.tile_encoder(torch.cat((tiles, tile_ids), dim=-1))
        pooled = torch.cat(
            (tile_tokens.mean(dim=2), tile_tokens.amax(dim=2)),
            dim=-1,
        )
        summary_tokens = self.summary_encoder(summary)
        relative_players = player_embeddings.view(
            1,
            V4_PLAYER_COUNT,
            -1,
        ).expand(batch, V4_PLAYER_COUNT, -1)
        return self.output(
            torch.cat(
                (
                    pooled,
                    summary_tokens,
                    relative_players,
                    discard_profiles,
                    meld_profiles,
                    ready_embeddings,
                ),
                dim=-1,
            )
        )


class DiscardHistoryEncoder(nn.Module):
    def __init__(self, config: V4ActorConfig):
        super().__init__()
        input_dim = (
            config.tile_embedding_dim
            + config.player_embedding_dim
            + 2
        )
        self.global_encoder = nn.GRU(
            input_dim,
            config.discard_hidden_dim,
            batch_first=True,
        )
        self.player_encoder = nn.GRU(
            input_dim,
            config.player_discard_hidden_dim,
            batch_first=True,
        )

    @staticmethod
    def _last_valid(outputs: torch.Tensor, lengths: torch.Tensor):
        indices = (lengths - 1).clamp_min(0)
        gathered = torch.gather(
            outputs,
            1,
            indices.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1]),
        ).squeeze(1)
        return gathered * (lengths > 0).unsqueeze(-1).to(outputs.dtype)

    def forward(
        self,
        batch: V4TensorBatch,
        tile_embedding: nn.Embedding,
        player_embedding: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            MAX_DISCARD_EVENTS,
            device=batch.discard_tiles.device,
            dtype=batch.discard_called.dtype,
        ).view(1, MAX_DISCARD_EVENTS, 1)
        positions = positions / max(MAX_DISCARD_EVENTS - 1, 1)
        inputs = torch.cat(
            (
                tile_embedding(batch.discard_tiles),
                player_embedding(batch.discard_players),
                batch.discard_called.unsqueeze(-1),
                positions.expand(batch.batch_size, -1, -1),
            ),
            dim=-1,
        )
        inputs = inputs * batch.discard_valid.unsqueeze(-1).to(inputs.dtype)
        global_outputs, _ = self.global_encoder(inputs)
        global_lengths = batch.discard_valid.sum(dim=1)
        global_summary = self._last_valid(global_outputs, global_lengths)

        player_ids = torch.arange(
            V4_PLAYER_COUNT,
            device=batch.discard_players.device,
        ).view(1, V4_PLAYER_COUNT, 1)
        grouped_valid = (
            batch.discard_valid.unsqueeze(1)
            & (batch.discard_players.unsqueeze(1) == player_ids)
        )
        original_positions = torch.arange(
            MAX_DISCARD_EVENTS,
            device=batch.discard_players.device,
        ).view(1, 1, MAX_DISCARD_EVENTS)
        sort_keys = torch.where(
            grouped_valid,
            original_positions,
            original_positions + MAX_DISCARD_EVENTS,
        )
        order = sort_keys.argsort(dim=2)
        grouped = inputs.unsqueeze(1).expand(-1, V4_PLAYER_COUNT, -1, -1)
        grouped = torch.gather(
            grouped,
            2,
            order.unsqueeze(-1).expand(-1, -1, -1, inputs.shape[-1]),
        )
        grouped = grouped.reshape(
            batch.batch_size * V4_PLAYER_COUNT,
            MAX_DISCARD_EVENTS,
            inputs.shape[-1],
        )
        player_outputs, _ = self.player_encoder(grouped)
        player_lengths = grouped_valid.sum(dim=2).reshape(-1)
        player_summaries = self._last_valid(
            player_outputs,
            player_lengths,
        ).reshape(batch.batch_size, V4_PLAYER_COUNT, -1)
        return global_summary, player_summaries


class MeldHistoryEncoder(nn.Module):
    def __init__(self, config: V4ActorConfig):
        super().__init__()
        input_dim = (
            config.tile_embedding_dim
            + 2 * config.player_embedding_dim
            + config.meld_type_embedding_dim
        )
        self.event_encoder = _mlp(
            input_dim,
            config.meld_event_hidden_dim,
            config.meld_event_hidden_dim,
            config.dropout,
        )

    def forward(
        self,
        batch: V4TensorBatch,
        tile_embedding: nn.Embedding,
        player_embedding: nn.Embedding,
        meld_type_embedding: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events = self.event_encoder(torch.cat(
            (
                tile_embedding(batch.meld_tiles),
                player_embedding(batch.meld_owners),
                player_embedding(batch.meld_sources),
                meld_type_embedding(batch.meld_types),
            ),
            dim=-1,
        ))
        valid = batch.meld_valid.unsqueeze(-1).to(events.dtype)
        global_summary = (events * valid).sum(dim=1) / valid.sum(
            dim=1
        ).clamp_min(1.0)
        owner_ids = torch.arange(
            V4_PLAYER_COUNT,
            device=batch.meld_owners.device,
        ).view(1, V4_PLAYER_COUNT, 1)
        owner_mask = (
            batch.meld_valid.unsqueeze(1)
            & (batch.meld_owners.unsqueeze(1) == owner_ids)
        ).unsqueeze(-1).to(events.dtype)
        profiles = (
            events.unsqueeze(1) * owner_mask
        ).sum(dim=2) / owner_mask.sum(dim=2).clamp_min(1.0)
        return global_summary, profiles


class MaskedAttentionPool(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(mask.shape) != tuple(values.shape[:-1]):
            raise ValueError("attention mask does not match values")
        scores = self.score(values).squeeze(-1)
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=-1) * mask.to(values.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.sum(values * weights.unsqueeze(-1), dim=-2)


class ActionFamilyScorer(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * action_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        state = state.unsqueeze(1).expand_as(action)
        interaction = torch.cat(
            (state, action, state * action, torch.abs(state - action)),
            dim=-1,
        )
        return self.network(interaction).squeeze(-1)


class KanRelativeAdapter(nn.Module):
    """Score each kan as a delta from the best legal non-kan action."""

    def __init__(self, action_dim: int, config: V4ActorConfig):
        super().__init__()
        if config.kan_adapter_feature_hidden_dim <= 0:
            raise ValueError("kan adapter feature dimension must be positive")
        if config.kan_adapter_hidden_dim < 2:
            raise ValueError("kan adapter hidden dimension must be at least two")
        self.feature_encoder = _mlp(
            KAN_ADAPTER_FEATURE_DIM,
            config.kan_adapter_feature_hidden_dim,
            config.kan_adapter_feature_hidden_dim,
            config.dropout,
        )
        input_dim = (
            4 * action_dim
            + config.kan_adapter_feature_hidden_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.kan_adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.kan_adapter_hidden_dim,
                config.kan_adapter_hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(config.kan_adapter_hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(
            self.network[-1].bias,
            config.kan_adapter_initial_delta,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        state = state.unsqueeze(1).expand_as(action)
        interaction = torch.cat(
            (
                state,
                action,
                state * action,
                torch.abs(state - action),
                self.feature_encoder(features.to(dtype=state.dtype)),
            ),
            dim=-1,
        )
        return self.network(interaction).squeeze(-1)


def v4_actor_architecture_version(config: V4ActorConfig) -> int:
    return (
        V4_KAN_ADAPTER_ARCHITECTURE_VERSION
        if config.kan_adapter_enabled
        else V4_BASE_ACTOR_ARCHITECTURE_VERSION
    )


class V4PublicStateEncoder(nn.Module):
    """Encode the public V4 state without scoring candidate actions."""

    def __init__(self, config: V4ActorConfig = V4ActorConfig()):
        super().__init__()
        self.config = config
        self.tile_embedding = nn.Embedding(
            V4_TILE_PAD + 1,
            config.tile_embedding_dim,
            padding_idx=V4_TILE_PAD,
        )
        self.player_embedding = nn.Embedding(
            V4_PLAYER_PAD + 1,
            config.player_embedding_dim,
            padding_idx=V4_PLAYER_PAD,
        )
        self.draw_kind_embedding = nn.Embedding(
            V4_DRAW_KIND_COUNT,
            config.draw_embedding_dim,
        )
        self.phase_embedding = nn.Embedding(
            V4_PHASE_COUNT,
            config.phase_embedding_dim,
        )
        self.ready_embedding = nn.Embedding(3, config.ready_embedding_dim)
        self.meld_type_embedding = nn.Embedding(
            V4_MELD_TYPE_COUNT,
            config.meld_type_embedding_dim,
        )

        self.hand_encoder = HandPatternEncoder(config)
        self.discard_history_encoder = DiscardHistoryEncoder(config)
        self.meld_history_encoder = MeldHistoryEncoder(config)
        self.player_encoder = PublicPlayerEncoder(config)
        progress_input = (
            V4_PROGRESS_DIM
            + V4_CONTEXT_DIM
            + config.draw_embedding_dim
            + config.phase_embedding_dim
            + config.tile_embedding_dim
            + 4 * config.player_embedding_dim
            + V4_PLAYER_COUNT * config.ready_embedding_dim
            + 2 * config.tile_embedding_dim
            + config.discard_hidden_dim
            + config.meld_event_hidden_dim
        )
        self.progress_encoder = _mlp(
            progress_input,
            config.progress_hidden_dim,
            config.progress_hidden_dim,
            config.dropout,
        )
        state_input = (
            config.hand_output_dim
            + V4_PLAYER_COUNT * config.player_output_dim
            + config.progress_hidden_dim
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden_dim),
        )

    def _tile_table(self) -> torch.Tensor:
        tile_ids = torch.arange(
            TILE_TYPE_COUNT,
            device=self.tile_embedding.weight.device,
        )
        return self.tile_embedding(tile_ids)

    def _player_table(self) -> torch.Tensor:
        player_ids = torch.arange(
            V4_PLAYER_COUNT,
            device=self.player_embedding.weight.device,
        )
        return self.player_embedding(player_ids)

    def encode_state(
        self,
        batch: V4TensorBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tile_table = self._tile_table()
        player_table = self._player_table()
        hand = self.hand_encoder(
            batch.hand_summary,
            batch.hand_tiles,
            tile_table,
        )
        global_discards, player_discards = self.discard_history_encoder(
            batch,
            self.tile_embedding,
            self.player_embedding,
        )
        global_melds, player_melds = self.meld_history_encoder(
            batch,
            self.tile_embedding,
            self.player_embedding,
            self.meld_type_embedding,
        )
        ready_embeddings = self.ready_embedding(batch.ready_kinds)
        players = self.player_encoder(
            batch.player_tiles,
            batch.player_summary,
            tile_table,
            player_table,
            player_discards,
            player_melds,
            ready_embeddings,
        )
        current_waits = self._tile_set_summary(
            batch.current_waits,
            tile_table,
        )
        locked_waits = self._tile_set_summary(
            batch.locked_ready_waits,
            tile_table,
        )
        progress = self.progress_encoder(torch.cat(
            (
                batch.progress,
                batch.context,
                self.draw_kind_embedding(batch.last_draw_kind),
                self.phase_embedding(batch.phase),
                self.player_embedding(batch.current_actor),
                self.tile_embedding(batch.current_discard),
                self.player_embedding(batch.current_discard_actor),
                self.player_embedding(batch.dealer),
                ready_embeddings.flatten(start_dim=1),
                current_waits,
                locked_waits,
                self.player_embedding(batch.hot_discard_player),
                global_discards,
                global_melds,
            ),
            dim=-1,
        ))
        state = self.state_encoder(torch.cat(
            (hand, players.flatten(start_dim=1), progress),
            dim=-1,
        ))
        return state, players

    @staticmethod
    def _tile_set_summary(
        membership: torch.Tensor,
        tile_table: torch.Tensor,
    ) -> torch.Tensor:
        counts = membership.sum(dim=-1, keepdim=True)
        weights = membership / counts.clamp_min(1.0)
        return torch.matmul(weights, tile_table)

    def forward(self, batch: V4TensorBatch) -> torch.Tensor:
        state, _ = self.encode_state(batch)
        return state


class V4Actor(V4PublicStateEncoder):
    """Shared semantic action scorer without a free 87-action embedding."""

    def __init__(self, config: V4ActorConfig = V4ActorConfig()):
        super().__init__(config)
        if config.family_hidden_dim < 2:
            raise ValueError("family hidden dimension must be at least two")
        self.action_type_embedding = nn.Embedding(
            V4_ACTION_TYPE_COUNT,
            config.action_type_embedding_dim,
            padding_idx=V4_ACTION_TYPE_COUNT - 1,
        )
        self.action_scalar_encoder = _mlp(
            V4_ACTION_SCALAR_DIM,
            config.action_scalar_hidden_dim,
            config.action_scalar_hidden_dim,
            config.dropout,
        )
        semantic_input = (
            config.action_type_embedding_dim
            + config.tile_embedding_dim
            + 2 * config.player_embedding_dim
            + 2 * config.player_output_dim
            + config.action_scalar_hidden_dim
            + config.tile_embedding_dim
            + 2 * config.hand_output_dim
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(semantic_input, config.action_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.action_hidden_dim, config.action_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.action_hidden_dim),
        )
        self.state_for_action = nn.Sequential(
            nn.Linear(config.state_hidden_dim, config.action_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.action_hidden_dim),
        )
        self.pon_follow_up_pool = MaskedAttentionPool(config.hand_output_dim)
        self.family_scorers = nn.ModuleList([
            ActionFamilyScorer(
                config.action_hidden_dim,
                config.family_hidden_dim,
                config.dropout,
            )
            for _ in range(V4_ACTION_FAMILY_COUNT - 1)
        ])
        self.kan_adapter = (
            KanRelativeAdapter(config.action_hidden_dim, config)
            if config.kan_adapter_enabled
            else None
        )

    @staticmethod
    def _gather_player_tokens(
        players: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        padding = torch.zeros(
            players.shape[0],
            1,
            players.shape[-1],
            dtype=players.dtype,
            device=players.device,
        )
        padded = torch.cat((players, padding), dim=1)
        return torch.gather(
            padded,
            1,
            indices.unsqueeze(-1).expand(-1, -1, players.shape[-1]),
        )

    def encode_actions(
        self,
        batch: V4TensorBatch,
        players: torch.Tensor,
    ) -> torch.Tensor:
        tile_table = self._tile_table()
        discard_hands = self.hand_encoder(
            batch.discard_post_hand_summary,
            batch.discard_post_hand_tiles,
            tile_table,
        )
        safe_tiles = batch.action_tile.clamp(max=TILE_TYPE_COUNT - 1)
        gathered_discard_hands = torch.gather(
            discard_hands,
            1,
            safe_tiles.unsqueeze(-1).expand(
                -1,
                -1,
                discard_hands.shape[-1],
            ),
        )
        gathered_discard_hands = gathered_discard_hands * (
            batch.action_type == V4_DISCARD_TYPE
        ).unsqueeze(-1).to(gathered_discard_hands.dtype)

        pon_hands = self.hand_encoder(
            batch.pon_follow_up_summary,
            batch.pon_follow_up_tiles,
            tile_table,
        )
        pon_follow_up = self.pon_follow_up_pool(
            pon_hands,
            batch.pon_follow_up_mask,
        )
        pon_follow_up = pon_follow_up.unsqueeze(1) * (
            batch.action_type == V4_PON_TYPE
        ).unsqueeze(-1).to(pon_follow_up.dtype)

        wait_counts = batch.action_ready_waits.sum(dim=-1, keepdim=True)
        wait_weights = batch.action_ready_waits / wait_counts.clamp_min(1.0)
        ready_waits = torch.matmul(wait_weights, tile_table)

        source_tokens = self._gather_player_tokens(
            players,
            batch.action_source,
        )
        target_tokens = self._gather_player_tokens(
            players,
            batch.action_target,
        )
        semantic = torch.cat(
            (
                self.action_type_embedding(batch.action_type),
                self.tile_embedding(batch.action_tile),
                self.player_embedding(batch.action_source),
                self.player_embedding(batch.action_target),
                source_tokens,
                target_tokens,
                self.action_scalar_encoder(batch.action_scalar),
                ready_waits,
                gathered_discard_hands,
                pon_follow_up,
            ),
            dim=-1,
        )
        return self.action_encoder(semantic)

    def forward(self, batch: V4TensorBatch) -> torch.Tensor:
        state, players = self.encode_state(batch)
        actions = self.encode_actions(batch, players)
        action_state = self.state_for_action(state)
        family_scores = torch.stack(
            [scorer(action_state, actions) for scorer in self.family_scorers],
            dim=-1,
        )
        family_index = batch.action_family.clamp(
            min=0,
            max=len(self.family_scorers) - 1,
        )
        logits = torch.gather(
            family_scores,
            -1,
            family_index.unsqueeze(-1),
        ).squeeze(-1)
        base_valid = (
            batch.selectable_actions
            & (batch.action_family < len(self.family_scorers))
        )
        logits = logits.masked_fill(~base_valid, -torch.inf)
        if self.kan_adapter is None:
            return logits
        if batch.kan_adapter_features is None:
            raise ValueError("kan-enabled V4 actor requires kan adapter features")

        kan_type = (
            (batch.action_type == V4_ANKAN_TYPE)
            | (batch.action_type == V4_KAKAN_TYPE)
            | (batch.action_type == V4_DAIMINKAN_TYPE)
        )
        legal_hu = batch.action_legal & (
            (batch.action_type == V4_TSUMO_TYPE)
            | (batch.action_type == V4_RON_TYPE)
        )
        known_kan = batch.kan_adapter_features[..., 0] > 0.5
        kan_valid = (
            batch.action_legal
            & kan_type
            & known_kan
            & ~legal_hu.any(dim=1, keepdim=True)
        )
        if not kan_valid.any():
            return logits

        non_kan_valid = base_valid & ~kan_type
        rows_without_reference = kan_valid.any(dim=1) & ~non_kan_valid.any(dim=1)
        if rows_without_reference.any():
            raise ValueError("kan adapter requires a legal non-kan reference")
        reference = logits.masked_fill(~non_kan_valid, -torch.inf).amax(
            dim=1,
            keepdim=True,
        )
        delta = self.kan_adapter(
            action_state,
            actions,
            batch.kan_adapter_features,
        )
        kan_logits = reference.detach() + delta
        return torch.where(kan_valid, kan_logits, logits)

    def base_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("kan_adapter."):
                yield parameter

    def kan_adapter_parameters(self):
        if self.kan_adapter is not None:
            yield from self.kan_adapter.parameters()

    def set_base_trainable(self, trainable: bool) -> None:
        for parameter in self.base_parameters():
            parameter.requires_grad_(trainable)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def upgrade_v4_actor_with_kan_adapter(
    actor: V4Actor,
    *,
    feature_hidden_dim: int = 32,
    hidden_dim: int = 128,
    initial_delta: float = -4.0,
) -> V4Actor:
    if actor.config.kan_adapter_enabled:
        raise ValueError("V4 actor already has a kan adapter")
    config = replace(
        actor.config,
        kan_adapter_enabled=True,
        kan_adapter_feature_hidden_dim=feature_hidden_dim,
        kan_adapter_hidden_dim=hidden_dim,
        kan_adapter_initial_delta=initial_delta,
    )
    reference = next(actor.parameters())
    upgraded = V4Actor(config).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    upgraded_state = upgraded.state_dict()
    for name, value in actor.state_dict().items():
        if name not in upgraded_state or upgraded_state[name].shape != value.shape:
            raise ValueError(f"cannot preserve V4 actor parameter {name}")
        upgraded_state[name] = value.detach().clone()
    upgraded.load_state_dict(upgraded_state, strict=True)
    upgraded.train(actor.training)
    return upgraded
    V4_KAKAN_TYPE,
    V4_RON_TYPE,
    V4_TSUMO_TYPE,
