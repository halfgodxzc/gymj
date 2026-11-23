from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, NamedTuple

import torch
from torch import nn

from ._gymj import ACTION_SPACE_SIZE
from .behavior_cloning import (
    BehaviorClonePolicy,
    PolicyConfig,
    policy_config_from_dict,
)


@dataclass(frozen=True)
class CriticConfig:
    observation_config: PolicyConfig = field(default_factory=PolicyConfig)
    legal_mask_hidden_dim: int = 32
    hidden_dim: int = 256


class CriticValues(NamedTuple):
    score_value: torch.Tensor
    shanten_value: torch.Tensor

    def combined(
        self,
        score_weight: float,
        shanten_weight: float,
    ) -> torch.Tensor:
        return (
            score_weight * self.score_value
            + shanten_weight * self.shanten_value
        )


class GuiyangCritic(nn.Module):
    """Estimate future score and shanten returns for one player."""

    def __init__(self, config: CriticConfig = CriticConfig()):
        super().__init__()
        if config.legal_mask_hidden_dim <= 0:
            raise ValueError("legal mask hidden dimension must be positive")
        if config.hidden_dim <= 0:
            raise ValueError("critic hidden dimension must be positive")

        self.config = config
        self.observation_encoder = BehaviorClonePolicy(
            config.observation_config
        )
        self.observation_encoder.policy_head = nn.Identity()
        self.observation_encoder.profile_head = None

        self.legal_mask_encoder = nn.Sequential(
            nn.Linear(ACTION_SPACE_SIZE, config.legal_mask_hidden_dim),
            nn.GELU(),
        )
        value_input_dim = (
            self.observation_encoder.encoder_dim
            + self.observation_encoder.profile_dim
            + config.legal_mask_hidden_dim
        )
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(value_input_dim),
            nn.Linear(value_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(config.hidden_dim, 1)
        self.shanten_head = nn.Linear(config.hidden_dim, 1)
        self._initialize_value_heads()

    def _initialize_value_heads(self) -> None:
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        nn.init.zeros_(self.shanten_head.weight)
        nn.init.zeros_(self.shanten_head.bias)

    def initialize_observation_encoder(
        self,
        policy: BehaviorClonePolicy,
    ) -> None:
        """Copy policy features without copying either action head."""
        encoder_state = {
            name: value
            for name, value in policy.state_dict().items()
            if not name.startswith((
                "policy_head.",
                "profile_head.",
                "action_feature_head.",
            ))
        }
        self.observation_encoder.load_state_dict(encoder_state, strict=True)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        legal_mask: torch.Tensor,
    ) -> CriticValues:
        observation_features = self.observation_encoder.encode_policy_features(
            batch
        )
        if legal_mask.ndim != 2:
            raise ValueError("legal mask must have shape [batch, actions]")
        if legal_mask.shape != (
            observation_features.shape[0],
            ACTION_SPACE_SIZE,
        ):
            raise ValueError(
                "legal mask shape must match the observation batch and "
                "action space"
            )

        legal_features = self.legal_mask_encoder(
            legal_mask.to(
                device=observation_features.device,
                dtype=observation_features.dtype,
            )
        )
        features = self.value_trunk(
            torch.cat((observation_features, legal_features), dim=1)
        )
        return CriticValues(
            score_value=self.score_head(features).squeeze(-1),
            shanten_value=self.shanten_head(features).squeeze(-1),
        )


def load_critic_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[GuiyangCritic, Mapping[str, object]]:
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    values = dict(checkpoint["critic_config"])
    values["observation_config"] = policy_config_from_dict(
        values["observation_config"]
    )
    critic = GuiyangCritic(CriticConfig(**values)).to(device)
    critic.load_state_dict(checkpoint["model_state_dict"])
    critic.eval()
    return critic, checkpoint


__all__ = [
    "CriticConfig",
    "CriticValues",
    "GuiyangCritic",
    "load_critic_checkpoint",
]
