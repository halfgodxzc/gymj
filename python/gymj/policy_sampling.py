import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as functional

from ._gymj import ACTION_SPACE_SIZE, TILE_TYPE_COUNT
from .behavior_cloning import BehaviorClonePolicy


ANKAN_ACTION_START = TILE_TYPE_COUNT
KAKAN_ACTION_START = TILE_TYPE_COUNT * 2
PON_ACTION_ID = TILE_TYPE_COUNT * 3
DAIMINKAN_ACTION_ID = PON_ACTION_ID + 1
TSUMO_ACTION_ID = DAIMINKAN_ACTION_ID + 1
RON_ACTION_ID = TSUMO_ACTION_ID + 1
KAN_ACTION_IDS = tuple(range(ANKAN_ACTION_START, PON_ACTION_ID)) + (
    DAIMINKAN_ACTION_ID,
)
WIN_ACTION_IDS = (TSUMO_ACTION_ID, RON_ACTION_ID)
DEFAULT_KAN_LOGIT_BIAS = -3.0


@dataclass(frozen=True)
class ActorSamplingConfig:
    kan_learning_enabled: bool = True
    kan_logit_bias: float = DEFAULT_KAN_LOGIT_BIAS
    temperature: float = 1.0
    force_kan_when_available: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.kan_logit_bias):
            raise ValueError("kan logit bias must be finite")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("sampling temperature must be positive and finite")


class PolicyEvaluation(NamedTuple):
    log_probability: torch.Tensor
    entropy: torch.Tensor


class SampledPolicyActions(NamedTuple):
    action_ids: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor


def initialize_kan_rows_for_rl(
    policy: BehaviorClonePolicy,
    bias: float = 0.0,
) -> None:
    """Reset V2's unsupervised Kan rows before the first PPO rollout."""
    if not math.isfinite(bias):
        raise ValueError("kan row bias must be finite")
    output_layer = policy.policy_head[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("policy head must end with a linear output layer")
    if output_layer.out_features != ACTION_SPACE_SIZE:
        raise ValueError("policy output layer does not match the action space")

    with torch.no_grad():
        output_layer.weight[list(KAN_ACTION_IDS)] = 0.0
        output_layer.bias[list(KAN_ACTION_IDS)] = bias


def masked_policy_logits(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    config: ActorSamplingConfig = ActorSamplingConfig(),
) -> torch.Tensor:
    _validate_policy_inputs(logits, legal_mask)
    selectable = legal_mask.bool()
    kan_mask = _kan_mask(logits.device)
    if not config.kan_learning_enabled:
        selectable = selectable & ~kan_mask.unsqueeze(0)
    if not torch.all(selectable.any(dim=1)):
        raise ValueError("policy row has no selectable actions")

    adjusted = logits
    if config.kan_learning_enabled and config.kan_logit_bias != 0.0:
        adjusted = adjusted + (
            kan_mask.to(dtype=logits.dtype).unsqueeze(0)
            * config.kan_logit_bias
        )
    if config.force_kan_when_available:
        win_mask = _action_mask(WIN_ACTION_IDS, logits.device)
        kan_available = (selectable & kan_mask.unsqueeze(0)).any(dim=1)
        win_available = (selectable & win_mask.unsqueeze(0)).any(dim=1)
        force_rows = kan_available & ~win_available
        selectable = torch.where(
            force_rows.unsqueeze(1),
            selectable & kan_mask.unsqueeze(0),
            selectable,
        )
    adjusted = adjusted / config.temperature
    return adjusted.masked_fill(~selectable, -torch.inf)


def evaluate_policy_actions(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    action_ids: torch.Tensor,
    config: ActorSamplingConfig = ActorSamplingConfig(),
) -> PolicyEvaluation:
    masked_logits = masked_policy_logits(logits, legal_mask, config)
    if action_ids.ndim != 1 or action_ids.shape[0] != logits.shape[0]:
        raise ValueError("action ids must have shape [batch]")
    action_ids = action_ids.to(device=logits.device, dtype=torch.long)
    if torch.any(action_ids < 0) or torch.any(action_ids >= ACTION_SPACE_SIZE):
        raise ValueError("action id is outside the action space")

    selected_logits = masked_logits.gather(1, action_ids.unsqueeze(1))
    if not torch.all(torch.isfinite(selected_logits)):
        raise ValueError("selected action is not selectable")

    log_probabilities = functional.log_softmax(masked_logits, dim=1)
    probabilities = log_probabilities.exp()
    finite_log_probabilities = torch.where(
        torch.isfinite(log_probabilities),
        log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    entropy = -(probabilities * finite_log_probabilities).sum(dim=1)
    return PolicyEvaluation(
        log_probability=log_probabilities.gather(
            1,
            action_ids.unsqueeze(1),
        ).squeeze(1),
        entropy=entropy,
    )


def sample_policy_actions(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    config: ActorSamplingConfig = ActorSamplingConfig(),
    generator: Optional[torch.Generator] = None,
) -> SampledPolicyActions:
    masked_logits = masked_policy_logits(logits, legal_mask, config)
    probabilities = functional.softmax(masked_logits, dim=1)
    action_ids = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(1)
    evaluation = evaluate_policy_actions(
        logits,
        legal_mask,
        action_ids,
        config,
    )
    return SampledPolicyActions(
        action_ids=action_ids,
        log_probability=evaluation.log_probability,
        entropy=evaluation.entropy,
    )


def _validate_policy_inputs(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> None:
    if logits.ndim != 2 or logits.shape[1] != ACTION_SPACE_SIZE:
        raise ValueError("logits must have shape [batch, actions]")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal mask shape must match logits")


def _kan_mask(device: torch.device) -> torch.Tensor:
    return _action_mask(KAN_ACTION_IDS, device)


def _action_mask(
    action_ids: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros(
        ACTION_SPACE_SIZE,
        dtype=torch.bool,
        device=device,
    )
    result[list(action_ids)] = True
    return result


__all__ = [
    "ActorSamplingConfig",
    "DEFAULT_KAN_LOGIT_BIAS",
    "KAN_ACTION_IDS",
    "RON_ACTION_ID",
    "TSUMO_ACTION_ID",
    "WIN_ACTION_IDS",
    "PolicyEvaluation",
    "SampledPolicyActions",
    "evaluate_policy_actions",
    "initialize_kan_rows_for_rl",
    "masked_policy_logits",
    "sample_policy_actions",
]
