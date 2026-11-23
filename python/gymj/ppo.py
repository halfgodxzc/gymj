from typing import NamedTuple

import torch


class PPOObjective(NamedTuple):
    loss: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def combine_returns(
    score_return: torch.Tensor,
    shanten_return: torch.Tensor,
    score_weight: float,
    shanten_weight: float,
) -> torch.Tensor:
    if score_return.shape != shanten_return.shape:
        raise ValueError("return tensors must have the same shape")
    if score_weight < 0.0 or shanten_weight < 0.0:
        raise ValueError("return weights must be non-negative")
    return score_weight * score_return + shanten_weight * shanten_return


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.ndim != 1 or len(advantages) == 0:
        raise ValueError("advantages must be a non-empty vector")
    return (
        (advantages - advantages.mean())
        / advantages.std(unbiased=False).clamp(min=1e-8)
    )


def clipped_policy_objective(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    clip_range: float = 0.2,
    entropy_coefficient: float = 0.01,
) -> PPOObjective:
    shapes = {
        tuple(new_log_probability.shape),
        tuple(old_log_probability.shape),
        tuple(advantages.shape),
        tuple(entropy.shape),
    }
    if len(shapes) != 1 or new_log_probability.ndim != 1:
        raise ValueError("PPO objective inputs must be equal-sized vectors")
    if not 0.0 < clip_range < 1.0:
        raise ValueError("clip range must be between zero and one")
    if entropy_coefficient < 0.0:
        raise ValueError("entropy coefficient must be non-negative")

    log_ratio = new_log_probability - old_log_probability
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    mean_entropy = entropy.mean()
    return PPOObjective(
        loss=policy_loss - entropy_coefficient * mean_entropy,
        entropy=mean_entropy,
        approximate_kl=((ratio - 1.0) - log_ratio).mean(),
        clip_fraction=((ratio - 1.0).abs() > clip_range).float().mean(),
    )


__all__ = [
    "PPOObjective",
    "clipped_policy_objective",
    "combine_returns",
    "normalize_advantages",
]
