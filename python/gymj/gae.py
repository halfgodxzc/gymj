"""GAE for complete per-player trajectories with undiscounted MC labels."""
from __future__ import annotations

import math
import numpy as np
import torch


@torch.no_grad()
def complete_episode_gae(returns, values, episode_ids, player_ids,
                         decision_indices, *, gae_lambda=0.95):
    """Return detached advantages and lambda targets in original row order.

    Gamma is one. MC suffix differences recover rewards between consecutive
    decisions of the SAME player. Callers must supply complete, unfiltered
    player trajectories; terminal bootstrap is zero. Supports multiple heads.
    """
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError('gae_lambda must be finite and in [0, 1]')
    returns = torch.as_tensor(returns).detach().cpu()
    values = torch.as_tensor(values).detach().cpu()
    if returns.shape != values.shape or returns.ndim not in (1, 2) or not len(returns):
        raise ValueError('returns and values must have matching nonempty [N] or [N,H] shapes')
    if not returns.is_floating_point() or not values.is_floating_point():
        raise ValueError('returns and values must be floating point')
    if not torch.isfinite(returns).all() or not torch.isfinite(values).all():
        raise ValueError('nonfinite returns or values')
    keys = [np.asarray(torch.as_tensor(k).cpu()) for k in
            (episode_ids, player_ids, decision_indices)]
    if any(k.shape != (len(returns),) or k.dtype.kind not in 'iu' for k in keys):
        raise ValueError('trajectory keys must be matching integer vectors')
    episodes, players, steps = keys
    if np.any((players < 0) | (players > 3)) or np.any(steps < 0):
        raise ValueError('invalid player or decision index')
    # A global decision belongs to exactly one player, even in opponent pools.
    if len(set(zip(episodes.tolist(), steps.tolist()))) != len(returns):
        raise ValueError('duplicate episode/decision key')
    order = np.lexsort((steps, players, episodes))
    advantages = torch.empty_like(values)
    carry = torch.zeros_like(values[0])
    for position in range(len(order)-1, -1, -1):
        i = int(order[position])
        j = int(order[position+1]) if position+1 < len(order) else None
        terminal = j is None or episodes[i] != episodes[j] or players[i] != players[j]
        if terminal:
            carry = returns[i] - values[i]
        else:
            reward = returns[i] - returns[j]
            delta = reward + values[j] - values[i]
            carry = delta + gae_lambda * carry
        advantages[i] = carry
    return advantages, advantages + values
