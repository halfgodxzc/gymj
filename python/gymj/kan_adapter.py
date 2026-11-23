from typing import Sequence

import numpy as np
import torch

from ._gymj import ACTION_SPACE_SIZE, ActionType
from .kan_teacher import KanHeuristicEvaluator


KAN_ADAPTER_FEATURE_VERSION = 1
KAN_ADAPTER_FEATURE_NAMES = (
    "known",
    "baseline_shanten_progress",
    "expected_post_kan_shanten_progress",
    "improve_probability",
    "same_probability",
    "worse_probability",
    "expected_shanten_delta",
)
KAN_ADAPTER_FEATURE_DIM = len(KAN_ADAPTER_FEATURE_NAMES)


def _shanten_progress(value: float) -> float:
    if value < -1.0 or value > 8.0:
        raise ValueError(f"shanten must be in range -1..8, got {value}")
    return (8.0 - value) / 9.0


def tensorize_kan_adapter_features(
    decisions: Sequence,
    device: torch.device | str | None = None,
    *,
    evaluator: KanHeuristicEvaluator | None = None,
) -> torch.Tensor:
    """Build visible-information-only features for legal kan actions."""
    if not decisions:
        raise ValueError("kan adapter feature batch cannot be empty")
    evaluator = evaluator or KanHeuristicEvaluator()
    values = np.zeros(
        (len(decisions), ACTION_SPACE_SIZE, KAN_ADAPTER_FEATURE_DIM),
        dtype=np.float32,
    )

    for row, decision in enumerate(decisions):
        if len(decision.actions) != ACTION_SPACE_SIZE:
            raise ValueError("decision must contain the complete action space")
        has_legal_kan = any(
            candidate.legal
            and candidate.semantics.type in (
                ActionType.ANKAN,
                ActionType.KAKAN,
                ActionType.DAIMINKAN,
            )
            for candidate in decision.actions
        )
        if not has_legal_kan:
            continue
        legal_mask = tuple(
            1 if candidate.legal else 0
            for candidate in decision.actions
        )
        _, evaluations = evaluator.evaluate_best_shanten_baseline(
            decision.observation,
            legal_mask,
        )
        for evaluation in evaluations:
            probabilities = (
                evaluation.improve_probability
                + evaluation.same_probability
                + evaluation.worse_probability
            )
            if abs(probabilities - 1.0) > 1.0e-5:
                raise ValueError("kan outcome probabilities must sum to one")
            values[row, evaluation.action_id] = (
                1.0,
                _shanten_progress(float(evaluation.baseline_shanten)),
                _shanten_progress(
                    float(evaluation.expected_shanten_after_kan)
                ),
                float(evaluation.improve_probability),
                float(evaluation.same_probability),
                float(evaluation.worse_probability),
                float(evaluation.expected_shanten_delta) / 9.0,
            )

    result = torch.from_numpy(values)
    if device is not None:
        result = result.to(device)
    return result


__all__ = [
    "KAN_ADAPTER_FEATURE_DIM",
    "KAN_ADAPTER_FEATURE_NAMES",
    "KAN_ADAPTER_FEATURE_VERSION",
    "tensorize_kan_adapter_features",
]
