from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ._gymj import (
    ACTION_FEATURE_COUNT,
    ACTION_SPACE_SIZE,
    MAX_MELDS,
    TILE_TYPE_COUNT,
    ActionFeatureEncoder,
    ActionFeatureIndex,
    CompactObservation,
)


_SHANTEN = slice(
    int(ActionFeatureIndex.NORMAL_SHANTEN),
    int(ActionFeatureIndex.ALL_TRIPLETS_SHANTEN) + 1,
)
_EFFECTIVE_KINDS = slice(
    int(ActionFeatureIndex.NORMAL_EFFECTIVE_DRAW_KINDS),
    int(ActionFeatureIndex.ALL_TRIPLETS_EFFECTIVE_DRAW_KINDS) + 1,
)
_EFFECTIVE_COPIES = slice(
    int(ActionFeatureIndex.NORMAL_EFFECTIVE_DRAW_COPIES),
    int(ActionFeatureIndex.ALL_TRIPLETS_EFFECTIVE_DRAW_COPIES) + 1,
)
_OFF_SUIT = slice(
    int(ActionFeatureIndex.WAN_FLUSH_OFF_SUIT_TILES),
    int(ActionFeatureIndex.TIAO_FLUSH_OFF_SUIT_TILES) + 1,
)
_MAX_TOTAL_TILE_COPIES = 4 * TILE_TYPE_COUNT
_POST_DISCARD_TILE_COUNT = 13
_DAIMINKAN_ACTION_ID = 3 * TILE_TYPE_COUNT + 1


def normalize_action_feature_batch(
    raw_features: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    if raw_features.ndim != 3 or raw_features.shape[1:] != (
        ACTION_SPACE_SIZE,
        ACTION_FEATURE_COUNT,
    ):
        raise ValueError(
            f"raw action features must have shape [B, {ACTION_SPACE_SIZE}, {ACTION_FEATURE_COUNT}]"
        )
    if legal_mask.shape != raw_features.shape[:2]:
        raise ValueError("legal mask must have shape [B, 87]")
    if not torch.all((legal_mask == 0) | (legal_mask == 1)):
        raise ValueError("legal mask entries must be 0 or 1")

    features = raw_features.to(dtype=torch.float32).clone()
    if not torch.isfinite(features).all():
        raise ValueError("raw action features must be finite")
    discards = features[:, :TILE_TYPE_COUNT]
    discards[:, :, _SHANTEN] = (8.0 - discards[:, :, _SHANTEN]) / 9.0
    discards[:, :, _EFFECTIVE_KINDS] /= TILE_TYPE_COUNT
    discards[:, :, _EFFECTIVE_COPIES] /= _MAX_TOTAL_TILE_COPIES
    discards[:, :, _OFF_SUIT] /= _POST_DISCARD_TILE_COUNT
    features[:, :, int(ActionFeatureIndex.CONCEALED_TILES_CONSUMED)] /= 4

    supported = torch.arange(ACTION_SPACE_SIZE, device=features.device) <= _DAIMINKAN_ACTION_ID
    valid = legal_mask.to(device=features.device, dtype=torch.bool) & supported
    features *= valid.unsqueeze(-1)
    return features


def tensorize_action_features(
    observations: Sequence[CompactObservation],
    legal_mask_words: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    if not observations:
        raise ValueError("at least one observation is required")
    masks = (
        [observation.legal_mask for observation in observations]
        if legal_mask_words is None else legal_mask_words
    )
    if len(masks) != len(observations):
        raise ValueError("each observation needs one legal mask")

    raw_rows = []
    expanded_masks = []
    for observation, words in zip(observations, masks):
        if len(words) != 2 or any(not 0 <= int(word) < 1 << 64 for word in words):
            raise ValueError("legal mask must contain two uint64 words")
        raw_rows.append(ActionFeatureEncoder.encode(observation, words))
        expanded_masks.append([
            (int(words[action_id // 64]) >> (action_id % 64)) & 1
            for action_id in range(ACTION_SPACE_SIZE)
        ])

    return _normalize_rows(raw_rows, expanded_masks)


def tensorize_action_features_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> torch.Tensor:
    shapes = {
        "hand_counts": (TILE_TYPE_COUNT,),
        "visible_counts": (TILE_TYPE_COUNT,),
        "meld_tiles": (MAX_MELDS,),
        "meld_owners": (MAX_MELDS,),
        "meld_valid": (MAX_MELDS,),
        "meld_count": (),
        "current_discard": (),
        "legal_mask_words": (2,),
    }
    if "hand_counts" not in arrays:
        raise ValueError("action feature arrays are missing hand_counts")
    batch_size = len(arrays["hand_counts"])
    if batch_size == 0:
        raise ValueError("at least one observation is required")

    for name, trailing_shape in shapes.items():
        if name not in arrays:
            raise ValueError(f"action feature arrays are missing {name}")
        array = np.asarray(arrays[name])
        if array.shape != (batch_size,) + trailing_shape:
            raise ValueError(f"{name} has an invalid action feature shape")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must have integer values")

    raw_rows = []
    expanded_masks = []
    for index in range(batch_size):
        words = arrays["legal_mask_words"][index].tolist()
        raw_rows.append(ActionFeatureEncoder.encode_from_arrays(
            arrays["hand_counts"][index].tolist(),
            arrays["visible_counts"][index].tolist(),
            arrays["meld_tiles"][index].tolist(),
            arrays["meld_owners"][index].tolist(),
            arrays["meld_valid"][index].tolist(),
            int(arrays["meld_count"][index]),
            words,
            int(arrays["current_discard"][index]),
        ))
        expanded_masks.append([
            (int(words[action_id // 64]) >> (action_id % 64)) & 1
            for action_id in range(ACTION_SPACE_SIZE)
        ])
    return _normalize_rows(raw_rows, expanded_masks)


def _normalize_rows(raw_rows, expanded_masks):
    raw = torch.from_numpy(np.asarray(raw_rows, dtype=np.float32))
    legal_mask = torch.from_numpy(np.asarray(expanded_masks, dtype=np.bool_))
    return normalize_action_feature_batch(raw, legal_mask)
