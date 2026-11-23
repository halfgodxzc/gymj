from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ._gymj import ScoreCategory


COMPONENT_NAMES = ("hu", "chicken", "bean", "cha_jiao")


def settlement_component_deltas(settlement) -> tuple[tuple[int, ...], ...]:
    """Return [player][component] net chips from a completed settlement."""
    category_component = {
        int(ScoreCategory.HU): 0,
        int(ScoreCategory.CHICKEN): 1,
        int(ScoreCategory.RESPONSIBILITY_CHICKEN): 1,
        int(ScoreCategory.ANKAN_BEAN): 2,
        int(ScoreCategory.KAKAN_BEAN): 2,
        int(ScoreCategory.DAIMINKAN_BEAN): 2,
        int(ScoreCategory.CHA_JIAO): 3,
    }
    rows = [[0] * len(COMPONENT_NAMES) for _ in range(4)]
    for transfer in settlement.ledger.transfers:
        try:
            component = category_component[int(transfer.category)]
        except KeyError as error:
            raise ValueError("unknown settlement score category") from error
        points = int(transfer.points)
        source = int(transfer.from_player)
        destination = int(transfer.to_player)
        if not 0 <= source < 4 or not 0 <= destination < 4 or points <= 0:
            raise ValueError("invalid settlement score transfer")
        rows[source][component] -= points
        rows[destination][component] += points

    totals = tuple(map(int, settlement.ledger.score_deltas))
    if tuple(map(sum, rows)) != totals:
        raise ValueError("settlement components do not sum to score deltas")
    return tuple(tuple(row) for row in rows)


def load_component_ledger(
    root: Path,
) -> dict[tuple[int, int], np.ndarray]:
    ledger: dict[tuple[int, int], np.ndarray] = {}
    paths = sorted(Path(root).glob("*.jsonl.gz"))
    if not paths:
        raise ValueError("data root contains no diagnostic sidecars")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                episode = json.loads(line)
                seed = int(episode["episode_seed"])
                terminal = episode["terminal"]
                category = terminal["ledger"]["category_net_by_player"]
                score_deltas = np.asarray(
                    terminal["score_deltas"], dtype=np.float64,
                )
                rows = np.column_stack((
                    np.asarray(category["hu"], dtype=np.float64),
                    np.asarray(category["chicken"], dtype=np.float64)
                    + np.asarray(
                        category["responsibility_chicken"],
                        dtype=np.float64,
                    ),
                    np.asarray(category["an_kan_bean"], dtype=np.float64)
                    + np.asarray(category["kakan_bean"], dtype=np.float64)
                    + np.asarray(
                        category["daiminkan_bean"], dtype=np.float64,
                    ),
                    np.asarray(category["cha_jiao"], dtype=np.float64),
                ))
                if not np.array_equal(rows.sum(axis=1), score_deltas):
                    raise ValueError(
                        f"ledger components do not sum for episode {seed}"
                    )
                for player_id in range(4):
                    key = (seed, player_id)
                    if key in ledger:
                        raise ValueError(f"duplicate ledger key {key}")
                    ledger[key] = rows[player_id]
    return ledger


def component_rows_for_tensors(
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    ledger: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    seeds = tensors["episode_seed"][indices].tolist()
    players = tensors["player_id"][indices].tolist()
    try:
        return np.asarray([
            ledger[(int(seed), int(player))]
            for seed, player in zip(seeds, players)
        ], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"missing ledger entry {error.args[0]}") from error


__all__ = [
    "COMPONENT_NAMES",
    "component_rows_for_tensors",
    "load_component_ledger",
    "settlement_component_deltas",
]
