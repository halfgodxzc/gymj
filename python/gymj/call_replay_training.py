"""Off-policy paired-call supervision, separate from PPO returns and ratios."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F


def paired_call_loss(logits, original, alternative, differences, *, cap=1.0):
    """Native-point differences weight a bounded, pairwise ranking objective."""
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("call bonus cap must be finite and positive")
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("expected nonempty batched logits")
    for values in (original, alternative, differences):
        if values.shape != (len(logits),):
            raise ValueError("pair labels must match batch size")
    if not torch.isfinite(differences).all() or (original == alternative).any():
        raise ValueError("invalid pair labels")
    for ids in (original, alternative):
        if ids.dtype != torch.long or ((ids < 0) | (ids >= logits.shape[1])).any():
            raise ValueError("invalid action IDs")
    left = logits.gather(1, original[:, None]).flatten()
    right = logits.gather(1, alternative[:, None]).flatten()
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("pair action is illegal or unsupported by Actor")
    bonus = (differences.detach() / 30.0).clamp(-cap, cap)
    return (bonus.abs() * F.softplus(-bonus.sign() * (left - right))).mean()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kan_pair_blocked_by_hu(case):
    # Current R5 deliberately masks every kan when Ron/Tsumo is legal.
    from .teacher import TSUMO_ACTION_ID, RON_ACTION_ID
    return case["family"] == "kan" and bool({TSUMO_ACTION_ID, RON_ACTION_ID}.intersection(case["legal_ids"]))


class CallReplayAuxiliary:
    """Episode-balanced sampling; rebuild only public Actor inputs at roots."""

    def __init__(self, directory, *, seed=0):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        settings = manifest["settings"]
        source = Path(settings["source"])
        episode_file = source / "episodes.jsonl"
        if _digest(episode_file) != settings["source_episodes_sha256"]:
            raise ValueError("replay source episodes changed")
        from . import call_replay
        if _digest(call_replay.__file__) != settings["module_sha256"]:
            raise ValueError("replay implementation differs from frozen collector")
        import sys
        extension = Path(sys.modules[call_replay.Arena.__module__].__file__)
        frozen = self.directory / "runtime/gymj" / extension.name
        if not frozen.exists() or _digest(extension) != _digest(frozen):
            raise ValueError("Arena differs from replay runtime")
        wanted = set(settings["source_seeds"])
        self.episodes = {ep["seed"]: ep for ep in map(json.loads, episode_file.read_text().splitlines()) if ep["seed"] in wanted}
        if set(self.episodes) != wanted:
            raise ValueError("missing source episodes")
        self.groups = defaultdict(list)
        seen = set()
        for seed_id in sorted(wanted):
            done = json.loads((self.directory / "episodes" / f"{seed_id}.done.json").read_text())
            if not done["all_branches_audited"]:
                raise ValueError("unaudited replay")
            name = f"{seed_id}.roots.jsonl.gz"
            path = self.directory / "episodes" / name
            if _digest(path) != done["files"][name]:
                raise ValueError("root shard changed")
            with gzip.open(path, "rt") as stream:
                for record in map(json.loads, stream):
                    case, summary = record["case"], record["summary"]
                    delta = summary["original_minus_alternative"]
                    if record["root_id"] in seen or case["episode_seed"] != seed_id:
                        raise ValueError("duplicate or misplaced root")
                    seen.add(record["root_id"])
                    if summary["training_scope"] != "auxiliary_pair_only_not_on_policy_ppo" or not math.isfinite(delta):
                        raise ValueError("invalid replay target")
                    repeats = settings["pon_repeats"] if case["family"] == "pon" else settings["kan_repeats"]
                    if case["family"] not in ("pon", "kan") or summary["repeats_per_action"] != repeats:
                        raise ValueError("invalid replay family or repeats")
                    if not math.isclose(delta, summary["original_mean"] - summary["alternative_mean"], abs_tol=1e-6):
                        raise ValueError("inconsistent paired score difference")
                    if case["original"] == case["alternative"] or any(a not in case["legal_ids"] for a in (case["original"], case["alternative"])):
                        raise ValueError("invalid root actions")
                    if delta != 0:
                        self.groups[seed_id].append((case, delta))
        if not self.groups:
            raise ValueError("no nonzero paired targets")
        self.seeds = sorted(self.groups)
        self.rng = random.Random(seed)
        self.metadata = dict(directory=str(self.directory), manifest_sha256=_digest(manifest_path),
                             episodes=len(self.seeds), pairs=sum(map(len, self.groups.values())),
                             sampling="uniform_episode_then_uniform_pair", score_divisor=30.0,
                             skipped_hu_kan_pairs=0, used_pairs=0)

    def loss(self, actor, batch_size, device, *, temperature=1.0, cap=1.0):
        from . import DecisionFeatureEncoder
        from .call_replay import restore
        from .v4_features import tensorize_v4_decisions
        features, original, alternative, differences = [], [], [], []
        for _ in range(batch_size):
            seed_id = self.rng.choice(self.seeds)
            case, delta = self.rng.choice(self.groups[seed_id])
            if kan_pair_blocked_by_hu(case):
                self.metadata["skipped_hu_kan_pairs"] += 1
                continue
            arena = restore(self.episodes[seed_id], case)
            features.append(DecisionFeatureEncoder.encode(arena, case["player_id"]))
            original.append(case["original"])
            alternative.append(case["alternative"])
            differences.append(delta)
        self.metadata["used_pairs"] += len(features)
        if not features:
            return next(actor.parameters()).sum() * 0.0
        logits = actor(tensorize_v4_decisions(features, device=device)) / temperature
        return paired_call_loss(logits, torch.tensor(original, device=device),
                                torch.tensor(alternative, device=device),
                                torch.tensor(differences, device=device), cap=cap)
