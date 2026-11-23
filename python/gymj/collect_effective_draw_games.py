"""Collect frozen, greedy V4/R5 self-play with complete replay sidecars."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import torch

from . import _gymj
from ._gymj import DecisionFeatureEncoder
from .diagnostic_dataset import (
    DIAGNOSTIC_SIDECAR_VERSION, _terminal_record, observation_fingerprint,
)
from .env import GymjEnv
from .trajectory import pack_legal_mask
from .v4_agent import V4Agent
from .v4_features import tensorize_v4_decision


_AGENT = None
_POLICY_HASH = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_prediction(agent, features):
    batch = tensorize_v4_decision(features, device=agent.device)
    with torch.inference_mode():
        logits = agent.actor(batch)[0] / agent.temperature
        finite = torch.isfinite(logits)
        if not finite.any():
            raise RuntimeError("actor has no selectable action")
        action_id = int(logits.argmax().item())
        log_probabilities = torch.log_softmax(logits, dim=0)
        probabilities = torch.softmax(logits, dim=0)
        entropy = -(probabilities[finite] * log_probabilities[finite]).sum()
        values = logits.cpu().tolist()
        mask = finite.cpu().tolist()
        return action_id, float(log_probabilities[action_id]), float(entropy), [
            value if valid else None for value, valid in zip(values, mask)
        ]


def collect_episode(agent, seed: int, dealer: int, policy_sha256: str,
                    max_steps: int = 1000) -> dict:
    if not agent.greedy:
        raise ValueError("effective-draw collection requires a greedy agent")
    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    env = GymjEnv(dealer=dealer)
    state = env.reset(seed=seed)
    agent.reset_episode(seed)
    records = []
    for index in range(max_steps):
        legal_ids = [action_id for action_id, valid in enumerate(state.legal_mask) if valid]
        forced = len(legal_ids) == 1
        if forced:
            action_id, log_probability, entropy, logits = legal_ids[0], 0.0, 0.0, None
        else:
            features = DecisionFeatureEncoder.encode(env._arena, state.player_id)
            action_id, log_probability, entropy, logits = greedy_prediction(agent, features)
        action = env.action_for_id(action_id)
        records.append({
            "decision_index": index, "player_id": state.player_id,
            "action_id": action_id,
            "action": {"type": int(action.type), "type_name": action.type.name,
                       "actor": int(action.actor), "tile": int(action.tile.id),
                       "target": int(action.target)},
            "legal_action_ids": legal_ids,
            "observation_sha256": observation_fingerprint(
                state.observation, pack_legal_mask(state.legal_mask),
            ),
            "old_log_probability": log_probability, "entropy": entropy,
            "actor_trainable": not forced, "policy_logits": logits,
            "inference_skipped_forced": forced,
        })
        result = env.step(action_id)
        if result.done:
            terminal = _terminal_record(env)
            if sum(terminal["score_deltas"]) != 0:
                raise RuntimeError("terminal ledger is not zero-sum")
            return {
                "diagnostic_sidecar_version": DIAGNOSTIC_SIDECAR_VERSION,
                "episode_seed": seed, "dealer": dealer,
                "policy_sha256": policy_sha256,
                "sampling": {"selection": "greedy_argmax", "temperature": agent.temperature,
                             "all_four_seats_same_actor": True, "extra_kan_logit_bias": 0.0},
                "decision_count": len(records), "decisions": records,
                "terminal": terminal,
            }
        if result.next_decision is None:
            raise RuntimeError("unfinished episode has no next decision")
        state = result.next_decision
    raise RuntimeError(f"episode exceeded {max_steps} actions: seed={seed}")


def _init_worker(checkpoint: str, expected_hash: str):
    global _AGENT, _POLICY_HASH
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha256(Path(checkpoint)) != expected_hash:
        raise ValueError("frozen actor hash mismatch")
    _POLICY_HASH = expected_hash
    _AGENT = V4Agent.from_checkpoint(checkpoint, device="cpu", greedy=True)


def _collect_shard(task):
    destination, indices, seed_start, max_steps = task
    path = Path(destination)
    counts, reasons, actions, fans, patterns = Counter(), Counter(), Counter(), Counter(), Counter()
    scores = [0, 0, 0, 0]
    with gzip.open(path, "xt", encoding="utf-8", compresslevel=3) as stream:
        for index in indices:
            episode = collect_episode(_AGENT, seed_start + index, index % 4, _POLICY_HASH, max_steps)
            stream.write(json.dumps(episode, separators=(",", ":")) + "\n")
            counts["episodes"] += 1
            counts["decisions"] += episode["decision_count"]
            counts["nonforced_decisions"] += sum(record["actor_trainable"] for record in episode["decisions"])
            reasons[episode["terminal"]["reason_name"]] += 1
            for record in episode["decisions"]:
                actions[record["action"]["type_name"]] += 1
            for detail in episode["terminal"]["hu_settlements"]:
                fans[str(detail["score"]["total_fan"])] += 1
                patterns[detail["features"]["pattern_name"]] += 1
            scores = [a + b for a, b in zip(scores, episode["terminal"]["score_deltas"])]
    return {"path": str(path), "sha256": sha256(path), "counts": dict(counts),
            "end_reasons": dict(reasons), "actions": dict(actions), "hu_fans": dict(fans),
            "hu_patterns": dict(patterns), "score_totals_by_seat": scores}


def collect_games(checkpoint: Path, expected_hash: str, output: Path,
                  rounds: int, seed_start: int, workers: int = 12,
                  shard_rounds: int = 50, max_steps: int = 1000) -> dict:
    if min(rounds, workers, shard_rounds, max_steps) <= 0:
        raise ValueError("rounds, workers, shard rounds and max steps must be positive")
    if not 0 <= seed_start <= seed_start + rounds - 1 <= 0xFFFFFFFF:
        raise ValueError("episode seeds must fit uint32")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if sha256(checkpoint) != expected_hash.lower():
        raise ValueError("actor does not match expected SHA256")
    output.mkdir(parents=True, exist_ok=False)
    shards_dir = output / "episodes"
    shards_dir.mkdir()
    frozen_actor = output / "actor.pt"
    shutil.copy2(checkpoint, frozen_actor)
    repo = Path(__file__).resolve().parents[2]
    source_paths = sorted(set((repo / "python/gymj").rglob("*.py")) |
                          set((repo / "libgy").rglob("*.cpp")) |
                          set((repo / "libgy").rglob("*.h")))
    source_hashes = {str(path.relative_to(repo)): sha256(path) for path in source_paths}
    extension = Path(_gymj.__file__).resolve()
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(repo))
        archive.write(extension, Path("python/gymj") / extension.name)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "actor_checkpoint_source": str(checkpoint), "actor_checkpoint": str(frozen_actor),
        "actor_sha256": expected_hash.lower(), "extension_sha256": sha256(extension),
        "source_hashes": source_hashes, "source_snapshot_sha256": sha256(output / "source_snapshot.zip"),
        "git_head": git_head,
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "python_version": sys.version, "torch_version": str(torch.__version__),
        "command": sys.argv, "device": "cpu", "torch_threads_per_worker": 1,
        "workers": workers, "rounds": rounds, "shard_rounds": shard_rounds,
        "seed_start": seed_start, "seed_end": seed_start + rounds - 1,
        "dealer_rule": "episode_index % 4", "selection": "greedy_argmax",
        "temperature": 1.0, "all_four_seats_same_actor": True,
        "extra_kan_logit_bias": 0.0, "training_performed": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tasks = [(str(shards_dir / f"episodes_{index // shard_rounds:04d}.jsonl.gz"),
              list(range(index, min(index + shard_rounds, rounds))), seed_start, max_steps)
             for index in range(0, rounds, shard_rounds)]
    completed = []
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(frozen_actor), expected_hash.lower())) as pool:
            futures = [pool.submit(_collect_shard, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
                done = sum(item["counts"]["episodes"] for item in completed)
                print(f"Collected {done}/{rounds} rounds; {time.perf_counter() - started:.1f}s", flush=True)
        if sha256(frozen_actor) != expected_hash.lower() or sha256(extension) != manifest["extension_sha256"]:
            raise ValueError("checkpoint/extension changed during collection")
        if any(sha256(repo / path) != value for path, value in source_hashes.items()):
            raise ValueError("source changed during collection")
        counters = {name: Counter() for name in ("counts", "end_reasons", "actions", "hu_fans", "hu_patterns")}
        scores = [0, 0, 0, 0]
        for item in completed:
            for name, counter in counters.items():
                counter.update(item[name])
            scores = [a + b for a, b in zip(scores, item["score_totals_by_seat"])]
        if counters["counts"]["episodes"] != rounds:
            raise ValueError("collected episode count differs from request")
        summary = {name: dict(counter) for name, counter in counters.items()}
        summary["score_totals_by_seat"] = scores
        summary["elapsed_seconds"] = time.perf_counter() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest.update(status="complete", shards=sorted(completed, key=lambda item: item["path"]))
        return summary
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-rounds", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(collect_games(args.actor, args.expected_sha256, args.output,
                                  args.rounds, args.seed_start, args.workers, args.shard_rounds), indent=2))


if __name__ == "__main__":
    main()
