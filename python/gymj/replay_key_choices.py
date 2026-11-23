"""Replay public two-family discard roots under a frozen greedy policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import shutil
import time

import torch

from .collect_effective_draw_games import collect_episode, sha256
from .filter_replay_choices import REPLAY_SCREEN_VERSION, screen_key_discards
from . import replay_effective_draw_choices as legacy
from .v4_agent import V4Agent


KEY_REPLAY_VERSION = 1
_AGENT = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def verify_public_case(candidates, task, observation, chicken, player_id):
    actual = screen_key_discards(candidates, task["selected_action_id"],
                               observation, chicken, player_id)
    if actual is None or any(actual[key] != task[key] for key in (
            "families", "candidates", "alternatives", "same_shanten_definition")):
        raise ValueError("screened public features/alternatives differ from replay")


def restore_key_task(episode, task):
    return legacy.restore_task(episode, task, discard_verifier=verify_public_case)


def verify_original(agent, episode):
    if not agent.greedy or agent.temperature != 1.0:
        raise ValueError("replay requires frozen greedy policy")
    rebuilt = collect_episode(agent, episode["episode_seed"], episode["dealer"], episode["policy_sha256"])
    if rebuilt["decision_count"] != episode["decision_count"] or rebuilt["terminal"] != episode["terminal"]:
        raise ValueError("original greedy termination/ledger differs from source")
    for actual, stored in zip(rebuilt["decisions"], episode["decisions"]):
        for key in ("decision_index", "player_id", "action_id", "action",
                    "legal_action_ids", "observation_sha256"):
            if actual[key] != stored[key]:
                raise ValueError(f"original greedy {key} differs from source")
    legacy.validate_terminal(episode["terminal"])


def replay_wall(agent, episode, cases, *, repeat_check=False):
    verify_original(agent, episode)
    roots, outcomes = [], []
    for case in cases:
        if (case["policy_sha256"] != episode["policy_sha256"]
                or case["episode_seed"] != episode["episode_seed"]):
            raise ValueError("root/source identity mismatch")
        seat = case["player_id"]
        baseline = episode["terminal"]
        alternatives, compact_pairs = [], []
        for pair in case["alternatives"]:
            branch = legacy.continue_branch(agent, episode, case, pair["action_id"], restore=restore_key_task)
            if repeat_check:
                repeated = legacy.continue_branch(agent, episode, case, pair["action_id"], restore=restore_key_task)
                if branch != repeated:
                    raise ValueError("alternative continuation is not reproducible")
            delta = [b - a for a, b in zip(baseline["score_deltas"], branch["terminal"]["score_deltas"])]
            categories = {category: [b - a for a, b in zip(values,
                          branch["terminal"]["ledger"]["category_net_by_player"][category])]
                          for category, values in baseline["ledger"]["category_net_by_player"].items()}
            if sum(delta) != 0 or [sum(values[p] for values in categories.values()) for p in range(4)] != delta:
                raise ValueError("paired four-player/category deltas disagree")
            compact = {
                "action_id": pair["action_id"], "families": pair["families"],
                "score_deltas": branch["terminal"]["score_deltas"],
                "score_delta_by_player": delta, "actor_score_delta": delta[seat],
                "category_delta_by_player": categories,
            }
            compact_pairs.append(compact)
            alternatives.append({**compact, "branch": branch})
        root_info = {
            "episode_seed": case["episode_seed"], "dealer": case["dealer"],
            "source_episode_index": case["source_episode_index"],
            "decision_index": case["decision_index"], "player_id": seat,
            "selected_action_id": case["selected_action_id"], "families": case["families"],
            "observation_sha256": case["observation_sha256"],
            "baseline_score_deltas": baseline["score_deltas"],
        }
        roots.append({**root_info, "alternatives": alternatives})
        outcomes.append({**root_info, "alternatives": compact_pairs})
    return {
        "key_replay_version": KEY_REPLAY_VERSION, "episode_seed": episode["episode_seed"],
        "baseline_matches_recorded": True, "baseline_terminal": episode["terminal"],
        "baseline_source_episode_index": cases[0]["source_episode_index"],
        "original_trajectory_note": "full original decisions stored once in materialized episode; restore prefix and verify fingerprints",
        "roots": roots, "repeat_check_passed": bool(repeat_check),
    }, outcomes


def _init_worker(actor, expected_hash):
    global _AGENT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha256(actor) != expected_hash:
        raise ValueError("replay actor hash mismatch")
    _AGENT = V4Agent.from_checkpoint(actor, device="cpu", greedy=True)


def _run_wall(job):
    info, cases, repeat_check = job
    episode = legacy._load_episode(info["path"], info["sha256"])
    return replay_wall(_AGENT, episode, cases, repeat_check=repeat_check)


def _results(jobs, workers, actor, actor_hash):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(actor), actor_hash)) as pool:
        source, pending = iter(jobs), deque()
        for _ in range(workers * 2):
            job = next(source, None)
            if job is None:
                break
            pending.append(pool.submit(_run_wall, job))
        while pending:
            yield pending.popleft().result()
            job = next(source, None)
            if job is not None:
                pending.append(pool.submit(_run_wall, job))


def validate_inputs(data_root, screen_root, actor):
    return legacy.validate_inputs(data_root, screen_root, actor,
                                 screen_version_key="replay_screen_version",
                                 expected_screen_version=REPLAY_SCREEN_VERSION)


def run_replays(data_root, screen_root, output, *, limit=0, workers=12,
                selection_seed=2026101415, repeat_check=False):
    if limit < 0 or workers < 1:
        raise ValueError("invalid replay budget")
    data_root, screen_root, output = (Path(path).resolve() for path in (data_root, screen_root, output))
    actor = data_root / "actor.pt"
    collection, screening, critical = validate_inputs(data_root, screen_root, actor)
    cases = []
    with gzip.open(screen_root / "candidates.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            cases.append(json.loads(line))
    keys = [(case["episode_seed"], case["decision_index"]) for case in cases]
    if not cases or len(set(keys)) != len(keys):
        raise ValueError("empty or duplicate roots")
    if limit > len(cases):
        raise ValueError("requested more roots than screening contains")
    if limit:
        cases = random.Random(selection_seed).sample(cases, limit)
    cases.sort(key=lambda case: (case["source_episode_index"], case["decision_index"]))
    by_episode = defaultdict(list)
    for case in cases:
        if case["policy_sha256"] != collection["actor_sha256"]:
            raise ValueError("screening policy differs from collection")
        by_episode[case["source_episode_index"]].append(case)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    shutil.copy2(actor, output / "actor.pt")
    shutil.copy2(Path(__file__), output / "executor_source.py")
    episode_files = {}
    with gzip.open(screen_root / "source_episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in by_episode:
                path = output / "episodes" / f"episode_{index:05d}.json.gz"
                with gzip.open(path, "xt", encoding="utf-8", compresslevel=3) as target:
                    target.write(line)
                episode_files[index] = {"path": str(path), "sha256": sha256(path)}
    if set(episode_files) != set(by_episode):
        raise ValueError("missing source episodes")
    with gzip.open(output / "selected_cases.jsonl.gz", "xt", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, separators=(",", ":")) + "\n")
    manifest = {
        "key_replay_version": KEY_REPLAY_VERSION, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "screen_root": str(screen_root),
        "source_collection_manifest_sha256": sha256(data_root / "manifest.json"),
        "source_screen_manifest_sha256": sha256(screen_root / "manifest.json"),
        "actor_sha256": collection["actor_sha256"], "extension_sha256": collection["extension_sha256"],
        "critical_source_hashes": critical, "executor_sha256": sha256(Path(__file__)),
        "legacy_executor_sha256": sha256(Path(legacy.__file__)),
        "torch_version": str(torch.__version__), "roots": len(cases), "walls": len(by_episode),
        "alternative_branches": sum(len(case["alternatives"]) for case in cases),
        "workers": workers, "selection_seed": selection_seed, "repeat_check": repeat_check,
        "baseline_mode": "full greedy original verified once per wall; trajectory stored once",
        "episode_files": episode_files, "training_performed": False,
        "label_note": "single-wall greedy action-conditioned outcomes, not expected Q or on-policy PPO data",
    }
    write_json(output / "manifest.json", manifest)
    counts, family_sums, family_counts = Counter(), Counter(), Counter()
    started = time.perf_counter()
    try:
        jobs = ((episode_files[index], by_episode[index], repeat_check) for index in sorted(by_episode))
        with gzip.open(output / "results.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as details, \
                gzip.open(output / "outcomes.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as outcomes:
            for result, compact_roots in _results(jobs, workers, output / "actor.pt", collection["actor_sha256"]):
                details.write(json.dumps(result, separators=(",", ":")) + "\n")
                counts["verified_walls"] += 1
                for root in compact_roots:
                    outcomes.write(json.dumps(root, separators=(",", ":")) + "\n")
                    counts["completed_roots"] += 1
                    for pair in root["alternatives"]:
                        counts["completed_alternatives"] += 1
                        delta = pair["actor_score_delta"]
                        counts["positive" if delta > 0 else "negative" if delta < 0 else "tied"] += 1
                        for family in pair["families"]:
                            family_counts[family] += 1
                            family_sums[family] += delta
                if counts["verified_walls"] == 1 or counts["verified_walls"] % 50 == 0:
                    progress = {"status": "running", "counts": dict(counts),
                                "elapsed_seconds": time.perf_counter() - started}
                    write_json(output / "progress.json", progress)
                    print(f"Replayed {counts['completed_roots']}/{len(cases)} roots; "
                          f"{counts['completed_alternatives']} alternatives; "
                          f"{time.perf_counter() - started:.1f}s", flush=True)
        if counts["completed_roots"] != len(cases) or counts["completed_alternatives"] != manifest["alternative_branches"]:
            raise ValueError("replay completion count mismatch")
        validate_inputs(data_root, screen_root, output / "actor.pt")
        if (sha256(Path(__file__)) != manifest["executor_sha256"]
                or sha256(Path(legacy.__file__)) != manifest["legacy_executor_sha256"]):
            raise ValueError("executor changed during run")
        summary = {
            "counts": dict(counts), "elapsed_seconds": time.perf_counter() - started,
            "descriptive_mean_delta_by_family": {name: family_sums[name] / count for name, count in family_counts.items()},
            "pairs_by_family": dict(family_counts), "family_counts_overlap": True,
            "statistical_unit": "episode_seed; no independent-pair confidence interval",
            "repeat_note": "repeat_check is deterministic reproducibility, not independent stochastic continuations",
        }
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", {"status": "complete", **summary})
        manifest.update(status="complete", outputs={name: sha256(output / name) for name in (
            "selected_cases.jsonl.gz", "results.jsonl.gz", "outcomes.jsonl.gz", "summary.json", "executor_source.py",
        )})
        return summary
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "progress.json", {"status": "failed", "error": manifest["error"]})
        raise
    finally:
        write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--repeat-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_replays(args.data_root, args.screen_root, args.output,
                                limit=args.limit, workers=args.workers, repeat_check=args.repeat_check), indent=2))


if __name__ == "__main__":
    main()
