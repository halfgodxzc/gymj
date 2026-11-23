"""Replay preselected same-shanten alternatives with a frozen greedy actor."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import torch

from . import _gymj
from ._gymj import DecisionFeatureEncoder
from .chongfeng import ChongfengTracker
from .collect_effective_draw_games import greedy_prediction, sha256
from ._observation_storage import allocate_observation_arrays, write_observation
from .diagnostic_dataset import (
    DIAGNOSTIC_SIDECAR_VERSION, EXPECTED_LEDGER_CATEGORIES,
    _terminal_record, observation_fingerprint, stored_observation_fingerprint,
)
from .env import GymjEnv
from .filter_effective_draw_choices import (
    SCREEN_VERSION, discard_record, screen_discards,
)
from .trajectory import pack_legal_mask
from .v4_agent import V4Agent


REPLAY_VERSION = 2
CONTINUATION_TRAJECTORY_VERSION = 1
DEFAULT_SELECTION_SEED = 2026101401
_AGENT = None


def task_from_case(case: dict) -> dict:
    primary = case["primary_max_copies"]
    if primary is None or not primary["changes_action"] or primary["copies_delta"] <= 0:
        raise ValueError("case has no preselected more-copies alternative")
    candidates = {item["action_id"]: item for item in case["candidates"]}
    selected = candidates[case["selected_action_id"]]
    alternative = candidates[primary["action_id"]]
    if selected["shanten"] != alternative["shanten"]:
        raise ValueError("primary pair does not have the same shanten vector")
    return {
        key: case[key] for key in (
            "episode_seed", "dealer", "decision_index", "player_id",
            "source_episode_index", "observation_sha256", "selected_action_id",
            "primary_max_copies", "hand_counts", "visible_counts", "legal_action_ids",
            "policy_sha256",
        )
    } | {"selected_features": selected, "alternative_features": alternative}


def select_tasks(candidates_path: Path, limit: int, selection_seed: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    eligible = []
    with gzip.open(candidates_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            primary = case["primary_max_copies"]
            if primary is not None and primary["changes_action"]:
                eligible.append(task_from_case(case))
    if len(eligible) < limit:
        raise ValueError(f"requested {limit} cases but only {len(eligible)} are eligible")
    keys = [(item["episode_seed"], item["decision_index"]) for item in eligible]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate screened decision")
    # Sample before seeing any alternative outcome; full runs retain every case.
    chosen = random.Random(selection_seed).sample(eligible, limit)
    return sorted(chosen, key=lambda item: (item["source_episode_index"], item["decision_index"]))


def _validate_record(env, state, record, index):
    if record["decision_index"] != index or state.player_id != record["player_id"]:
        raise ValueError("recorded decision index/player differs from replay")
    legal_ids = [action_id for action_id, valid in enumerate(state.legal_mask) if valid]
    if legal_ids != sorted(record["legal_action_ids"]):
        raise ValueError("recorded legal actions differ from replay")
    fingerprint = observation_fingerprint(state.observation, pack_legal_mask(state.legal_mask))
    if fingerprint != record["observation_sha256"]:
        raise ValueError("recorded observation fingerprint differs from replay")
    action = env.action_for_id(int(record["action_id"]))
    fields = {"type": int(action.type), "actor": int(action.actor),
              "tile": int(action.tile.id), "target": int(action.target)}
    if (any(record["action"][key] != value for key, value in fields.items())
            or record["action"]["type_name"] != action.type.name):
        raise ValueError("recorded action semantics differ from replay")


def restore_task(episode: dict, task: dict, *, discard_verifier=None):
    if episode.get("diagnostic_sidecar_version") != DIAGNOSTIC_SIDECAR_VERSION:
        raise ValueError("unsupported sidecar version")
    sampling = episode["sampling"]
    if (sampling.get("selection") != "greedy_argmax"
            or sampling.get("temperature") != 1.0
            or sampling.get("all_four_seats_same_actor") is not True
            or sampling.get("extra_kan_logit_bias") != 0.0):
        raise ValueError("source must be four-seat frozen greedy self-play without extra bias")
    for key in ("episode_seed", "dealer", "policy_sha256"):
        if episode[key] != task[key]:
            raise ValueError(f"case/source {key} mismatch")
    records = episode["decisions"]
    if len(records) != episode["decision_count"]:
        raise ValueError("source decision count mismatch")
    target = task["decision_index"]
    if not 0 <= target < len(records):
        raise ValueError("case decision index is outside source episode")
    env = GymjEnv(dealer=episode["dealer"])
    state = env.reset(seed=episode["episode_seed"])
    chicken = ChongfengTracker()
    for index in range(target + 1):
        record = records[index]
        _validate_record(env, state, record, index)
        if index == target:
            break
        if record["action"]["type_name"] == "DISCARD":
            chicken.observe_discard(record["player_id"], record["action"]["tile"])
        result = env.step(int(record["action_id"]))
        if result.done or result.next_decision is None:
            raise ValueError("source terminated before target decision")
        state = result.next_decision
    if records[target]["action_id"] != task["selected_action_id"]:
        raise ValueError("case original action differs from source")
    if task["player_id"] != state.player_id or task["observation_sha256"] != records[target]["observation_sha256"]:
        raise ValueError("case player/fingerprint differs from source")
    if list(state.observation.hand_counts) != task["hand_counts"]:
        raise ValueError("case hand differs from replay")
    if list(state.observation.visible_counts) != task["visible_counts"]:
        raise ValueError("case visible counts differ from replay")
    if sorted(task["legal_action_ids"]) != sorted(records[target]["legal_action_ids"]):
        raise ValueError("case legal actions differ from source")
    features = DecisionFeatureEncoder.encode(env._arena, state.player_id)
    candidates = [discard_record(features, tile, state.player_id, chicken)
                  for tile, valid in enumerate(state.legal_mask[:_gymj.TILE_TYPE_COUNT]) if valid]
    if discard_verifier is not None:
        discard_verifier(candidates, task, state.observation, chicken, state.player_id)
        return env, state
    actual = screen_discards(candidates, task["selected_action_id"])
    if actual is None or actual["primary_max_copies"] != task["primary_max_copies"]:
        raise ValueError("preselected alternative differs from recomputed public features")
    by_id = {item["action_id"]: item for item in candidates}
    if by_id[task["selected_action_id"]] != task["selected_features"]:
        raise ValueError("recorded original features differ from replay")
    if by_id[task["primary_max_copies"]["action_id"]] != task["alternative_features"]:
        raise ValueError("recorded alternative features differ from replay")
    return env, state


def _policy_prediction(agent, env, state):
    legal = [index for index, valid in enumerate(state.legal_mask) if valid]
    if len(legal) == 1:
        return legal[0], 0.0, 0.0, None
    return greedy_prediction(agent, DecisionFeatureEncoder.encode(env._arena, state.player_id))


def _attach_return_labels(actions, terminal):
    remaining_shanten = [0.0] * 4
    last_seen = set()
    for record in reversed(actions):
        seat = record["player_id"]
        remaining_shanten[seat] += record["shanten_reward"]
        record["future_shanten_reward"] = remaining_shanten[seat]
        record["arena_terminal_net_score"] = terminal["score_deltas"][seat]
        record["terminal_category_returns"] = {
            category: terminal["ledger"]["category_net_by_player"][category][seat]
            for category in EXPECTED_LEDGER_CATEGORIES
        }
        record["last_decision_for_player"] = seat not in last_seen
        last_seen.add(seat)


def continue_branch(agent, episode, task, forced_action_id, max_steps=1000,
                    check_original=False, *, restore=None):
    env, state = (restore or restore_task)(episode, task)
    agent.reset_episode(episode["episode_seed"])
    actions = []
    observation_arrays = allocate_observation_arrays(1)
    start = task["decision_index"]
    for offset in range(max_steps):
        index = start + offset
        predicted, policy_log_probability, entropy, logits = _policy_prediction(agent, env, state)
        if check_original:
            if index >= len(episode["decisions"]):
                raise ValueError("original continuation exceeded recorded episode")
            record = episode["decisions"][index]
            _validate_record(env, state, record, index)
            if predicted != record["action_id"]:
                raise ValueError("original greedy action differs from recorded policy")
        action_id = forced_action_id if offset == 0 else predicted
        action = env.action_for_id(int(action_id))
        intervention = int(action_id) != predicted
        if intervention:
            if logits is None or logits[action_id] is None:
                raise ValueError("forced alternative has no finite policy logit")
            finite = [value for value in logits if value is not None]
            maximum = max(finite)
            policy_log_probability = logits[action_id] - maximum - math.log(
                math.fsum(math.exp(value - maximum) for value in finite)
            )
        write_observation(observation_arrays, 0, state.observation)
        mask_words = pack_legal_mask(state.legal_mask)
        transition = {
            "decision_index": index, "player_id": state.player_id,
            "action_id": int(action_id), "type": action.type.name,
            "tile": int(action.tile.id), "target": int(action.target),
            "public_observation": {name: values[0].tolist() for name, values in observation_arrays.items()},
            "observation_sha256": stored_observation_fingerprint(observation_arrays, 0, mask_words),
            "legal_mask_words": list(mask_words),
            "legal_action_ids": [index for index, valid in enumerate(state.legal_mask) if valid],
            "policy_argmax_action_id": predicted, "policy_logits": logits,
            "policy_softmax_log_probability": policy_log_probability, "policy_entropy": entropy,
            "behavior": "forced_alternative" if intervention else "greedy_policy",
            "behavior_log_probability": None if intervention else 0.0,
            "is_forced_intervention": intervention,
        }
        result = env.step(int(action_id))
        transition.update(
            before_shanten=result.reward_step.before_shanten,
            after_shanten=result.reward_step.after_shanten,
            shanten_reward=result.reward_step.shanten_reward,
            done=result.done,
        )
        actions.append(transition)
        if result.done:
            terminal = _terminal_record(env)
            if check_original and (
                    index + 1 != episode["decision_count"] or terminal != episode["terminal"]):
                raise ValueError("original terminal details/ledger differ from source")
            validate_terminal(terminal)
            _attach_return_labels(actions, terminal)
            return {"forced_action_id": forced_action_id,
                    "continuation_steps": len(actions), "actions": actions, "terminal": terminal,
                    "trajectory_version": CONTINUATION_TRAJECTORY_VERSION,
                    "feature_reconstruction": "source episode prefix + these submitted actions; verify each observation fingerprint"}
        if result.next_decision is None:
            raise ValueError("unfinished continuation has no next decision")
        state = result.next_decision
    raise RuntimeError(f"continuation exceeded {max_steps} steps")


def validate_terminal(terminal):
    scores = terminal["score_deltas"]
    categories = terminal["ledger"]["category_net_by_player"]
    if len(scores) != 4 or sum(scores) != 0 or set(categories) != set(EXPECTED_LEDGER_CATEGORIES):
        raise ValueError("invalid terminal scores/categories")
    if any(len(values) != 4 or sum(values) != 0 for values in categories.values()):
        raise ValueError("terminal category is not a four-player zero-sum ledger")
    if [sum(values[seat] for values in categories.values()) for seat in range(4)] != scores:
        raise ValueError("terminal categories do not reconstruct scores")


def replay_task(agent, episode, task, *, max_steps=1000, repeat_check=False):
    if not agent.greedy or agent.temperature != 1.0:
        raise ValueError("replay requires a greedy actor at temperature 1")
    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    baseline = continue_branch(agent, episode, task, task["selected_action_id"],
                               max_steps, check_original=True)
    alternative = continue_branch(agent, episode, task, task["primary_max_copies"]["action_id"], max_steps)
    if repeat_check:
        repeated = continue_branch(agent, episode, task, task["primary_max_copies"]["action_id"], max_steps)
        if repeated != alternative:
            raise ValueError("greedy alternative continuation is not reproducible")
    score_delta = [b - a for a, b in zip(baseline["terminal"]["score_deltas"],
                                       alternative["terminal"]["score_deltas"])]
    categories = {
        category: [b - a for a, b in zip(
            baseline["terminal"]["ledger"]["category_net_by_player"][category],
            alternative["terminal"]["ledger"]["category_net_by_player"][category],
        )] for category in EXPECTED_LEDGER_CATEGORIES
    }
    seat = task["player_id"]
    return {"replay_version": REPLAY_VERSION, "case": task,
            "baseline": baseline, "alternative": alternative,
            "baseline_matches_recorded": True,
            "repeat_check_passed": True if repeat_check else None,
            "score_delta_by_player": score_delta,
            "category_delta_by_player": categories, "actor_score_delta": score_delta[seat],
            "actor_category_delta": {name: values[seat] for name, values in categories.items()}}


@lru_cache(maxsize=4)
def _load_episode(path, expected_hash):
    if sha256(Path(path)) != expected_hash:
        raise ValueError("materialized source episode hash mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _init_worker(checkpoint, expected_hash):
    global _AGENT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha256(Path(checkpoint)) != expected_hash:
        raise ValueError("replay actor hash mismatch")
    _AGENT = V4Agent.from_checkpoint(checkpoint, device="cpu", greedy=True)


def _run_task(job):
    task, episode_info, max_steps, repeat_check = job
    episode = _load_episode(episode_info["path"], episode_info["sha256"])
    try:
        return replay_task(_AGENT, episode, task, max_steps=max_steps, repeat_check=repeat_check)
    except Exception as error:
        raise RuntimeError(f"case {task['episode_seed']}/{task['decision_index']}: {error}") from error


def _ordered_results(jobs, workers, actor, actor_hash):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(actor), actor_hash)) as pool:
        source = iter(jobs)
        pending = deque()
        for _ in range(workers * 2):
            job = next(source, None)
            if job is None:
                break
            pending.append(pool.submit(_run_task, job))
        while pending:
            yield pending.popleft().result()
            job = next(source, None)
            if job is not None:
                pending.append(pool.submit(_run_task, job))


def validate_inputs(data_root, screen_root, actor, *, screen_version_key="screen_version",
                    expected_screen_version=SCREEN_VERSION):
    collection = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    screening = json.loads((screen_root / "manifest.json").read_text(encoding="utf-8"))
    if collection["status"] != "complete" or screening["status"] != "complete":
        raise ValueError("collection and screening must both be complete")
    if screening.get(screen_version_key) != expected_screen_version:
        raise ValueError("incompatible screening version")
    if collection["selection"] != "greedy_argmax" or collection["all_four_seats_same_actor"] is not True:
        raise ValueError("collection was not frozen greedy self-play")
    if collection["temperature"] != 1.0 or collection["extra_kan_logit_bias"] != 0.0:
        raise ValueError("collection sampling configuration is incompatible")
    if sha256(actor) != collection["actor_sha256"]:
        raise ValueError("checkpoint differs from collection")
    extension_hash = sha256(Path(_gymj.__file__))
    if extension_hash != collection["extension_sha256"] or extension_hash != screening["extension_sha256"]:
        raise ValueError("loaded engine differs from collection/screening")
    if str(torch.__version__) != collection["torch_version"]:
        raise ValueError("PyTorch version differs from collection")
    for name, expected in screening["outputs"].items():
        if sha256(screen_root / name) != expected:
            raise ValueError(f"screening output hash mismatch: {name}")
    if Counter(item["sha256"] for item in collection["shards"]) != Counter(
            item["sha256"] for item in screening["sources"]):
        raise ValueError("screened shards differ from collected shards")
    repo = Path(__file__).resolve().parents[2]
    critical_python = {
        f"python/gymj/{name}.py" for name in (
            "v4_actor", "v4_agent", "v4_features", "kan_adapter", "env", "reward",
            "diagnostic_dataset", "_observation_storage", "trajectory", "chongfeng",
        )
    }
    hashes = {
        name: expected for name, expected in collection["source_hashes"].items()
        if name.replace("\\", "/") in critical_python
        or name.replace("\\", "/").startswith("libgy/src/")
    }
    # Collector manifests on Windows retain native path separators.
    recorded_python = {name.replace("\\", "/") for name in collection["source_hashes"]}
    if not critical_python.issubset(recorded_python):
        raise ValueError("collection lacks critical source hashes")
    for name, expected in hashes.items():
        if sha256(repo / name.replace("\\", "/")) != expected:
            raise ValueError(f"critical source changed since collection: {name}")
    return collection, screening, hashes


def run_replays(data_root: Path, screen_root: Path, output: Path, *,
                limit=100, workers=12, selection_seed=DEFAULT_SELECTION_SEED,
                repeat_check=False, max_steps=1000, actor: Path | None = None):
    if min(limit, workers, max_steps) <= 0:
        raise ValueError("limit, workers and max steps must be positive")
    data_root, screen_root, output = map(lambda path: Path(path).resolve(), (data_root, screen_root, output))
    actor = Path(actor).resolve() if actor is not None else data_root / "actor.pt"
    collection, screening, critical_hashes = validate_inputs(data_root, screen_root, actor)
    tasks = select_tasks(screen_root / "candidates.jsonl.gz", limit, selection_seed)
    output.mkdir(parents=True, exist_ok=False)
    frozen_actor = output / "actor.pt"
    shutil.copy2(actor, frozen_actor)
    shutil.copy2(Path(__file__), output / "executor_source.py")
    manifest = {
        "replay_version": REPLAY_VERSION, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "screen_root": str(screen_root),
        "source_collection_manifest_sha256": sha256(data_root / "manifest.json"),
        "source_screen_manifest_sha256": sha256(screen_root / "manifest.json"),
        "actor_sha256": collection["actor_sha256"],
        "extension_sha256": collection["extension_sha256"], "critical_source_hashes": critical_hashes,
        "executor_sha256": sha256(Path(__file__)), "torch_version": str(torch.__version__),
        "cases": limit, "workers": workers, "device": "cpu", "greedy": True,
        "selection_seed": selection_seed, "selection_algorithm": "python.random.Random.sample",
        "baseline_mode": "recompute_and_verify_full_original_suffix",
        "alternative_rule": "screening-time primary_max_copies, no outcome-based selection",
        "repeat_check": repeat_check, "max_steps": max_steps, "command": sys.argv,
        "training_performed": False,
        "continuation_trajectory_version": CONTINUATION_TRAJECTORY_VERSION,
        "observation_version": _gymj.OBSERVATION_VERSION,
        "training_data_scope": "public observations and actions; separate raw Arena/category returns and auxiliary shanten returns",
        "behavior_note": "deterministic greedy continuation except forced alternative; softmax log probability is NOT behavior likelihood or on-policy PPO data",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        episode_dir = output / "episodes"
        episode_dir.mkdir()
        required = {task["source_episode_index"] for task in tasks}
        episode_files = {}
        with gzip.open(screen_root / "source_episodes.jsonl.gz", "rt", encoding="utf-8") as source:
            for index, line in enumerate(source):
                if index not in required:
                    continue
                episode = json.loads(line)
                if episode["policy_sha256"] != collection["actor_sha256"]:
                    raise ValueError("source episode policy hash mismatch")
                path = episode_dir / f"episode_{index:05d}.json.gz"
                with gzip.open(path, "xt", encoding="utf-8", compresslevel=3) as target:
                    target.write(line)
                episode_files[index] = {"path": str(path), "sha256": sha256(path)}
        if required != set(episode_files):
            raise ValueError("selected case references a missing source episode")
        with gzip.open(output / "selected_cases.jsonl.gz", "xt", encoding="utf-8") as target:
            for task in tasks:
                target.write(json.dumps(task, separators=(",", ":")) + "\n")
        jobs = ((task, episode_files[task["source_episode_index"]], max_steps, repeat_check) for task in tasks)
        counts, category_sums = Counter(), Counter()
        score_sum = 0
        with gzip.open(output / "results.jsonl.gz", "xt", encoding="utf-8", compresslevel=3) as target:
            for result in _ordered_results(jobs, workers, frozen_actor, collection["actor_sha256"]):
                target.write(json.dumps(result, separators=(",", ":")) + "\n")
                counts["completed_cases"] += 1
                counts["baseline_verified"] += result["baseline_matches_recorded"]
                counts["repeated_alternatives_verified"] += result["repeat_check_passed"] is True
                for branch in ("baseline", "alternative"):
                    counts[f"{branch}_continuation_decisions"] += len(result[branch]["actions"])
                counts["forced_interventions"] += sum(
                    record["is_forced_intervention"] for record in result["alternative"]["actions"]
                )
                delta = result["actor_score_delta"]
                counts["positive_delta" if delta > 0 else "negative_delta" if delta < 0 else "zero_delta"] += 1
                score_sum += delta
                category_sums.update(result["actor_category_delta"])
                report_every = 10 if limit <= 100 else 100
                if counts["completed_cases"] == 1 or counts["completed_cases"] % report_every == 0:
                    print(f"Replayed {counts['completed_cases']}/{limit} cases; {time.perf_counter() - started:.1f}s", flush=True)
                    progress = {"status": "running", "counts": dict(counts),
                                "elapsed_seconds": time.perf_counter() - started}
                    (output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        if counts["completed_cases"] != limit:
            raise ValueError("completed replay count differs from request")
        # Recheck frozen assets and input artifacts before declaring success.
        validate_inputs(data_root, screen_root, frozen_actor)
        if sha256(Path(__file__)) != manifest["executor_sha256"]:
            raise ValueError("executor source changed during replay")
        summary = {
            "counts": dict(counts), "actor_score_delta_sum": score_sum,
            "actor_score_delta_mean": score_sum / limit,
            "actor_category_delta_sums": dict(category_sums),
            "unique_episode_seeds": len({task["episode_seed"] for task in tasks}),
            "elapsed_seconds": time.perf_counter() - started,
            "mean_scope": "descriptive decision-weighted mean, not an independent-sample confidence interval",
            "statistical_unit": "episode_seed; cluster by wall for later statistical analysis",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "progress.json").write_text(json.dumps({"status": "complete", **summary}, indent=2), encoding="utf-8")
        manifest.update(status="complete", episode_files=episode_files,
                        outputs={name: sha256(output / name) for name in
                                 ("selected_cases.jsonl.gz", "results.jsonl.gz", "summary.json", "executor_source.py")})
        return summary
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "progress.json").write_text(json.dumps(
            {"status": "failed", "error": manifest["error"]}, indent=2), encoding="utf-8")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run_replays(args.data_root, args.screen_root, args.output,
                                limit=args.limit, workers=args.workers,
                                selection_seed=args.selection_seed, repeat_check=args.repeat_check,
                                max_steps=args.max_steps, actor=args.actor), indent=2))


if __name__ == "__main__":
    main()
