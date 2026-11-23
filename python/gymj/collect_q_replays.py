"""Collect paired action-conditioned Monte Carlo targets, not PPO rollouts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np
import torch

from . import _gymj
from .collect_effective_draw_games import sha256
from .diagnostic_dataset import _terminal_record, observation_fingerprint
from .replay_effective_draw_choices import _load_episode, validate_terminal
from .replay_key_choices import restore_key_task, validate_inputs
from .trajectory import pack_legal_mask
from .v4_agent import V4Agent
from .v4_critic import V4_PRIVILEGED_TENSOR_VERSION, tensorize_v4_privileged_states
from .v4_features import V4_PLAYER_COUNT, V4_TENSOR_VERSION, tensorize_v4_decisions


Q_REPLAY_VERSION = 1
SCORE_DIVISOR = 30.0
SPLITS = ("train", "validation", "test")
_AGENT = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def select_cases(cases, splits, count, seed):
    """One root per wall, preserving source splits and ignoring outcomes."""
    if count < 1:
        raise ValueError("root count must be positive")
    groups = {name: set(splits[name]) for name in SPLITS}
    if any(len(groups[name]) != len(splits[name]) for name in SPLITS):
        raise ValueError("duplicate seed within split")
    if any(groups[a] & groups[b] for a, b in (("train", "validation"),
                                             ("train", "test"), ("validation", "test"))):
        raise ValueError("source splits overlap")
    owner = {wall: name for name, walls in groups.items() for wall in walls}
    by_wall, seen = defaultdict(list), set()
    for case in cases:
        key = case["episode_seed"], case["decision_index"]
        if key in seen:
            raise ValueError("duplicate screened root")
        seen.add(key)
        if case["episode_seed"] not in owner:
            raise ValueError("screened wall has no source split")
        original = case["selected_action_id"]
        alternatives = [pair["action_id"] for pair in case["alternatives"]]
        if not alternatives or len(set(alternatives)) != len(alternatives):
            raise ValueError("missing or duplicate alternative")
        if original in alternatives or any(not 0 <= action < _gymj.TILE_TYPE_COUNT
                                           for action in [original, *alternatives]):
            raise ValueError("Q replay supports distinct discard actions only")
        by_wall[case["episode_seed"]].append(case)
    # Match the source wall proportions, rounding by the largest remainder.
    total = sum(map(len, groups.values()))
    if not total:
        raise ValueError("empty source splits")
    exact = {name: count * len(groups[name]) / total for name in SPLITS}
    quotas = {name: int(exact[name]) for name in SPLITS}
    for name in sorted(SPLITS, key=lambda name: -(exact[name] - quotas[name]))[:count - sum(quotas.values())]:
        quotas[name] += 1
    rng, chosen = random.Random(seed), []
    for name in SPLITS:
        walls = sorted(groups[name] & by_wall.keys())
        if len(walls) < quotas[name]:
            raise ValueError(f"not enough distinct {name} walls")
        for wall in sorted(rng.sample(walls, quotas[name])):
            case = rng.choice(sorted(by_wall[wall], key=lambda row: row["decision_index"]))
            pair = rng.choice(sorted(case["alternatives"], key=lambda row: row["action_id"]))
            chosen.append({**case, "split": name, "chosen_alternative": pair})
    return [{**case, "root_index": index} for index, case in enumerate(chosen)]


def repetition_seed(policy_seed, case, repeat):
    # Both candidate actions use the same seed for this repetition. Divergent
    # trajectories need not consume random draws at corresponding game events.
    identity = f"q-v1:{policy_seed}:{case['episode_seed']}:{case['decision_index']}:{repeat}"
    return int.from_bytes(hashlib.sha256(identity.encode("ascii")).digest()[:8], "little") & ((1 << 63) - 1)


def root_inputs(env, state):
    player = state.player_id
    decision = _gymj.DecisionFeatureEncoder.encode(env._arena, player)
    hidden = _gymj.PrivilegedStateEncoder.encode(env._arena, player)
    patterns = _gymj.PrivilegedPatternEncoder.encode(env._arena, player)
    public = tensorize_v4_decisions([decision])
    privileged = tensorize_v4_privileged_states([decision], [hidden], [patterns])
    arrays = {f"public__{item.name}": getattr(public, item.name).numpy()
              for item in fields(public)}
    arrays.update({f"privileged__{item.name}": getattr(privileged, item.name).numpy()
                   for item in fields(privileged)})
    # Exact hidden state is retained for replay auditing, not as a new model input.
    arrays.update(audit__wall_tiles=np.asarray(hidden.wall_tiles, dtype=np.uint8),
                  audit__wall_count=np.asarray(hidden.wall_count, dtype=np.uint8),
                  audit__opponent_hands=np.asarray(hidden.opponent_hands, dtype=np.uint8))
    return arrays


def arrays_fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str((value.dtype.str, value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def continue_forced(agent, episode, case, forced_action_id, sampling_seed, *,
                    expected_input_hash=None, max_steps=1000):
    if agent.greedy or agent.temperature != 1.0:
        raise ValueError("Q continuations require stochastic temperature=1")
    if max_steps < 1:
        raise ValueError("max steps must be positive")
    permitted = (case["selected_action_id"], case["chosen_alternative"]["action_id"])
    if forced_action_id not in permitted:
        raise ValueError("forced action is not a selected candidate")
    env, state = restore_key_task(episode, case)
    input_hash = arrays_fingerprint(root_inputs(env, state))
    if expected_input_hash is not None and input_hash != expected_input_hash:
        raise ValueError("candidate branches have different root inputs")
    agent.reset_episode(sampling_seed)
    actions = []
    for offset in range(max_steps):
        legal = [index for index, valid in enumerate(state.legal_mask) if valid]
        if offset == 0:
            action_id, behavior, log_probability = forced_action_id, "forced_candidate", None
        elif len(legal) == 1:
            action_id, behavior, log_probability = legal[0], "forced_legal", 0.0
        else:
            decision = _gymj.DecisionFeatureEncoder.encode(env._arena, state.player_id)
            selection = agent.select_action(decision)
            action_id, behavior = selection.action_id, "policy_sample"
            log_probability = selection.log_probability
        if action_id not in legal or (log_probability is not None and not math.isfinite(log_probability)):
            raise ValueError("invalid continuation action/probability")
        action = env.action_for_id(action_id)
        actions.append({
            "decision_index": case["decision_index"] + offset,
            "player_id": state.player_id, "action_id": action_id,
            "type": action.type.name, "tile": int(action.tile.id), "target": int(action.target),
            "observation_sha256": observation_fingerprint(state.observation, pack_legal_mask(state.legal_mask)),
            "behavior": behavior, "behavior_log_probability": log_probability,
            "forced_intervention": offset == 0,
        })
        env._arena.submit_action(action)
        if env._arena.finished:
            terminal = _terminal_record(env)
            validate_terminal(terminal)
            score = terminal["score_deltas"][case["player_id"]]
            return {"forced_action_id": forced_action_id, "sampling_seed": sampling_seed,
                    "root_input_sha256": input_hash, "actions": actions, "terminal": terminal,
                    "target_player_id": case["player_id"], "target_score": score,
                    "score_return": score / SCORE_DIVISOR}
        state = env._current_state()
    raise RuntimeError("Q continuation exceeded action limit")


def audit_branch(episode, case, branch):
    """Replay every saved action without a network, checking terminal labels."""
    env, state = restore_key_task(episode, case)
    if arrays_fingerprint(root_inputs(env, state)) != branch["root_input_sha256"]:
        raise ValueError("audit root inputs differ")
    if not branch["actions"] or branch["actions"][0]["action_id"] != branch["forced_action_id"]:
        raise ValueError("audit forced candidate differs")
    for offset, record in enumerate(branch["actions"]):
        action_id = record["action_id"]
        if (record["decision_index"] != case["decision_index"] + offset
                or state.player_id != record["player_id"]
                or not 0 <= action_id < len(state.legal_mask) or not state.legal_mask[action_id]
                or observation_fingerprint(state.observation, pack_legal_mask(state.legal_mask)) != record["observation_sha256"]):
            raise ValueError("saved suffix decision does not match replay")
        action = env.action_for_id(action_id)
        if (action.type.name != record["type"] or int(action.tile.id) != record["tile"]
                or int(action.target) != record["target"]):
            raise ValueError("saved suffix action semantics differ")
        expected_behavior = "forced_candidate" if offset == 0 else "forced_legal" if sum(state.legal_mask) == 1 else "policy_sample"
        probability = record["behavior_log_probability"]
        if (record["behavior"] != expected_behavior or record["forced_intervention"] != (offset == 0)
                or (offset == 0 and probability is not None)
                or (offset > 0 and (probability is None or not math.isfinite(probability) or probability > 1e-6))):
            raise ValueError("invalid forced/sampled behavior annotation")
        env._arena.submit_action(action)
        if env._arena.finished != (offset == len(branch["actions"]) - 1):
            raise ValueError("saved suffix termination differs")
        if not env._arena.finished:
            state = env._current_state()
    terminal = _terminal_record(env)
    validate_terminal(terminal)
    if terminal != branch["terminal"]:
        raise ValueError("saved terminal ledger differs from replay")
    score = terminal["score_deltas"][case["player_id"]]
    if (branch["target_player_id"] != case["player_id"] or branch["target_score"] != score
            or branch["score_return"] != score / SCORE_DIVISOR):
        raise ValueError("Q target ownership or normalization differs")


def summarize_root(case, branches, repeats):
    action_ids = [case["selected_action_id"], case["chosen_alternative"]["action_id"]]
    scores = np.zeros((2, repeats, V4_PLAYER_COUNT), dtype=np.int64)
    seen, seed_by_repeat = set(), {}
    input_hashes = {branch["root_input_sha256"] for branch in branches}
    if len(branches) != 2 * repeats or len(input_hashes) != 1:
        raise ValueError("incomplete branches or inconsistent root inputs")
    for branch in branches:
        slot, repeat = action_ids.index(branch["forced_action_id"]), branch["repeat_index"]
        if not 0 <= repeat < repeats or (slot, repeat) in seen:
            raise ValueError("duplicate or invalid repetition")
        seen.add((slot, repeat))
        seed = branch["sampling_seed"]
        if repeat in seed_by_repeat and seed_by_repeat[repeat] != seed:
            raise ValueError("paired repetitions have different seeds")
        seed_by_repeat[repeat] = seed
        validate_terminal(branch["terminal"])
        values = branch["terminal"]["score_deltas"]
        target = values[case["player_id"]]
        if branch["target_score"] != target or branch["score_return"] != target / SCORE_DIVISOR:
            raise ValueError("incorrect action-conditioned label")
        scores[slot, repeat] = values
    if len(set(seed_by_repeat.values())) != repeats:
        raise ValueError("repetitions must have distinct seeds")
    target = scores[:, :, case["player_id"]].astype(np.float64)
    delta = target[1] - target[0]
    return {"root_index": case["root_index"], "episode_seed": case["episode_seed"],
            "decision_index": case["decision_index"], "player_id": case["player_id"],
            "split": case["split"], "families": case["chosen_alternative"]["families"],
            "action_ids": action_ids, "repeat_count": repeats,
            "sampling_seeds": [seed_by_repeat[index] for index in range(repeats)],
            "root_input_sha256": next(iter(input_hashes)),
            "score_deltas": scores.tolist(), "mean_score": target.mean(axis=1).tolist(),
            "q_targets": (target.mean(axis=1) / SCORE_DIVISOR).tolist(),
            "sample_std_score": target.std(axis=1, ddof=1).tolist(),
            "mean_standard_error": (target.std(axis=1, ddof=1) / math.sqrt(repeats)).tolist(),
            "paired_score_differences": delta.tolist(), "mean_paired_difference": float(delta.mean()),
            "paired_standard_error": float(delta.std(ddof=1) / math.sqrt(repeats)),
            "distinct_target_scores": [len(set(row)) for row in target.tolist()]}


def _init_worker(actor, actor_hash):
    global _AGENT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha256(actor) != actor_hash:
        raise ValueError("frozen Actor hash mismatch")
    _AGENT = V4Agent.from_checkpoint(actor, device="cpu", greedy=False, temperature=1.0, seed=0)


def _run_root(job):
    case, episode_file, output, repeats, policy_seed, max_steps = job
    episode = _load_episode(episode_file["path"], episode_file["sha256"])
    root = Path(output) / "roots" / f"root_{case['root_index']:04d}"
    env, state = restore_key_task(episode, case)
    arrays = root_inputs(env, state)
    input_hash = arrays_fingerprint(arrays)
    branches = []
    for action_id in (case["selected_action_id"], case["chosen_alternative"]["action_id"]):
        for repeat in range(repeats):
            branch = continue_forced(_AGENT, episode, case, action_id,
                                     repetition_seed(policy_seed, case, repeat),
                                     expected_input_hash=input_hash, max_steps=max_steps)
            branch["repeat_index"] = repeat
            audit_branch(episode, case, branch)
            branches.append(branch)
    summary = summarize_root(case, branches, repeats)
    arrays.update(q_replay_version=np.asarray(Q_REPLAY_VERSION, dtype=np.int32),
                  action_ids=np.asarray(summary["action_ids"], dtype=np.int64),
                  score_deltas=np.asarray(summary["score_deltas"], dtype=np.int64),
                  q_targets=np.asarray(summary["q_targets"], dtype=np.float32),
                  sampling_seeds=np.asarray(summary["sampling_seeds"], dtype=np.int64))
    with root.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with gzip.open(root.with_suffix(".json.gz"), "xt", encoding="utf-8", compresslevel=3) as stream:
        json.dump({"q_replay_version": Q_REPLAY_VERSION, "case": case,
                   "summary": summary, "branches": branches, "all_suffixes_replayed": True},
                  stream, separators=(",", ":"), allow_nan=False)
    return summary


def run_collection(data_root, screen_root, output, *, roots=200, repeats=8, workers=12,
                   selection_seed=2026101604, policy_seed=2026101605, max_steps=1000):
    if roots < 1 or repeats < 2 or workers < 1 or max_steps < 1:
        raise ValueError("invalid Q replay budget")
    data_root, screen_root, output = [Path(value).resolve() for value in (data_root, screen_root, output)]
    actor = data_root / "actor.pt"
    collection, _, critical = validate_inputs(data_root, screen_root, actor)
    splits = json.loads((data_root / "splits.json").read_text(encoding="utf-8"))
    with gzip.open(screen_root / "candidates.jsonl.gz", "rt", encoding="utf-8") as stream:
        selected = select_cases([json.loads(line) for line in stream], splits, roots, selection_seed)
    if any(case["policy_sha256"] != collection["actor_sha256"] for case in selected):
        raise ValueError("screened Actor differs from frozen Actor")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("roots", "episodes", "frozen_engine"):
        (output / name).mkdir()
    shutil.copy2(actor, output / "actor.pt")
    shutil.copy2(_gymj.__file__, output / "frozen_engine" / Path(_gymj.__file__).name)
    shutil.copy2(data_root / "splits.json", output / "source_splits.json")
    repo = Path(__file__).resolve().parents[2]
    sources = {**critical, **{str(Path(path).resolve().relative_to(repo)): sha256(path)
                             for path in (Path(__file__), Path(__file__).with_name("replay_key_choices.py"),
                                          Path(__file__).with_name("replay_effective_draw_choices.py"),
                                          Path(__file__).with_name("v4_critic.py"))}}
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(repo / name.replace("\\", "/"), name.replace("\\", "/"))
    write_json(output / "selected_cases.json", selected)
    write_json(output / "splits.json", {"statistical_unit": "source wall; one root per wall",
                                       **{name: [case["episode_seed"] for case in selected if case["split"] == name]
                                          for name in SPLITS}})
    needed = {case["source_episode_index"] for case in selected}
    episode_files = {}
    with gzip.open(screen_root / "source_episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in needed:
                continue
            episode = json.loads(line)
            if episode["policy_sha256"] != collection["actor_sha256"]:
                raise ValueError("materialized episode Actor differs")
            path = output / "episodes" / f"episode_{index:05d}.json.gz"
            with gzip.open(path, "xt", encoding="utf-8", compresslevel=3) as destination:
                destination.write(line)
            episode_files[index] = {"path": str(path), "sha256": sha256(path)}
    if set(episode_files) != needed:
        raise ValueError("source episode missing")
    manifest = {
        "q_replay_version": Q_REPLAY_VERSION, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "screen_root": str(screen_root),
        "source_manifest_sha256": sha256(data_root / "manifest.json"),
        "screen_manifest_sha256": sha256(screen_root / "manifest.json"),
        "source_splits_sha256": sha256(data_root / "splits.json"),
        "actor_sha256": collection["actor_sha256"], "extension_sha256": collection["extension_sha256"],
        "source_hashes": sources, "torch_version": str(torch.__version__),
        "tensor_versions": {"public": V4_TENSOR_VERSION, "privileged": V4_PRIVILEGED_TENSOR_VERSION},
        "roots": roots, "actions_per_root": 2, "repeats_per_action": repeats,
        "expected_trajectories": roots * 2 * repeats, "workers": workers, "max_steps": max_steps,
        "selection_seed": selection_seed, "policy_seed": policy_seed,
        "sampling": {"greedy": False, "temperature": 1.0, "all_four_same_actor": True,
                     "extra_kan_logit_bias": 0.0, "root_action_forced": True,
                     "paired_initial_rng_seeds": True},
        "target": {"score_divisor": SCORE_DIVISOR, "gamma": 1.0, "shanten_weight": 0.0,
                   "meaning": "root player final Arena net score; forced action then frozen stochastic R5"},
        "not_on_policy_ppo_data": True, "training_performed": False,
        "exact_wall_arrays": "audit__ fields only; excluded from existing network input",
        "episode_files": episode_files,
    }
    write_json(output / "manifest.json", manifest)
    started, summaries = time.perf_counter(), []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(output / "actor.pt"), collection["actor_sha256"])) as pool:
            futures = [pool.submit(_run_root, (case, episode_files[case["source_episode_index"]], str(output),
                                               repeats, policy_seed, max_steps)) for case in selected]
            for future in as_completed(futures):
                summaries.append(future.result())
                progress = {"status": "running", "completed_roots": len(summaries),
                            "completed_trajectories": len(summaries) * 2 * repeats,
                            "elapsed_seconds": time.perf_counter() - started}
                write_json(output / "progress.json", progress)
                if len(summaries) == 1 or len(summaries) % 10 == 0:
                    print(f"Q replay {len(summaries)}/{roots} roots; {progress['completed_trajectories']} suffixes; "
                          f"{progress['elapsed_seconds']:.1f}s", flush=True)
        validate_inputs(data_root, screen_root, output / "actor.pt")
        for name, expected in sources.items():
            if sha256(repo / name.replace("\\", "/")) != expected:
                raise ValueError(f"source changed during Q replay: {name}")
        if sha256(data_root / "splits.json") != manifest["source_splits_sha256"]:
            raise ValueError("source splits changed during Q replay")
        summaries.sort(key=lambda row: row["root_index"])
        if len(summaries) != roots:
            raise ValueError("incomplete Q replay")
        write_json(output / "q_targets.json", summaries)
        counts, reasons = Counter(), Counter()
        for case in selected:
            with gzip.open(output / "roots" / f"root_{case['root_index']:04d}.json.gz", "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            for branch in record["branches"]:
                counts["trajectories"] += 1
                counts["suffix_decisions"] += len(branch["actions"])
                counts["policy_sample_decisions"] += sum(row["behavior"] == "policy_sample" for row in branch["actions"])
                reasons[branch["terminal"]["reason_name"]] += 1
        summary = {"roots": roots, "unique_walls": len({case["episode_seed"] for case in selected}),
                   "counts": dict(counts), "end_reasons": dict(reasons),
                   "split_roots": dict(Counter(case["split"] for case in selected)),
                   "selected_pair_families": dict(Counter(family for case in selected
                                                          for family in case["chosen_alternative"]["families"])),
                   "all_trajectories_replayed_and_labels_verified": True,
                   "action_groups_with_nonconstant_scores": sum(value > 1 for row in summaries for value in row["distinct_target_scores"]),
                   "mean_action_target_standard_error": float(np.mean([value for row in summaries for value in row["mean_standard_error"]])),
                   "mean_paired_standard_error": float(np.mean([row["paired_standard_error"] for row in summaries])),
                   "elapsed_seconds": time.perf_counter() - started,
                   "interpretation": "diagnostic action-conditioned labels; not proof of an improved policy"}
        if counts["trajectories"] != roots * 2 * repeats:
            raise ValueError("trajectory count mismatch")
        write_json(output / "data_audit.json", summary)
        outputs = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                   if path.is_file() and path.name not in ("manifest.json", "progress.json")}
        manifest.update(status="complete", elapsed_seconds=summary["elapsed_seconds"], outputs=outputs)
        write_json(output / "progress.json", {"status": "complete", **summary})
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
    parser.add_argument("--roots", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--selection-seed", type=int, default=2026101604)
    parser.add_argument("--policy-seed", type=int, default=2026101605)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run_collection(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
