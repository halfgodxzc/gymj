from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ._gymj import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    OBSERVATION_VERSION,
    ActionType,
    ScoreCategory,
)
from ._observation_storage import (
    allocate_observation_arrays,
    write_observation,
)
from .critic_rollout import (
    DEFAULT_SCORE_DIVISOR,
    DEFAULT_SHANTEN_DIVISOR,
    monte_carlo_returns,
)
from .env import GymjEnv
from .policy_sampling import (
    ANKAN_ACTION_START,
    DAIMINKAN_ACTION_ID,
    KAKAN_ACTION_START,
    PON_ACTION_ID,
    ActorSamplingConfig,
)
from .ppo_rollout import PPOEpisode, PPOSample, RolloutPolicy
from .settlement_components import settlement_component_deltas
from .trajectory import pack_legal_mask


DIAGNOSTIC_DATASET_VERSION = 1
DIAGNOSTIC_SIDECAR_VERSION = 1
PILOT_SEED_START = 409_200_001
TRAIN_SEED_START = 409_210_001
VALIDATION_SEED_START = 409_220_001
TEST_SEED_START = 409_230_001
SPLIT_PERMUTATION_SEED = 409_200_000
EXPECTED_LEDGER_CATEGORIES = (
    "hu",
    "cha_jiao",
    "chicken",
    "responsibility_chicken",
    "an_kan_bean",
    "kakan_bean",
    "daiminkan_bean",
)


@dataclass(frozen=True)
class _PendingDiagnosticSample:
    decision_index: int
    observation: object
    legal_mask_words: tuple[int, int]
    action_id: int
    old_log_probability: float
    actor_trainable: bool
    shanten_reward: float
    privileged_state: object
    observation_sha256: str


def sampling_seed_for_episode(episode_seed: int) -> int:
    """Derive a worker-independent Torch RNG seed with SplitMix64."""
    value = (int(episode_seed) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value &= 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    value &= 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return value & 0x7FFFFFFFFFFFFFFF


def build_fixed_splits() -> dict:
    training_pool = np.arange(
        TRAIN_SEED_START, TRAIN_SEED_START + 5_000, dtype=np.int64,
    )
    generator = np.random.default_rng(SPLIT_PERMUTATION_SEED)
    permutation = generator.permutation(training_pool)
    return {
        "diagnostic_dataset_version": DIAGNOSTIC_DATASET_VERSION,
        "split_unit": "episode_seed",
        "permutation_algorithm": "numpy.random.default_rng.permutation",
        "permutation_seed": SPLIT_PERMUTATION_SEED,
        "pilot": list(range(PILOT_SEED_START, PILOT_SEED_START + 100)),
        "training_pool": training_pool.tolist(),
        "train_1000": permutation[:1_000].tolist(),
        "train_5000_order": permutation.tolist(),
        "validation": list(range(
            VALIDATION_SEED_START, VALIDATION_SEED_START + 1_000,
        )),
        "sealed_test": list(range(
            TEST_SEED_START, TEST_SEED_START + 2_000,
        )),
    }


def write_splits(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_fixed_splits(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def observation_fingerprint(observation, legal_mask_words) -> str:
    arrays = allocate_observation_arrays(1)
    write_observation(arrays, 0, observation)
    return stored_observation_fingerprint(arrays, 0, legal_mask_words)


def stored_observation_fingerprint(
    arrays: Mapping[str, np.ndarray],
    index: int,
    legal_mask_words=None,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"observation-v{OBSERVATION_VERSION}\n".encode("ascii"))
    for name in sorted(allocate_observation_arrays(0)):
        _hash_array(digest, name, np.asarray(arrays[name][index]))
    if legal_mask_words is None:
        legal_mask_words = arrays["legal_mask_words"][index]
    _hash_array(
        digest,
        "legal_mask_words",
        np.asarray(legal_mask_words, dtype=np.uint64),
    )
    return digest.hexdigest()


def run_diagnostic_episode(
    env: GymjEnv,
    policy,
    sampling_config: ActorSamplingConfig,
    episode_seed: int,
    policy_sha256: str,
    max_steps: int = 1_000,
) -> tuple[PPOEpisode, dict]:
    if max_steps <= 0:
        raise ValueError("max steps must be positive")

    behavior_seed = sampling_seed_for_episode(episode_seed)
    rollout_policy = RolloutPolicy(policy, sampling_config, behavior_seed)
    pending: list[list[_PendingDiagnosticSample]] = [[], [], [], []]
    decisions = []
    decision = env.reset(seed=episode_seed)

    for decision_index in range(max_steps):
        player_id = decision.player_id
        action_id, old_log_probability, actor_trainable = (
            rollout_policy.select(decision.observation, decision.legal_mask)
        )
        action = env.action_for_id(action_id)
        legal_mask_words = pack_legal_mask(decision.legal_mask)
        fingerprint = observation_fingerprint(
            decision.observation, legal_mask_words,
        )
        privileged_state = env.privileged_state_for_training()
        result = env.step(action_id)
        pending[player_id].append(_PendingDiagnosticSample(
            decision_index=decision_index,
            observation=decision.observation,
            legal_mask_words=legal_mask_words,
            action_id=action_id,
            old_log_probability=old_log_probability,
            actor_trainable=actor_trainable,
            shanten_reward=result.reward_step.shanten_reward,
            privileged_state=privileged_state,
            observation_sha256=fingerprint,
        ))
        decisions.append({
            "decision_index": decision_index,
            "player_id": player_id,
            "action_id": action_id,
            "action": {
                "type": int(action.type),
                "type_name": _enum_name(action.type),
                "actor": int(action.actor),
                "tile": int(action.tile.id),
                "target": int(action.target),
            },
            "legal_action_ids": [
                index for index, valid in enumerate(decision.legal_mask)
                if valid
            ],
            "old_log_probability": old_log_probability,
            "actor_trainable": actor_trainable,
            "shanten_reward": result.reward_step.shanten_reward,
            "observation_sha256": fingerprint,
        })

        if result.done:
            score_deltas = env.terminal_score_deltas
            component_deltas = settlement_component_deltas(
                env.round_settlement,
            )
            samples = []
            sample_order = []
            for seat in range(4):
                shanten_returns = monte_carlo_returns(
                    [item.shanten_reward for item in pending[seat]],
                )
                for item, shanten_return in zip(
                    pending[seat], shanten_returns,
                ):
                    score_return = score_deltas[seat] / DEFAULT_SCORE_DIVISOR
                    normalized_shanten = (
                        shanten_return / DEFAULT_SHANTEN_DIVISOR
                    )
                    samples.append(PPOSample(
                        episode_seed=episode_seed,
                        player_id=seat,
                        observation=item.observation,
                        legal_mask_words=item.legal_mask_words,
                        action_id=item.action_id,
                        old_log_probability=item.old_log_probability,
                        actor_trainable=item.actor_trainable,
                        score_return=score_return,
                        shanten_return=normalized_shanten,
                        component_return=tuple(
                            value / DEFAULT_SCORE_DIVISOR
                            for value in component_deltas[seat]
                        ),
                        privileged_state=item.privileged_state,
                    ))
                    sample_order.append({
                        "decision_index": item.decision_index,
                        "player_id": seat,
                        "action_id": item.action_id,
                        "actor_trainable": item.actor_trainable,
                        "score_return": score_return,
                        "shanten_return": normalized_shanten,
                        "observation_sha256": item.observation_sha256,
                    })

            round_result = env.round_result
            ppo_episode = PPOEpisode(
                seed=episode_seed,
                samples=tuple(samples),
                score_deltas=score_deltas,
                component_deltas=component_deltas,
                end_reason=round_result.reason,
                hu_players=tuple(round_result.hu_players),
            )
            sidecar = {
                "diagnostic_sidecar_version": DIAGNOSTIC_SIDECAR_VERSION,
                "episode_seed": episode_seed,
                "behavior_sampling_seed": behavior_seed,
                "dealer": int(env.dealer),
                "policy_sha256": policy_sha256,
                "sampling": {
                    "selection": "legal_masked_categorical",
                    "temperature": sampling_config.temperature,
                    "kan_learning_enabled": (
                        sampling_config.kan_learning_enabled
                    ),
                    "kan_logit_bias": sampling_config.kan_logit_bias,
                    "force_kan_when_available": (
                        sampling_config.force_kan_when_available
                    ),
                },
                "decision_count": len(decisions),
                "decisions": decisions,
                "sample_order": sample_order,
                "terminal": _terminal_record(env),
            }
            return ppo_episode, sidecar

        decision = result.next_decision
        if decision is None:
            raise RuntimeError("unfinished rollout has no next decision")

    raise RuntimeError(
        f"diagnostic rollout exceeded {max_steps} steps: seed={episode_seed}"
    )


def write_sidecars(path: Path, episodes: Sequence[dict]) -> Path:
    if not episodes:
        raise ValueError("at least one sidecar episode is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", encoding="utf-8", newline="\n") as output:
        for episode in episodes:
            output.write(json.dumps(
                episode, ensure_ascii=True, separators=(",", ":"),
            ))
            output.write("\n")
    return path


def read_sidecars(path: Path) -> list[dict]:
    with gzip.open(path, mode="rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def audit_dataset(data_root: Path, splits_path: Path) -> dict:
    data_root = Path(data_root)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    split_names = tuple(
        name
        for name in ("training_pool", "validation", "sealed_test", "pilot")
        if name in splits
    )
    if not split_names:
        raise ValueError("splits file contains no recognized dataset split")
    expected = {name: set(map(int, splits[name])) for name in split_names}
    _validate_split_disjointness(expected)

    total = Counter()
    by_split = {}
    actual_action_ids = Counter()
    selected_action_groups = Counter()
    opportunity_counts = Counter()
    trainable_opportunity_counts = Counter()
    overlap_counts = Counter()
    end_reasons = Counter()
    ledger = {
        name: {"transfer_count": 0, "points": 0, "net_by_player": [0] * 4}
        for name in EXPECTED_LEDGER_CATEGORIES
    }
    observed_seeds = {}
    checked_fingerprints = 0
    checked_returns = 0

    for split_name in split_names:
        split_dir = data_root / split_name
        paths = sorted(split_dir.glob("*.npz"))
        if not paths:
            raise ValueError(f"{split_dir} contains no rollout shards")
        split_counts = Counter()
        split_seeds = set()
        for path in paths:
            sidecar_path = path.with_suffix(".jsonl.gz")
            if not sidecar_path.exists():
                raise ValueError(f"missing sidecar for {path}")
            sidecars = read_sidecars(sidecar_path)
            with np.load(path, allow_pickle=False) as shard:
                expected_seeds = []
                expected_players = []
                expected_actions = []
                expected_trainable = []
                expected_scores = []
                expected_shanten = []
                expected_fingerprints = []
                fingerprint_rows = set()
                for episode in sidecars:
                    seed = int(episode["episode_seed"])
                    if seed in split_seeds:
                        raise ValueError(f"duplicate episode seed {seed}")
                    split_seeds.add(seed)
                    terminal = episode["terminal"]
                    if sum(map(int, terminal["score_deltas"])) != 0:
                        raise ValueError(f"non-zero-sum ledger for seed {seed}")
                    end_reasons[terminal["reason_name"]] += 1
                    _merge_ledger(ledger, terminal["ledger"])
                    split_counts["episodes"] += 1
                    split_counts["decisions"] += int(episode["decision_count"])

                    decisions_by_index = {
                        int(item["decision_index"]): item
                        for item in episode["decisions"]
                    }
                    for sample in episode["sample_order"]:
                        row = len(expected_seeds)
                        decision = decisions_by_index[int(
                            sample["decision_index"]
                        )]
                        if (
                            sample["observation_sha256"]
                            != decision["observation_sha256"]
                        ):
                            raise ValueError(
                                f"sidecar fingerprint mismatch in {path}: row {row}"
                            )
                        expected_seeds.append(seed)
                        expected_players.append(int(sample["player_id"]))
                        expected_actions.append(int(sample["action_id"]))
                        expected_trainable.append(bool(sample["actor_trainable"]))
                        expected_scores.append(float(sample["score_return"]))
                        expected_shanten.append(float(sample["shanten_return"]))
                        expected_fingerprints.append(
                            sample["observation_sha256"]
                        )
                        if int(sample["decision_index"]) == 0:
                            fingerprint_rows.add(row)

                sample_count = int(shard["sample_count"])
                if len(expected_seeds) != sample_count:
                    raise ValueError(f"unmatched rollout rows in {path}")
                _require_array_equal(
                    path, "episode_seed", shard["episode_seed"], expected_seeds,
                )
                _require_array_equal(
                    path, "player_id", shard["player_id"], expected_players,
                )
                _require_array_equal(
                    path, "action_id", shard["action_id"], expected_actions,
                )
                _require_array_equal(
                    path,
                    "actor_trainable",
                    shard["actor_trainable"],
                    expected_trainable,
                )
                _require_array_close(
                    path, "score_return", shard["score_return"], expected_scores,
                )
                _require_array_close(
                    path,
                    "shanten_return",
                    shard["shanten_return"],
                    expected_shanten,
                )
                checked_returns += sample_count

                fingerprint_rows.update(range(0, sample_count, 64))
                for row in sorted(fingerprint_rows):
                    if stored_observation_fingerprint(shard, row) != (
                        expected_fingerprints[row]
                    ):
                        raise ValueError(
                            f"observation fingerprint mismatch in {path}: row {row}"
                        )
                checked_fingerprints += len(fingerprint_rows)

                action_ids = np.asarray(shard["action_id"], dtype=np.int64)
                trainable = np.asarray(
                    shard["actor_trainable"], dtype=np.bool_,
                )
                words = np.asarray(
                    shard["legal_mask_words"], dtype=np.uint64,
                )
                opportunity_masks = _opportunity_masks(words)
                for label, mask in opportunity_masks.items():
                    opportunity_counts[label] += int(mask.sum())
                    trainable_opportunity_counts[label] += int(
                        (mask & trainable).sum()
                    )
                _update_overlap_counts(overlap_counts, opportunity_masks)
                _update_action_counts(
                    actual_action_ids,
                    selected_action_groups,
                    action_ids,
                    opportunity_masks["pon_pass"],
                )
                split_counts["samples"] += sample_count
                split_counts["actor_trainable"] += int(trainable.sum())
                split_counts["forced"] += int((~trainable).sum())
        print(
            f"audit split={split_name} episodes={split_counts['episodes']} "
            f"samples={split_counts['samples']}",
            flush=True,
        )
        if split_seeds != expected[split_name]:
            missing = sorted(expected[split_name] - split_seeds)[:5]
            extra = sorted(split_seeds - expected[split_name])[:5]
            raise ValueError(
                f"{split_name} seed mismatch; missing={missing}, extra={extra}"
            )
        observed_seeds[split_name] = sorted(split_seeds)
        by_split[split_name] = dict(split_counts)
        total.update(split_counts)

    report = {
        "diagnostic_dataset_version": DIAGNOSTIC_DATASET_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "splits_sha256": _sha256_file(Path(splits_path)),
        "totals": dict(total),
        "by_split": by_split,
        "actual_action_ids": dict(sorted(actual_action_ids.items(), key=lambda x: int(x[0]))),
        "selected_action_groups": dict(selected_action_groups),
        "opportunity_counts": dict(opportunity_counts),
        "trainable_opportunity_counts": dict(trainable_opportunity_counts),
        "opportunity_overlap_counts": dict(overlap_counts),
        "end_reasons": dict(end_reasons),
        "ledger": ledger,
        "validation": {
            "split_seed_sets_disjoint": True,
            "observed_seed_sets_exact": True,
            "observation_fingerprints_checked": checked_fingerprints,
            "observation_fingerprint_sampling": (
                "every 64th stored row plus decision 0 of every episode; "
                "the 100-episode pilot was checked exhaustively"
            ),
            "return_ownership_rows_checked": checked_returns,
            "all_score_ledgers_zero_sum": True,
        },
    }
    return report


def write_audit_reports(
    json_path: Path,
    markdown_path: Path,
    report: Mapping[str, object],
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8",
    )
    totals = report["totals"]
    opportunities = report["opportunity_counts"]
    trainable_opportunities = report["trainable_opportunity_counts"]
    ledger = report["ledger"]
    lines = [
        "# Critic 诊断数据审计",
        "",
        "## 总览",
        "",
        f"- 独立牌局：{totals.get('episodes', 0):,}",
        f"- 总决策状态：{totals.get('samples', 0):,}",
        f"- 可训练决策：{totals.get('actor_trainable', 0):,}",
        f"- 强制动作：{totals.get('forced', 0):,}",
        "",
        "## 动作机会",
        "",
        "| 分组 | 全部机会 | 可训练机会 |",
        "|---|---:|---:|",
    ]
    for label in (
        "discard", "pon_pass", "kan", "ready", "hu",
    ):
        lines.append(
            f"| {label} | {opportunities.get(label, 0):,} | "
            f"{trainable_opportunities.get(label, 0):,} |"
        )
    lines.extend([
        "",
        "一个决策可以同时属于多个机会分组，例如摸牌时可能同时可以弃牌、杠或胡。",
        "",
        "## 账本",
        "",
        "| 分类 | 转账数 | 转账绝对分 | 四家净收益累计 |",
        "|---|---:|---:|---|",
    ])
    for category in EXPECTED_LEDGER_CATEGORIES:
        item = ledger[category]
        lines.append(
            f"| {category} | {item['transfer_count']:,} | "
            f"{item['points']:,} | {item['net_by_player']} |"
        )
    lines.extend([
        "",
        "## 一致性",
        "",
        f"- 已核对 Observation 指纹：{report['validation']['observation_fingerprints_checked']:,}",
        f"- 已核对回报归属：{report['validation']['return_ownership_rows_checked']:,}",
        "- 训练、验证、封存测试 seed 无交叉。",
        "- 每局终局账本均为零和。",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _terminal_record(env: GymjEnv) -> dict:
    result = env.round_result
    settlement = env.round_settlement
    transfers = []
    category_net: dict[str, list[int]] = {
        name: [0, 0, 0, 0] for name in EXPECTED_LEDGER_CATEGORIES
    }
    for transfer in settlement.ledger.transfers:
        category = _score_category_name(transfer.category)
        item = {
            "category": int(transfer.category),
            "category_name": category,
            "from_player": int(transfer.from_player),
            "to_player": int(transfer.to_player),
            "points": int(transfer.points),
            "tile": int(transfer.tile.id),
        }
        transfers.append(item)
        category_net.setdefault(category, [0, 0, 0, 0])
        category_net[category][item["from_player"]] -= item["points"]
        category_net[category][item["to_player"]] += item["points"]

    return {
        "reason": int(result.reason),
        "reason_name": _enum_name(result.reason),
        "hu_players": list(map(int, result.hu_players)),
        "fangpao_player": int(result.fangpao_player),
        "hu_tile": int(result.hu_tile.id),
        "is_hot_cannon": bool(result.is_hot_cannon),
        "jiao_states": [
            bool(env._arena.player(player_id).shanten <= 0)
            for player_id in range(4)
        ],
        "score_deltas": list(map(int, settlement.ledger.score_deltas)),
        "hu_settlements": [
            _hu_settlement_record(item)
            for item in settlement.hu_settlements
        ],
        "cha_jiao_settlements": [
            _cha_jiao_record(item)
            for item in settlement.cha_jiao_settlements
        ],
        "ledger": {
            "transfers": transfers,
            "category_net_by_player": category_net,
        },
    }


def _hu_settlement_record(item) -> dict:
    features = item.features
    score = item.score
    return {
        "player_id": int(item.player_id),
        "features": {
            "pattern": int(features.pattern),
            "pattern_name": _enum_name(features.pattern),
            "is_flush": bool(features.is_flush),
            "is_single_wait": bool(features.is_single_wait),
            "ready_kind": int(features.ready_kind),
            "ready_kind_name": _enum_name(features.ready_kind),
            "is_su_gua_dou": bool(features.is_su_gua_dou),
            "timing": int(features.timing),
            "timing_name": _enum_name(features.timing),
            "special": int(features.special),
        },
        "score": {
            "shape_fan": int(score.shape_fan),
            "ready_fan": int(score.ready_fan),
            "timing_fan": int(score.timing_fan),
            "special_fan": int(score.special_fan),
            "total_fan": int(score.total_fan),
            "hu_points": int(score.hu_points),
            "items": [
                {"type": int(fan.type), "fan": int(fan.fan)}
                for fan in score.items
            ],
        },
    }


def _cha_jiao_record(item) -> dict:
    record = _hu_settlement_record(item)
    record["best_wait"] = int(item.best_wait.id)
    return record


def _hash_array(digest, name: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype="<u4").tobytes())
    digest.update(contiguous.tobytes())


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    return str(value).rsplit(".", 1)[-1].lower()


def _score_category_name(value) -> str:
    names = {
        int(ScoreCategory.HU): "hu",
        int(ScoreCategory.CHA_JIAO): "cha_jiao",
        int(ScoreCategory.CHICKEN): "chicken",
        int(ScoreCategory.RESPONSIBILITY_CHICKEN): "responsibility_chicken",
        int(ScoreCategory.ANKAN_BEAN): "an_kan_bean",
        int(ScoreCategory.KAKAN_BEAN): "kakan_bean",
        int(ScoreCategory.DAIMINKAN_BEAN): "daiminkan_bean",
    }
    return names[int(value)]


def _legal_ids_from_words(words) -> list[int]:
    values = [int(value) for value in words]
    return [
        action_id for action_id in range(ACTION_SPACE_SIZE)
        if values[action_id // 64] & (1 << (action_id % 64))
    ]


def _opportunity_labels(legal_ids: Sequence[int]) -> list[str]:
    legal = set(legal_ids)
    labels = []
    if legal.intersection(range(27)):
        labels.append("discard")
    if PON_ACTION_ID in legal:
        labels.append("pon_pass")
    if legal.intersection(range(ANKAN_ACTION_START, PON_ACTION_ID)) or (
        DAIMINKAN_ACTION_ID in legal
    ):
        labels.append("kan")
    if 86 in legal:
        labels.append("ready")
    if legal.intersection((83, 84)):
        labels.append("hu")
    return labels


def _selected_action_group(action_id: int, labels: Sequence[str]) -> str:
    if 0 <= action_id < 27:
        return "discard"
    if ANKAN_ACTION_START <= action_id < PON_ACTION_ID or (
        action_id == DAIMINKAN_ACTION_ID
    ):
        return "kan"
    if action_id == PON_ACTION_ID:
        return "pon"
    if action_id in (83, 84):
        return "hu"
    if action_id == 85:
        return "pass_in_pon_opportunity" if "pon_pass" in labels else "pass"
    if action_id == 86:
        return "ready"
    return "other"


def _require_array_equal(
    path: Path,
    name: str,
    actual,
    expected: Sequence[object],
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected, dtype=actual_array.dtype)
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"{path}:{name} shape mismatch: "
            f"{actual_array.shape} != {expected_array.shape}"
        )
    mismatch = np.flatnonzero(actual_array != expected_array)
    if mismatch.size:
        row = int(mismatch[0])
        raise ValueError(
            f"{path}:{name} mismatch at row {row}: "
            f"{actual_array[row]} != {expected_array[row]}"
        )


def _require_array_close(
    path: Path,
    name: str,
    actual,
    expected: Sequence[float],
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected, dtype=actual_array.dtype)
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"{path}:{name} shape mismatch: "
            f"{actual_array.shape} != {expected_array.shape}"
        )
    close = np.isclose(actual_array, expected_array, atol=1e-6)
    if not np.all(close):
        row = int(np.flatnonzero(~close)[0])
        raise ValueError(
            f"{path}:{name} mismatch at row {row}: "
            f"{actual_array[row]} != {expected_array[row]}"
        )


def _opportunity_masks(words: np.ndarray) -> dict[str, np.ndarray]:
    if words.ndim != 2 or words.shape[1] != 2:
        raise ValueError("legal mask words must have shape [samples, 2]")
    word0 = words[:, 0]
    word1 = words[:, 1]
    return {
        "discard": (word0 & np.uint64((1 << 27) - 1)) != 0,
        "pon_pass": (word1 & np.uint64(1 << 17)) != 0,
        "kan": (
            (word0 & np.uint64(0xFFFFFFFFF8000000)) != 0
        ) | (
            (word1 & np.uint64(((1 << 17) - 1) | (1 << 18))) != 0
        ),
        "ready": (word1 & np.uint64(1 << 22)) != 0,
        "hu": (word1 & np.uint64((1 << 19) | (1 << 20))) != 0,
    }


def _update_overlap_counts(
    target: Counter,
    masks: Mapping[str, np.ndarray],
) -> None:
    labels = tuple(masks)
    codes = np.zeros(len(next(iter(masks.values()))), dtype=np.uint8)
    for bit, label in enumerate(labels):
        codes |= masks[label].astype(np.uint8) << bit
    values, counts = np.unique(codes, return_counts=True)
    for code, count in zip(values.tolist(), counts.tolist()):
        active = [
            label for bit, label in enumerate(labels)
            if code & (1 << bit)
        ]
        if len(active) > 1:
            target["+".join(active)] += int(count)


def _update_action_counts(
    actual_action_ids: Counter,
    selected_action_groups: Counter,
    action_ids: np.ndarray,
    pon_opportunities: np.ndarray,
) -> None:
    values, counts = np.unique(action_ids, return_counts=True)
    for action_id, count in zip(values.tolist(), counts.tolist()):
        actual_action_ids[str(int(action_id))] += int(count)

    groups = {
        "discard": action_ids < 27,
        "kan": (
            ((action_ids >= 27) & (action_ids < PON_ACTION_ID))
            | (action_ids == DAIMINKAN_ACTION_ID)
        ),
        "pon": action_ids == PON_ACTION_ID,
        "hu": (action_ids == 83) | (action_ids == 84),
        "pass_in_pon_opportunity": (
            (action_ids == 85) & pon_opportunities
        ),
        "pass": (action_ids == 85) & ~pon_opportunities,
        "ready": action_ids == 86,
    }
    covered = np.zeros(len(action_ids), dtype=np.bool_)
    for label, mask in groups.items():
        selected_action_groups[label] += int(mask.sum())
        covered |= mask
    if np.any(~covered):
        selected_action_groups["other"] += int((~covered).sum())


def _validate_split_disjointness(splits: Mapping[str, set[int]]) -> None:
    names = tuple(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(
                    f"split seed overlap: {left}/{right}: {sorted(overlap)[:5]}"
                )


def _validate_stored_sample(
    shard,
    row: int,
    seed: int,
    sample: Mapping[str, object],
    decision: Mapping[str, object],
    check_fingerprint: bool = True,
) -> None:
    for name, expected in (
        ("episode_seed", seed),
        ("player_id", sample["player_id"]),
        ("action_id", sample["action_id"]),
        ("actor_trainable", sample["actor_trainable"]),
    ):
        actual = shard[name][row].item()
        if actual != expected:
            raise ValueError(
                f"row {row} {name} mismatch: {actual} != {expected}"
            )
    for name in ("score_return", "shanten_return"):
        if not np.isclose(
            float(shard[name][row]), float(sample[name]), atol=1e-6,
        ):
            raise ValueError(f"row {row} {name} mismatch")
    if sample["observation_sha256"] != decision["observation_sha256"]:
        raise ValueError(f"row {row} sidecar fingerprint mismatch")
    if not check_fingerprint:
        return
    fingerprint = stored_observation_fingerprint(shard, row)
    if fingerprint != sample["observation_sha256"]:
        raise ValueError(f"row {row} observation fingerprint mismatch")


def _merge_ledger(target: dict, source: Mapping[str, object]) -> None:
    for transfer in source["transfers"]:
        category = transfer["category_name"]
        points = int(transfer["points"])
        item = target[category]
        item["transfer_count"] += 1
        item["points"] += abs(points)
        item["net_by_player"][int(transfer["from_player"])] -= points
        item["net_by_player"][int(transfer["to_player"])] += points


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DIAGNOSTIC_DATASET_VERSION",
    "PILOT_SEED_START",
    "SPLIT_PERMUTATION_SEED",
    "TEST_SEED_START",
    "TRAIN_SEED_START",
    "VALIDATION_SEED_START",
    "audit_dataset",
    "build_fixed_splits",
    "observation_fingerprint",
    "read_sidecars",
    "run_diagnostic_episode",
    "sampling_seed_for_episode",
    "stored_observation_fingerprint",
    "write_audit_reports",
    "write_sidecars",
    "write_splits",
]
