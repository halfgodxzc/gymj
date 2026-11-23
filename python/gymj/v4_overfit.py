import argparse
from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from ._gymj import (
    ACTION_SPACE_SIZE,
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    ObservationEncoder,
)
from .demonstration import TeacherDecision
from .env import EnvState
from .teacher import (
    DECLARE_READY_ACTION_ID,
    PASS_ACTION_ID,
    PON_ACTION_ID,
    RON_ACTION_ID,
    TSUMO_ACTION_ID,
    ShantenTeacher,
)
from .v4_actor import V4Actor, V4ActorConfig
from .v4_features import V4TensorBatch, tensorize_v4_decisions


V4_OVERFIT_DATASET_VERSION = 1

DISCARD_CATEGORY = "discard"
PON_PASS_CATEGORY = "pon_pass"
HU_CATEGORY = "hu"
READY_CATEGORY = "ready"
V4_OVERFIT_CATEGORIES = (
    DISCARD_CATEGORY,
    PON_PASS_CATEGORY,
    HU_CATEGORY,
    READY_CATEGORY,
)


@dataclass(frozen=True)
class V4LabeledDecision:
    features: object
    category: str
    supervision_action_ids: tuple[int, ...]
    target_action_ids: tuple[int, ...]


@dataclass(frozen=True)
class V4OverfitDataset:
    features: V4TensorBatch
    supervision_mask: torch.Tensor
    target_mask: torch.Tensor
    categories: torch.Tensor
    rounds_scanned: int

    def __len__(self) -> int:
        return self.features.batch_size

    def to(self, device: torch.device | str) -> "V4OverfitDataset":
        return V4OverfitDataset(
            features=self.features.to(device),
            supervision_mask=self.supervision_mask.to(device),
            target_mask=self.target_mask.to(device),
            categories=self.categories.to(device),
            rounds_scanned=self.rounds_scanned,
        )

    def index_select(self, indices: torch.Tensor) -> "V4OverfitDataset":
        return V4OverfitDataset(
            features=self.features.index_select(indices),
            supervision_mask=self.supervision_mask.index_select(
                0,
                indices.to(self.supervision_mask.device),
            ),
            target_mask=self.target_mask.index_select(
                0,
                indices.to(self.target_mask.device),
            ),
            categories=self.categories.index_select(
                0,
                indices.to(self.categories.device),
            ),
            rounds_scanned=self.rounds_scanned,
        )


@dataclass(frozen=True)
class V4OverfitConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.0
    seed: int = 20260929
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay cannot be negative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")


@dataclass(frozen=True)
class V4OverfitMetrics:
    loss: float
    overall_accuracy: float
    category_accuracy: Mapping[str, float]
    category_counts: Mapping[str, int]


@dataclass(frozen=True)
class V4OverfitResult:
    actor: V4Actor
    initial_metrics: V4OverfitMetrics
    final_metrics: V4OverfitMetrics
    dataset: V4OverfitDataset


def _legal_action_ids(features) -> tuple[int, ...]:
    return tuple(
        action_id
        for action_id, action in enumerate(features.actions)
        if action.legal
    )


def _supported_action_ids(features) -> set[int]:
    return {
        action_id
        for action_id, action in enumerate(features.actions)
        if action.legal and action.supported_by_v4
    }


def _teacher_decision(
    teacher: ShantenTeacher,
    player_id: int,
    observation,
    legal_action_ids: tuple[int, ...],
) -> TeacherDecision:
    legal_mask = tuple(
        int(action_id in legal_action_ids)
        for action_id in range(ACTION_SPACE_SIZE)
    )
    state = EnvState(
        player_id=player_id,
        observation=observation,
        legal_mask=legal_mask,
    )
    return teacher.decide(state)


def _encoded_action_ids(encoded) -> tuple[int, ...]:
    return tuple(
        action_id
        for action_id in range(ACTION_SPACE_SIZE)
        if encoded.is_legal(action_id)
    )


def label_v4_decision(
    features,
    teacher_decision: TeacherDecision,
) -> Optional[V4LabeledDecision]:
    legal = set(_legal_action_ids(features))
    supported = _supported_action_ids(features)
    if len(legal) <= 1:
        return None

    hu_targets = tuple(
        action_id
        for action_id in (TSUMO_ACTION_ID, RON_ACTION_ID)
        if action_id in legal
    )
    if hu_targets:
        supervision = tuple(sorted(legal & supported))
        return V4LabeledDecision(
            features=features,
            category=HU_CATEGORY,
            supervision_action_ids=supervision,
            target_action_ids=hu_targets,
        )

    if DECLARE_READY_ACTION_ID in legal:
        if PASS_ACTION_ID not in legal:
            raise ValueError("ready declaration must have a Pass alternative")
        return V4LabeledDecision(
            features=features,
            category=READY_CATEGORY,
            supervision_action_ids=(PASS_ACTION_ID, DECLARE_READY_ACTION_ID),
            target_action_ids=(DECLARE_READY_ACTION_ID,),
        )

    if not teacher_decision.trainable:
        return None
    supervision = tuple(
        action_id
        for action_id in teacher_decision.supervised_action_ids
        if action_id in supported
    )
    targets = tuple(
        action_id
        for action_id in teacher_decision.target_action_ids
        if action_id in supported
    )
    if not targets:
        return None
    if all(0 <= action_id < 27 for action_id in supervision):
        category = DISCARD_CATEGORY
    elif set(supervision) == {PON_ACTION_ID, PASS_ACTION_ID}:
        category = PON_PASS_CATEGORY
    else:
        return None
    return V4LabeledDecision(
        features=features,
        category=category,
        supervision_action_ids=supervision,
        target_action_ids=targets,
    )


def collect_v4_overfit_examples(
    category_counts: Mapping[str, int],
    *,
    seed_start: int = 510_000_000,
    teacher_seed: int = 20260929,
    max_rounds: int = 20_000,
) -> tuple[list[V4LabeledDecision], int]:
    requested = {name: int(category_counts.get(name, 0))
                 for name in V4_OVERFIT_CATEGORIES}
    if any(count < 0 for count in requested.values()):
        raise ValueError("category counts cannot be negative")
    if not any(requested.values()):
        raise ValueError("at least one V4 overfit example is required")
    teacher = ShantenTeacher(seed=teacher_seed, kan_probability=0.0)
    collected = {name: [] for name in V4_OVERFIT_CATEGORIES}

    for round_index in range(max_rounds):
        if all(len(collected[name]) >= requested[name]
               for name in V4_OVERFIT_CATEGORIES):
            examples = [
                example
                for name in V4_OVERFIT_CATEGORIES
                for example in collected[name]
            ]
            return examples, round_index

        arena = Arena(seed=seed_start + round_index)
        arena.start_round()
        while not arena.finished:
            decision = arena.current_decision()
            encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)
            legal_action_ids = _encoded_action_ids(encoded)
            teacher_decision = _teacher_decision(
                teacher,
                decision.player,
                ObservationEncoder.encode(arena, decision.player),
                legal_action_ids,
            )
            candidate_category = None
            if TSUMO_ACTION_ID in legal_action_ids or RON_ACTION_ID in legal_action_ids:
                candidate_category = HU_CATEGORY
            elif DECLARE_READY_ACTION_ID in legal_action_ids:
                candidate_category = READY_CATEGORY
            elif teacher_decision.trainable:
                supervised = set(teacher_decision.supervised_action_ids)
                if supervised and all(
                    0 <= action_id < 27 for action_id in supervised
                ):
                    candidate_category = DISCARD_CATEGORY
                elif supervised == {PON_ACTION_ID, PASS_ACTION_ID}:
                    candidate_category = PON_PASS_CATEGORY

            if (
                candidate_category is not None
                and len(collected[candidate_category]) < requested[candidate_category]
            ):
                features = DecisionFeatureEncoder.encode(arena, decision.player)
                label = label_v4_decision(features, teacher_decision)
                if label is None or label.category != candidate_category:
                    raise RuntimeError("V4 category prefilter disagrees with labeler")
                collected[label.category].append(label)

            selected = teacher_decision.selected_action_id
            if not encoded.is_legal(selected):
                raise RuntimeError("teacher selected an illegal rollout action")
            arena.submit_action(encoded.action_at(selected))

    missing = {
        name: requested[name] - len(collected[name])
        for name in V4_OVERFIT_CATEGORIES
        if len(collected[name]) < requested[name]
    }
    raise RuntimeError(
        f"could not collect V4 category quotas in {max_rounds} rounds: {missing}"
    )


def build_v4_overfit_dataset(
    category_counts: Mapping[str, int],
    *,
    seed_start: int = 510_000_000,
    teacher_seed: int = 20260929,
    max_rounds: int = 20_000,
) -> V4OverfitDataset:
    examples, rounds_scanned = collect_v4_overfit_examples(
        category_counts,
        seed_start=seed_start,
        teacher_seed=teacher_seed,
        max_rounds=max_rounds,
    )
    features = tensorize_v4_decisions([
        example.features for example in examples
    ])
    supervision = torch.zeros(
        (len(examples), ACTION_SPACE_SIZE),
        dtype=torch.bool,
    )
    targets = torch.zeros_like(supervision)
    categories = torch.empty(len(examples), dtype=torch.long)
    for row, example in enumerate(examples):
        supervision[row, list(example.supervision_action_ids)] = True
        targets[row, list(example.target_action_ids)] = True
        categories[row] = V4_OVERFIT_CATEGORIES.index(example.category)
    if torch.any(targets & ~supervision):
        raise RuntimeError("V4 targets must be a subset of supervision")
    if torch.any(supervision & ~features.selectable_actions):
        raise RuntimeError("V4 supervision contains an unsupported action")
    return V4OverfitDataset(
        features=features,
        supervision_mask=supervision,
        target_mask=targets,
        categories=categories,
        rounds_scanned=rounds_scanned,
    )


def v4_set_loss(
    logits: torch.Tensor,
    supervision_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if supervision_mask.shape != logits.shape:
        raise ValueError("supervision mask must match logits")
    if target_mask.shape != logits.shape:
        raise ValueError("target mask must match logits")
    if not torch.all(target_mask.any(dim=1)):
        raise ValueError("every row must have a target action")
    if torch.any(target_mask & ~supervision_mask):
        raise ValueError("target actions must be supervised")
    supervised = logits.masked_fill(~supervision_mask, -torch.inf)
    targets = logits.masked_fill(~target_mask, -torch.inf)
    return (
        torch.logsumexp(supervised, dim=1)
        - torch.logsumexp(targets, dim=1)
    ).mean()


def evaluate_v4_overfit(
    actor: V4Actor,
    dataset: V4OverfitDataset,
) -> V4OverfitMetrics:
    actor.eval()
    with torch.inference_mode():
        logits = actor(dataset.features)
        loss = v4_set_loss(
            logits,
            dataset.supervision_mask,
            dataset.target_mask,
        )
        predictions = logits.masked_fill(
            ~dataset.supervision_mask,
            -torch.inf,
        ).argmax(dim=1)
        correct = dataset.target_mask.gather(
            1,
            predictions.unsqueeze(1),
        ).squeeze(1)
    category_accuracy = {}
    category_counts = {}
    for category_id, name in enumerate(V4_OVERFIT_CATEGORIES):
        selected = dataset.categories == category_id
        count = int(selected.sum().item())
        category_counts[name] = count
        category_accuracy[name] = (
            float(correct[selected].float().mean().item())
            if count else float("nan")
        )
    return V4OverfitMetrics(
        loss=float(loss.item()),
        overall_accuracy=float(correct.float().mean().item()),
        category_accuracy=category_accuracy,
        category_counts=category_counts,
    )


def _training_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def train_v4_overfit(
    dataset: V4OverfitDataset,
    *,
    actor: Optional[V4Actor] = None,
    actor_config: V4ActorConfig = V4ActorConfig(),
    config: V4OverfitConfig = V4OverfitConfig(),
) -> V4OverfitResult:
    torch.manual_seed(config.seed)
    device = _training_device(config.device)
    actor = (actor or V4Actor(actor_config)).to(device)
    training_data = dataset.to(device)
    initial = evaluate_v4_overfit(actor, training_data)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)

    for _ in range(config.epochs):
        actor.train()
        order = torch.randperm(len(dataset), generator=generator)
        for start in range(0, len(dataset), config.batch_size):
            indices = order[start:start + config.batch_size]
            batch = training_data.index_select(indices)
            optimizer.zero_grad(set_to_none=True)
            logits = actor(batch.features)
            loss = v4_set_loss(
                logits,
                batch.supervision_mask,
                batch.target_mask,
            )
            loss.backward()
            optimizer.step()

    final = evaluate_v4_overfit(actor, training_data)
    return V4OverfitResult(
        actor=actor,
        initial_metrics=initial,
        final_metrics=final,
        dataset=dataset,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V4 memorization check")
    parser.add_argument("--samples-per-category", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    counts = {
        name: args.samples_per_category
        for name in V4_OVERFIT_CATEGORIES
    }
    dataset = build_v4_overfit_dataset(counts)
    result = train_v4_overfit(
        dataset,
        config=V4OverfitConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        ),
    )
    print(f"rounds_scanned={dataset.rounds_scanned}")
    print(
        f"initial loss={result.initial_metrics.loss:.6f} "
        f"accuracy={result.initial_metrics.overall_accuracy:.2%}"
    )
    print(
        f"final loss={result.final_metrics.loss:.6f} "
        f"accuracy={result.final_metrics.overall_accuracy:.2%}"
    )
    for name in V4_OVERFIT_CATEGORIES:
        print(
            f"{name}: count={result.final_metrics.category_counts[name]} "
            f"accuracy={result.final_metrics.category_accuracy[name]:.2%}"
        )


if __name__ == "__main__":
    main()
