from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch

from ._gymj import (
    ACTION_ENCODING_VERSION,
    DECISION_FEATURE_VERSION,
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    RoundEndReason,
)
from .v4_actor import (
    V4Actor,
    V4ActorConfig,
    v4_actor_architecture_version,
)
from .v4_features import V4_TENSOR_VERSION, tensorize_v4_decision


V4_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class V4RoundRun:
    seed: int
    step_count: int
    end_reason: RoundEndReason
    hu_players: tuple[int, ...]
    score_deltas: tuple[int, int, int, int]


@dataclass(frozen=True)
class V4ActionSelection:
    action_id: int
    log_probability: float
    entropy: float
    actor_trainable: bool


class V4Agent:
    def __init__(
        self,
        actor: V4Actor,
        device: torch.device | str = "cpu",
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        if not 0.0 < temperature < float("inf"):
            raise ValueError("temperature must be positive and finite")
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.greedy = greedy
        self.temperature = float(temperature)
        self._seed = seed
        self._generator = self._make_generator(seed)

    def _make_generator(self, seed: Optional[int]) -> torch.Generator:
        generator = torch.Generator(device=self.device.type)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        return generator

    def reset_episode(self, seed: int) -> None:
        combined = seed if self._seed is None else seed ^ self._seed
        self._generator.manual_seed(combined)

    def select_action(self, decision_features) -> V4ActionSelection:
        batch = tensorize_v4_decision(
            decision_features,
            device=self.device,
        )
        with torch.inference_mode():
            logits = self.actor(batch)[0] / self.temperature
            selectable = torch.isfinite(logits)
            if not selectable.any():
                raise RuntimeError("V4 decision has no selectable action")
            if self.greedy:
                action_id = int(logits.argmax().item())
            else:
                probabilities = torch.softmax(logits, dim=0)
                action_id = int(torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self._generator,
                ).item())
            log_probabilities = torch.log_softmax(logits, dim=0)
            probabilities = torch.softmax(logits, dim=0)
            entropy = -(
                probabilities[selectable] * log_probabilities[selectable]
            ).sum()
            return V4ActionSelection(
                action_id=action_id,
                log_probability=float(log_probabilities[action_id].item()),
                entropy=float(entropy.item()),
                actor_trainable=int(selectable.sum().item()) > 1,
            )

    def select_action_id(self, decision_features) -> int:
        return self.select_action(decision_features).action_id

    def select_from_arena(self, arena: Arena) -> int:
        if arena.finished:
            raise RuntimeError("finished arena has no current decision")
        decision = arena.current_decision()
        return self.select_action_id(
            DecisionFeatureEncoder.encode(arena, decision.player)
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        device: torch.device | str = "cpu",
        **agent_options,
    ) -> "V4Agent":
        actor, _ = load_v4_actor_checkpoint(path, device=device)
        return cls(actor, device=device, **agent_options)


def run_v4_round(
    agent: V4Agent,
    seed: int,
    *,
    dealer: int = 0,
    max_steps: int = 1000,
) -> V4RoundRun:
    if max_steps <= 0:
        raise ValueError("max steps must be positive")
    arena = Arena(seed=seed, dealer=dealer)
    arena.start_round()
    agent.reset_episode(seed)

    for step_count in range(1, max_steps + 1):
        decision = arena.current_decision()
        encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)
        action_id = agent.select_action_id(
            DecisionFeatureEncoder.encode(arena, decision.player)
        )
        if not encoded.is_legal(action_id):
            raise RuntimeError("V4 agent selected an illegal action")
        arena.submit_action(encoded.action_at(action_id))
        if arena.finished:
            result = arena.result
            return V4RoundRun(
                seed=seed,
                step_count=step_count,
                end_reason=result.reason,
                hu_players=tuple(int(player) for player in result.hu_players),
                score_deltas=tuple(
                    int(score)
                    for score in arena.round_settlement.ledger.score_deltas
                ),
            )
    raise RuntimeError(f"V4 round exceeded {max_steps} steps")


def save_v4_actor_checkpoint(
    path: Path | str,
    actor: V4Actor,
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": V4_CHECKPOINT_VERSION,
        "actor_architecture_version": v4_actor_architecture_version(
            actor.config
        ),
        "tensor_version": V4_TENSOR_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "actor_config": asdict(actor.config),
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in actor.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(checkpoint, destination)


def load_v4_actor_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[V4Actor, dict]:
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    config = V4ActorConfig(**checkpoint["actor_config"])
    expected_versions = {
        "checkpoint_version": V4_CHECKPOINT_VERSION,
        "actor_architecture_version": v4_actor_architecture_version(config),
        "tensor_version": V4_TENSOR_VERSION,
        "decision_feature_version": DECISION_FEATURE_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
    }
    for name, expected in expected_versions.items():
        if checkpoint.get(name) != expected:
            raise ValueError(
                f"V4 checkpoint {name} is incompatible: "
                f"expected {expected}, got {checkpoint.get(name)}"
            )
    actor = V4Actor(config)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.to(device).eval()
    return actor, checkpoint


__all__ = [
    "V4ActionSelection",
    "V4Agent",
    "V4RoundRun",
    "load_v4_actor_checkpoint",
    "run_v4_round",
    "save_v4_actor_checkpoint",
]
