from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from gymj import (
    ActionEncoder,
    Arena,
    DecisionFeatureEncoder,
    V4Actor,
    V4ActorConfig,
    V4Agent,
    load_v4_actor_checkpoint,
    run_v4_round,
    save_v4_actor_checkpoint,
    tensorize_v4_decision,
)
from gymj.driver import start_round_passing_initial_ready


def small_actor() -> V4Actor:
    torch.manual_seed(20260929)
    return V4Actor(V4ActorConfig(
        hand_tile_hidden_dim=24,
        hand_summary_hidden_dim=24,
        hand_output_dim=48,
        player_tile_hidden_dim=16,
        player_summary_hidden_dim=16,
        player_output_dim=32,
        progress_hidden_dim=32,
        state_hidden_dim=64,
        action_scalar_hidden_dim=32,
        action_hidden_dim=48,
        family_hidden_dim=48,
        dropout=0.0,
    ))


class V4AgentTest(unittest.TestCase):
    def test_greedy_and_sampled_agents_select_supported_legal_actions(self):
        arena = Arena(seed=20260929)
        start_round_passing_initial_ready(arena)
        decision = arena.current_decision()
        features = DecisionFeatureEncoder.encode(arena, decision.player)
        encoded = ActionEncoder.encode_legal_actions(decision.legal_actions)

        for agent in (
            V4Agent(small_actor(), greedy=True),
            V4Agent(small_actor(), greedy=False, seed=7),
        ):
            action_id = agent.select_action_id(features)
            self.assertTrue(encoded.is_legal(action_id))
            self.assertTrue(features.actions[action_id].supported_by_v4)

    def test_untrained_v4_agent_can_complete_real_rounds(self):
        agent = V4Agent(small_actor(), greedy=False, seed=31)

        results = [run_v4_round(agent, seed) for seed in (71, 72)]

        self.assertTrue(all(result.step_count > 0 for result in results))
        self.assertTrue(all(sum(result.score_deltas) == 0 for result in results))

    def test_checkpoint_round_trip_preserves_logits_and_metadata(self):
        arena = Arena(seed=91)
        start_round_passing_initial_ready(arena)
        decision = arena.current_decision()
        batch = tensorize_v4_decision(
            DecisionFeatureEncoder.encode(arena, decision.player)
        )
        actor = small_actor().eval()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "v4.pt"
            save_v4_actor_checkpoint(path, actor, metadata={"name": "test"})
            restored, checkpoint = load_v4_actor_checkpoint(path)

        with torch.inference_mode():
            torch.testing.assert_close(actor(batch), restored(batch))
        self.assertEqual(checkpoint["metadata"], {"name": "test"})


if __name__ == "__main__":
    unittest.main()
