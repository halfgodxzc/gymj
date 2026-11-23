import unittest

from gymj import (
    ActionType,
    Arena,
    MeldType,
    RoundEndReason,
    RoundPhase,
)
from gymj.driver import (
    first_action,
    pass_all_responses,
    run_passive_game,
    start_round_passing_initial_ready,
)


def scripted_wall(hands, front_draws, replacement_draw=None):
    hands = [list(hand) for hand in hands]
    remaining = [4] * 27

    def reserve(tile_id):
        if not 0 <= tile_id < 27:
            raise AssertionError(f"Invalid tile id: {tile_id}")
        if remaining[tile_id] == 0:
            raise AssertionError(f"More than four copies of tile {tile_id}")
        remaining[tile_id] -= 1

    for hand in hands:
        if len(hand) > 13:
            raise AssertionError("Initial hand has more than 13 tiles")
        for tile_id in hand:
            reserve(tile_id)
    for tile_id in front_draws:
        reserve(tile_id)
    if replacement_draw is not None:
        reserve(replacement_draw)

    for hand in hands:
        for tile_id in range(27):
            while len(hand) < 13 and remaining[tile_id] > 0:
                hand.append(tile_id)
                remaining[tile_id] -= 1
        if len(hand) != 13:
            raise AssertionError("Could not fill initial hand")

    wall = []
    for tile_index in range(13):
        for player_id in range(4):
            wall.append(hands[player_id][tile_index])

    wall.extend(front_draws)
    for tile_id, count in enumerate(remaining):
        wall.extend([tile_id] * count)
    if replacement_draw is not None:
        wall.append(replacement_draw)

    if len(wall) != 108:
        raise AssertionError(f"Expected 108 tiles, got {len(wall)}")
    return wall


def submit_pass(arena):
    decision = arena.current_decision()
    arena.submit_action(first_action(decision, ActionType.PASS))


def decline_ready_if_offered(arena):
    if arena.phase == RoundPhase.AWAIT_READY_DECLARATION:
        submit_pass(arena)


def start_daiminkan_wait(front_draws):
    hands = [
        [0],
        [
            0,
            0,
            0,
            1,
            2,
            3,
            9,
            10,
            11,
            18,
            19,
            20,
            26,
        ],
        [],
        [26],
    ]
    wall = scripted_wall(hands, front_draws, replacement_draw=8)
    arena = Arena.from_tiles(wall)
    start_round_passing_initial_ready(arena)

    arena.submit_action(
        first_action(arena.current_decision(), ActionType.DISCARD, 0)
    )
    decline_ready_if_offered(arena)

    decision = arena.current_decision()
    if decision.player != 1:
        raise AssertionError("P1 should respond first")
    arena.submit_action(first_action(decision, ActionType.DAIMINKAN, 0))
    submit_pass(arena)
    submit_pass(arena)

    melds = arena.player(1).melds
    if len(melds) != 1 or melds[0].type != MeldType.DAIMINKAN:
        raise AssertionError("P1 should have one exposed Kan")
    if len(arena.discard_history) != 1:
        raise AssertionError("The called discard should remain in global history")
    if not arena.discard_history[0].was_called:
        raise AssertionError("Daiminkan should mark its discard as called")

    arena.submit_action(
        first_action(arena.current_decision(), ActionType.DISCARD, 8)
    )
    pass_all_responses(arena)
    return arena


def advance_p2_then_p3(arena):
    if arena.current_player != 2:
        raise AssertionError("P2 should draw after P1 discards")
    arena.submit_action(
        first_action(arena.current_decision(), ActionType.DISCARD, 24)
    )
    pass_all_responses(arena)

    if arena.current_player != 3:
        raise AssertionError("P3 should draw after P2 discards")


class ArenaDriverTest(unittest.TestCase):
    def test_global_discard_history_uses_relative_player_views(self):
        hands = [[0], [0, 0, 7], [], []]
        arena = Arena.from_tiles(scripted_wall(hands, [25]))
        start_round_passing_initial_ready(arena)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 0)
        )
        decline_ready_if_offered(arena)

        self.assertEqual(len(arena.discard_history), 1)
        self.assertEqual(arena.discard_history[0].actor, 0)
        self.assertFalse(arena.discard_history[0].was_called)

        expected = [0, 3, 2, 1]
        for observer, relative_actor in enumerate(expected):
            observed = arena.discard_history_for(observer)
            self.assertEqual(observed[0].relative_actor, relative_actor)
            self.assertEqual(observed[0].tile.id, 0)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.PON, 0)
        )
        submit_pass(arena)
        submit_pass(arena)

        self.assertTrue(arena.discard_history[0].was_called)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 7)
        )
        self.assertEqual(len(arena.discard_history), 2)
        self.assertEqual(arena.discard_history[1].actor, 1)
        self.assertFalse(arena.discard_history[1].was_called)

        p3_view = arena.discard_history_for(3)
        self.assertEqual(
            [event.relative_actor for event in p3_view],
            [1, 2],
        )

    def test_four_players_standard_tsumo(self):
        hands = [
            [],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                9,
                10,
                11,
                18,
                19,
                20,
                26,
            ],
            [],
            [],
        ]
        arena = Arena.from_tiles(scripted_wall(hands, [18, 26]))
        start_round_passing_initial_ready(arena)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 18)
        )
        pass_all_responses(arena)

        self.assertEqual(arena.current_player, 1)
        self.assertEqual(arena.player(1).shanten, -1)
        arena.submit_action(
            first_action(arena.current_decision(), ActionType.TSUMO, 26)
        )

        self.assertTrue(arena.finished)
        self.assertEqual(arena.result.reason, RoundEndReason.TSUMO)
        self.assertEqual(arena.result.hu_players, [1])
        chickens = arena.chicken_result
        self.assertIsNotNone(chickens.indicator)
        self.assertIsNotNone(chickens.flipped_chicken)
        self.assertEqual(
            chickens.flipped_chicken.suit,
            chickens.indicator.suit,
        )
        self.assertEqual(
            chickens.flipped_chicken.rank,
            chickens.indicator.rank % 9 + 1,
        )
        self.assertEqual(len(chickens.players), 4)
        self.assertTrue(chickens.player(1).is_jiao)

    def test_four_players_exhaustive_draw(self):
        arena = run_passive_game(seed=20260628)

        self.assertTrue(arena.finished)
        self.assertEqual(arena.result.reason, RoundEndReason.EXHAUSTIVE_DRAW)
        self.assertEqual(arena.result.hu_players, [])
        self.assertEqual(arena.wall_remaining, 0)

    def test_four_players_seven_pairs_ron(self):
        hands = [
            [0],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            [],
            [],
        ]
        arena = Arena.from_tiles(scripted_wall(hands, [25]))
        start_round_passing_initial_ready(arena)
        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 0)
        )
        decline_ready_if_offered(arena)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.RON, 0)
        )
        submit_pass(arena)
        submit_pass(arena)

        self.assertTrue(arena.finished)
        self.assertEqual(arena.result.reason, RoundEndReason.RON)
        self.assertEqual(arena.result.hu_players, [1])
        self.assertEqual(arena.result.fangpao_player, 0)
        self.assertEqual(arena.result.hu_tile.id, 0)

    def test_daiminkan_player_rons_opposite_player(self):
        arena = start_daiminkan_wait([25, 24, 23])
        advance_p2_then_p3(arena)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 26)
        )
        decline_ready_if_offered(arena)

        submit_pass(arena)
        decision = arena.current_decision()
        self.assertEqual(decision.player, 1)
        arena.submit_action(first_action(decision, ActionType.RON, 26))
        submit_pass(arena)

        self.assertTrue(arena.finished)
        self.assertEqual(arena.result.reason, RoundEndReason.RON)
        self.assertEqual(arena.result.hu_players, [1])
        self.assertEqual(arena.result.fangpao_player, 3)

    def test_daiminkan_player_declines_ron_then_tsumos(self):
        arena = start_daiminkan_wait([25, 24, 23, 22, 26])
        advance_p2_then_p3(arena)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 26)
        )
        decline_ready_if_offered(arena)

        submit_pass(arena)
        decision = arena.current_decision()
        self.assertEqual(decision.player, 1)
        first_action(decision, ActionType.RON, 26)
        arena.submit_action(first_action(decision, ActionType.PASS))
        submit_pass(arena)

        self.assertEqual(arena.current_player, 0)
        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 22)
        )
        pass_all_responses(arena)

        self.assertEqual(arena.current_player, 1)
        self.assertEqual(arena.player(1).shanten, -1)
        arena.submit_action(
            first_action(arena.current_decision(), ActionType.TSUMO, 26)
        )

        self.assertTrue(arena.finished)
        self.assertEqual(arena.result.reason, RoundEndReason.TSUMO)
        self.assertEqual(arena.result.hu_players, [1])

    def test_last_replacement_draw_still_allows_discard_and_ron(self):
        hands = [
            [0, 0, 0, 26],
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 26],
            [],
            [],
        ]
        wall = scripted_wall(hands, [0])[:54]
        arena = Arena.from_tiles(wall)
        start_round_passing_initial_ready(arena)
        self.assertEqual(arena.wall_remaining, 1)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.ANKAN, 0)
        )

        self.assertEqual(arena.wall_remaining, 0)
        self.assertFalse(arena.finished)
        self.assertEqual(arena.phase, RoundPhase.AWAIT_TURN_ACTION)
        after_kan = arena.current_decision()
        self.assertTrue(after_kan.legal_actions)
        self.assertTrue(
            all(
                action.type == ActionType.DISCARD
                for action in after_kan.legal_actions
            )
        )

        arena.submit_action(
            first_action(after_kan, ActionType.DISCARD, 26)
        )
        self.assertFalse(arena.finished)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.RON, 26)
        )
        submit_pass(arena)
        submit_pass(arena)

        self.assertEqual(arena.result.reason, RoundEndReason.RON)
        self.assertEqual(arena.result.hu_players, [1])
        self.assertEqual(arena.result.fangpao_player, 0)

    def test_ron_has_priority_over_pon(self):
        hands = [
            [0],
            [0, 0],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            [],
        ]
        arena = Arena.from_tiles(scripted_wall(hands, [25]))
        start_round_passing_initial_ready(arena)
        arena.submit_action(
            first_action(arena.current_decision(), ActionType.DISCARD, 0)
        )
        decline_ready_if_offered(arena)

        arena.submit_action(
            first_action(arena.current_decision(), ActionType.PON, 0)
        )
        arena.submit_action(
            first_action(arena.current_decision(), ActionType.RON, 0)
        )
        submit_pass(arena)

        self.assertEqual(arena.result.reason, RoundEndReason.RON)
        self.assertEqual(arena.result.hu_players, [2])
        self.assertEqual(arena.result.fangpao_player, 0)
        self.assertEqual(arena.player(1).melds, [])
        self.assertFalse(arena.discard_history[0].was_called)


if __name__ == "__main__":
    unittest.main()
