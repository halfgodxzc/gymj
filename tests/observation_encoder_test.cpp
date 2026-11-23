#include "observation_encoder.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

std::vector<GYMJ::Tile> ordered_standard_wall() {
    std::vector<GYMJ::Tile> tiles;
    for (uint8_t tile_id = 0; tile_id < 27; ++tile_id) {
        for (int copy = 0; copy < 4; ++copy) {
            tiles.push_back(GYMJ::Tile{tile_id});
        }
    }
    return tiles;
}

GYMJ::Action find_action(
    const GYMJ::DecisionContext& decision,
    GYMJ::ActionType type,
    int tile_id = -1
) {
    const auto found = std::find_if(
        decision.legal_actions.begin(),
        decision.legal_actions.end(),
        [&](const GYMJ::Action& action) {
            return action.type == type &&
                   (tile_id < 0 || action.tile.id == tile_id);
        }
    );
    assert(found != decision.legal_actions.end());
    return *found;
}

void start_round_passing_initial_ready(GYMJ::Arena& arena) {
    arena.start_round();
    while (arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
}

int mask_count(const std::array<uint64_t, 2>& mask) {
    int count = 0;
    for (size_t action_id = 0; action_id < GYMJ::RL::ACTION_SPACE_SIZE; ++action_id) {
        const size_t word = action_id / 64;
        const size_t bit = action_id % 64;
        count += static_cast<int>((mask[word] >> bit) & 1);
    }
    return count;
}

} // namespace

void test_initial_observations_are_private_and_fixed_size() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{123456, 0});
    start_round_passing_initial_ready(arena);
    const auto decision = arena.current_decision();

    for (uint8_t observer = 0; observer < 4; ++observer) {
        const auto observation = ObservationEncoder::encode(arena, observer);
        const auto& player = arena.player(observer);

        assert(observation.hand_counts.size() == 27);
        assert(observation.discards.size() == MAX_DISCARD_EVENTS);
        assert(observation.melds.size() == MAX_MELDS);
        assert(observation.discard_count == 0);
        assert(observation.meld_count == 0);
        assert(observation.phase == RoundPhase::AwaitTurnAction);
        assert(observation.current_actor ==
               static_cast<uint8_t>((decision.player - observer + 4) % 4));
        assert(observation.current_discard == NO_TILE);
        assert(observation.current_discard_actor == NO_PLAYER);
        assert(observation.dealer ==
               static_cast<uint8_t>((arena.dealer() - observer + 4) % 4));
        assert(observation.wall_remaining == 55);

        for (uint8_t tile_id = 0; tile_id < 27; ++tile_id) {
            assert(observation.hand_counts[tile_id] == player.tehai[tile_id]);
            assert(observation.visible_counts[tile_id] == player.tehai[tile_id]);
            const bool waiting =
                (observation.waits & (uint32_t{1} << tile_id)) != 0;
            assert(waiting == player.waits[tile_id]);
        }
        assert(observation.shanten == player.shanten);

        if (observer == decision.player) {
            assert(mask_count(observation.legal_mask) ==
                   static_cast<int>(decision.legal_actions.size()));
        } else {
            assert(mask_count(observation.legal_mask) == 0);
        }
    }
}

void test_discard_context_rotates_for_each_observer() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(ordered_standard_wall())
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        0
    ));

    const uint8_t expected_discard_actor[4] = {0, 3, 2, 1};
    const uint8_t expected_current_actor[4] = {1, 0, 3, 2};
    for (uint8_t observer = 0; observer < 4; ++observer) {
        const auto observation = ObservationEncoder::encode(arena, observer);
        assert(observation.phase == RoundPhase::AwaitDiscardResponses);
        assert(observation.discard_count == 1);
        assert(observation.discards[0].tile == 0);
        assert(observation.discards[0].relative_actor ==
               expected_discard_actor[observer]);
        assert(observation.discards[0].was_called == 0);
        assert(observation.discards[1].tile == NO_TILE);
        assert(observation.current_discard == 0);
        assert(observation.current_discard_actor ==
               expected_discard_actor[observer]);
        assert(observation.current_actor ==
               expected_current_actor[observer]);

        if (observer == 1) {
            assert(mask_count(observation.legal_mask) > 0);
        } else {
            assert(mask_count(observation.legal_mask) == 0);
        }
    }
}

void test_called_discard_and_meld_are_not_double_counted() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    auto tiles = ordered_standard_wall();
    std::swap(tiles[2], tiles[5]);
    Arena arena(ArenaConfig{0, 0}, Wall::from_tiles(tiles));
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        0
    ));

    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Pon,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }

    const auto p3 = ObservationEncoder::encode(arena, 3);
    assert(p3.discard_count == 1);
    assert(p3.discards[0].tile == 0);
    assert(p3.discards[0].relative_actor == 1);
    assert(p3.discards[0].was_called == 1);

    assert(p3.meld_count == 1);
    assert(p3.melds[0].valid == 1);
    assert(p3.melds[0].tile == 0);
    assert(p3.melds[0].relative_owner == 2);
    assert(p3.melds[0].relative_source == 1);
    assert(p3.melds[0].type == MeldType::Pon);
    assert(p3.melds[1].valid == 0);
    assert(p3.melds[1].tile == NO_TILE);

    // P3 holds the fourth copy; the called discard is counted only in the Pon.
    assert(p3.hand_counts[0] == 1);
    assert(p3.visible_counts[0] == 4);
    assert(p3.current_discard == NO_TILE);
    assert(p3.current_discard_actor == NO_PLAYER);

    const auto p2 = ObservationEncoder::encode(arena, 2);
    assert(p2.hand_counts[0] == 0);
    assert(p2.visible_counts[0] == 3);
}

void test_invalid_observer_is_rejected() {
    GYMJ::Arena arena(GYMJ::ArenaConfig{1, 0});
    start_round_passing_initial_ready(arena);

    bool threw = false;
    try {
        GYMJ::RL::ObservationEncoder::encode(arena, 4);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    static_assert(GYMJ::RL::OBSERVATION_VERSION == 3);
    static_assert(GYMJ::RL::MAX_DISCARD_EVENTS == 72);
    static_assert(GYMJ::RL::MAX_MELDS == 16);

    test_initial_observations_are_private_and_fixed_size();
    test_discard_context_rotates_for_each_observer();
    test_called_discard_and_meld_are_not_double_counted();
    test_invalid_observer_is_rejected();
    return 0;
}
