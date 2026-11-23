#include "privileged_state_encoder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

std::vector<GYMJ::Tile> ordered_standard_wall() {
    std::vector<GYMJ::Tile> tiles;
    for (uint8_t tile_id = 0; tile_id < GYMJ::TILE_TYPE_COUNT; ++tile_id) {
        for (int copy = 0; copy < 4; ++copy) {
            tiles.push_back(GYMJ::Tile{tile_id});
        }
    }
    return tiles;
}

std::vector<GYMJ::Tile> wall_with_initial_ready_player() {
    using namespace GYMJ;

    std::array<int, TILE_TYPE_COUNT> copies{};
    copies.fill(MAX_TILE_COPIES);
    std::array<std::vector<Tile>, PLAYER_COUNT> hands;
    const std::array<uint8_t, 13> ready_hand{
        0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 10,
    };
    for (uint8_t tile : ready_hand) {
        hands[1].push_back(Tile{tile});
        --copies[tile];
    }
    for (uint8_t player : {uint8_t{0}, uint8_t{2}, uint8_t{3}}) {
        for (uint8_t tile = 0;
             tile < TILE_TYPE_COUNT && hands[player].size() < 13;
             ++tile) {
            while (copies[tile] > 0 && hands[player].size() < 13) {
                hands[player].push_back(Tile{tile});
                --copies[tile];
            }
        }
        assert(hands[player].size() == 13);
    }

    std::vector<Tile> wall;
    wall.reserve(TILE_TYPE_COUNT * MAX_TILE_COPIES);
    for (size_t tile_index = 0; tile_index < 13; ++tile_index) {
        for (uint8_t player = 0; player < PLAYER_COUNT; ++player) {
            wall.push_back(hands[player][tile_index]);
        }
    }
    for (uint8_t tile = 0; tile < TILE_TYPE_COUNT; ++tile) {
        while (copies[tile] > 0) {
            wall.push_back(Tile{tile});
            --copies[tile];
        }
    }
    assert(wall.size() == TILE_TYPE_COUNT * MAX_TILE_COPIES);
    return wall;
}

GYMJ::Action find_action(
    const GYMJ::DecisionContext& decision,
    GYMJ::ActionType type
) {
    const auto found = std::find_if(
        decision.legal_actions.begin(),
        decision.legal_actions.end(),
        [type](const GYMJ::Action& action) { return action.type == type; }
    );
    assert(found != decision.legal_actions.end());
    return *found;
}

void pass_initial_ready(GYMJ::Arena& arena) {
    arena.start_round();
    while (arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(), GYMJ::ActionType::Pass
        ));
    }
}

void assert_opponent_hands(
    const GYMJ::Arena& arena,
    uint8_t actor,
    const GYMJ::RL::PrivilegedState& state
) {
    for (size_t relative = 1; relative < GYMJ::PLAYER_COUNT; ++relative) {
        const auto& hand = arena.player(static_cast<uint8_t>(
            (actor + relative) % GYMJ::PLAYER_COUNT
        )).tehai;
        for (size_t tile = 0; tile < GYMJ::TILE_TYPE_COUNT; ++tile) {
            assert(state.opponent_hands[relative - 1][tile] == hand[tile]);
        }
    }
}

void assert_player_tenpai(
    const GYMJ::Arena& arena,
    uint8_t actor,
    const GYMJ::RL::PrivilegedState& state
) {
    for (size_t relative = 0; relative < GYMJ::PLAYER_COUNT; ++relative) {
        const auto absolute = static_cast<uint8_t>(
            (actor + relative) % GYMJ::PLAYER_COUNT
        );
        assert(state.player_tenpai[relative] == static_cast<uint8_t>(
            arena.player(absolute).shanten <= 0
        ));
    }
}

void assert_draw_forecast(
    const GYMJ::RL::PrivilegedState& state,
    const std::vector<GYMJ::Tile>& remaining,
    uint8_t actor,
    uint8_t next_draw_player
) {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<size_t, PLAYER_COUNT> counts{};
    const size_t forecast_count = std::min(
        remaining.size(),
        PLAYER_COUNT * PRIVILEGED_FUTURE_DRAW_COUNT
    );
    for (size_t offset = 0; offset < forecast_count; ++offset) {
        const uint8_t absolute = static_cast<uint8_t>(
            (next_draw_player + offset) % PLAYER_COUNT
        );
        const uint8_t relative = static_cast<uint8_t>(
            (absolute - actor + PLAYER_COUNT) % PLAYER_COUNT
        );
        assert(state.future_draw_tiles[relative][counts[relative]++] ==
               remaining[offset].id);
    }
    for (size_t relative = 0; relative < PLAYER_COUNT; ++relative) {
        for (size_t index = counts[relative];
             index < PRIVILEGED_FUTURE_DRAW_COUNT;
             ++index) {
            assert(state.future_draw_tiles[relative][index] ==
                   PRIVILEGED_TILE_PAD);
        }
    }
    const size_t replacement_count = std::min(
        remaining.size(),
        PRIVILEGED_FUTURE_DRAW_COUNT
    );
    for (size_t index = 0; index < replacement_count; ++index) {
        assert(state.replacement_draw_tiles[index] ==
               remaining[remaining.size() - 1 - index].id);
    }
}

void test_snapshot_tracks_decisions_and_wall_order() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    const auto tiles = ordered_standard_wall();
    Arena arena(ArenaConfig{0, 0}, Wall::from_tiles(tiles));
    pass_initial_ready(arena);

    const uint8_t first_actor = arena.current_decision().player;
    assert(first_actor == 0);
    const auto before = PrivilegedStateEncoder::encode(arena, first_actor);
    assert_opponent_hands(arena, first_actor, before);
    assert_player_tenpai(arena, first_actor, before);
    assert(before.wall_count == 55);
    assert(before.wall_tiles.size() == MAX_PRIVILEGED_WALL_TILES);
    for (size_t i = 0; i < before.wall_count; ++i) {
        assert(before.wall_tiles[i] == tiles[53 + i].id);
    }
    assert(before.wall_tiles[before.wall_count] == PRIVILEGED_TILE_PAD);
    assert_draw_forecast(
        before,
        std::vector<Tile>(tiles.begin() + 53, tiles.end()),
        first_actor,
        1
    );

    arena.submit_action(find_action(
        arena.current_decision(), ActionType::Discard
    ));
    for (int response = 0; response < 3; ++response) {
        const auto responder = arena.current_decision().player;
        const auto during = PrivilegedStateEncoder::encode(arena, responder);
        assert_opponent_hands(arena, responder, during);
        assert_player_tenpai(arena, responder, during);
        assert(during.wall_count == before.wall_count);
        assert_draw_forecast(
            during,
            std::vector<Tile>(tiles.begin() + 53, tiles.end()),
            responder,
            1
        );
        arena.submit_action(find_action(
            arena.current_decision(), ActionType::Pass
        ));
    }

    assert(arena.current_decision().player == 1);
    const auto after = PrivilegedStateEncoder::encode(arena, 1);
    assert_opponent_hands(arena, 1, after);
    assert_player_tenpai(arena, 1, after);
    assert(after.wall_count == 54);
    assert(after.wall_tiles[0] == tiles[54].id);
    assert(after.wall_tiles[after.wall_count - 1] == tiles.back().id);
    assert(after.wall_tiles[after.wall_count] == PRIVILEGED_TILE_PAD);
    assert(before.wall_tiles[0] == tiles[53].id);
    assert_draw_forecast(
        after,
        std::vector<Tile>(tiles.begin() + 54, tiles.end()),
        1,
        2
    );
}

void test_rejects_non_decision_snapshots() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{0, 0}, Wall::from_tiles(ordered_standard_wall()));
    bool rejected = false;
    try {
        PrivilegedStateEncoder::encode(arena, 0);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);

    pass_initial_ready(arena);
    rejected = false;
    try {
        PrivilegedStateEncoder::encode(arena, 4);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        PrivilegedStateEncoder::encode(arena, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

void test_initial_ready_forecast_starts_with_dealer_draw() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    const auto tiles = wall_with_initial_ready_player();
    Arena arena(ArenaConfig{0, 0}, Wall::from_tiles(tiles));
    arena.start_round();

    assert(arena.phase() == RoundPhase::AwaitReadyDeclaration);
    const uint8_t actor = arena.current_decision().player;
    assert(actor == 1);
    const auto state = PrivilegedStateEncoder::encode(arena, actor);
    assert(state.wall_count == MAX_PRIVILEGED_WALL_TILES);
    assert_draw_forecast(
        state,
        std::vector<Tile>(tiles.begin() + 52, tiles.end()),
        actor,
        0
    );
}

} // namespace

int main() {
    static_assert(GYMJ::RL::MAX_PRIVILEGED_WALL_TILES == 56);
    static_assert(GYMJ::RL::PRIVILEGED_FUTURE_DRAW_COUNT == 3);
    test_snapshot_tracks_decisions_and_wall_order();
    test_initial_ready_forecast_starts_with_dealer_draw();
    test_rejects_non_decision_snapshots();
    return 0;
}
