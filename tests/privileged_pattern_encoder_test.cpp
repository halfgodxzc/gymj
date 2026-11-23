#include "privileged_pattern_encoder.h"
#include "privileged_state_encoder.h"

#include "shanten.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

std::vector<GYMJ::Tile> scripted_wall(
    std::array<std::vector<uint8_t>, 4> hands,
    const std::vector<uint8_t>& front_draws,
    std::optional<uint8_t> replacement_draw = std::nullopt
) {
    std::array<int, GYMJ::TILE_TYPE_COUNT> remaining;
    remaining.fill(GYMJ::MAX_TILE_COPIES);
    auto reserve = [&](uint8_t tile_id) {
        assert(tile_id < GYMJ::TILE_TYPE_COUNT);
        assert(remaining[tile_id] > 0);
        --remaining[tile_id];
    };
    for (const auto& hand : hands) {
        for (uint8_t tile_id : hand) reserve(tile_id);
    }
    for (uint8_t tile_id : front_draws) reserve(tile_id);
    if (replacement_draw.has_value()) reserve(*replacement_draw);

    for (auto& hand : hands) {
        for (uint8_t tile_id = 0;
             hand.size() < 13 && tile_id < GYMJ::TILE_TYPE_COUNT;
             ++tile_id) {
            while (hand.size() < 13 && remaining[tile_id] > 0) {
                hand.push_back(tile_id);
                --remaining[tile_id];
            }
        }
        assert(hand.size() == 13);
    }

    std::vector<GYMJ::Tile> wall;
    for (size_t index = 0; index < 13; ++index) {
        for (uint8_t player = 0; player < GYMJ::PLAYER_COUNT; ++player) {
            wall.push_back(GYMJ::Tile{hands[player][index]});
        }
    }
    for (uint8_t tile_id : front_draws) wall.push_back({tile_id});
    for (uint8_t tile_id = 0; tile_id < GYMJ::TILE_TYPE_COUNT; ++tile_id) {
        while (remaining[tile_id] > 0) {
            wall.push_back({tile_id});
            --remaining[tile_id];
        }
    }
    if (replacement_draw.has_value()) wall.push_back({*replacement_draw});
    assert(wall.size() == 108);
    return wall;
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

void decline_ready_if_offered(GYMJ::Arena& arena) {
    if (arena.phase() != GYMJ::RoundPhase::AwaitReadyDeclaration) return;
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Pass
    ));
}

void pass_all_discard_responses(GYMJ::Arena& arena) {
    decline_ready_if_offered(arena);
    for (int i = 0; i < 3; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
    decline_ready_if_offered(arena);
}

void advance_to_player_one_kakan(
    GYMJ::Arena& arena,
    uint8_t pon_discard,
    const std::array<uint8_t, 3>& turn_discards
) {
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Pon,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        pon_discard
    ));
    pass_all_discard_responses(arena);

    const std::array<uint8_t, 3> players{2, 3, 0};
    for (size_t i = 0; i < players.size(); ++i) {
        assert(arena.current_player() == players[i]);
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Discard,
            turn_discards[i]
        ));
        pass_all_discard_responses(arena);
    }
    assert(arena.current_player() == 1);
}

} // namespace

void test_privileged_patterns_match_all_true_hands_and_wall() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{70101, 2});
    start_round_passing_initial_ready(arena);
    const uint8_t actor = arena.current_decision().player;
    const auto encoded = PrivilegedPatternEncoder::encode(arena, actor);
    const auto raw = PrivilegedStateEncoder::encode(arena, actor);

    std::array<uint8_t, TILE_TYPE_COUNT> expected_wall_counts{};
    for (size_t index = 0; index < raw.wall_count; ++index) {
        ++expected_wall_counts[raw.wall_tiles[index]];
    }
    assert(encoded.remaining_wall_counts == expected_wall_counts);

    for (uint8_t relative = 0; relative < PLAYER_COUNT; ++relative) {
        const uint8_t absolute = static_cast<uint8_t>(
            (actor + relative) % PLAYER_COUNT
        );
        const PlayerState& player = arena.player(absolute);
        const auto shanten = Algo::calculate_shanten(
            player.tehai,
            static_cast<int>(player.melds[0].size())
        );
        assert(encoded.player_patterns[relative].routes[0].shanten ==
               shanten.normal);
        assert(encoded.player_patterns[relative].routes[1].shanten ==
               shanten.seven_pairs);
        assert(encoded.player_patterns[relative].routes[2].shanten ==
               shanten.all_triplets);
        assert(encoded.true_shape_wait_masks[relative] ==
               encoded.player_patterns[relative].shape_wait_mask);
        const uint8_t chicken_count = static_cast<uint8_t>(
            encoded.economy.settlement_tile_counts[relative]
                [YAO_JI_TILE_ID] +
            encoded.economy.settlement_tile_counts[relative]
                [BA_TONG_TILE_ID]
        );
        const uint8_t expected_tenpai = static_cast<uint8_t>(
            player.shanten <= 0
        );
        assert(encoded.economy.player_tenpai[relative] == expected_tenpai);
        assert(encoded.economy.signed_fixed_chicken_counts[relative] ==
               (expected_tenpai
                    ? static_cast<int8_t>(chicken_count)
                    : static_cast<int8_t>(-static_cast<int>(chicken_count))));
    }
}

void test_responsibility_chicken_moves_to_the_true_owner() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {YAO_JI_TILE_ID};
    hands[1] = {YAO_JI_TILE_ID, YAO_JI_TILE_ID, 7};
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {26}))
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        YAO_JI_TILE_ID
    ));
    if (arena.phase() == RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Pon,
        YAO_JI_TILE_ID
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }

    assert(arena.current_decision().player == 1);
    const auto encoded = PrivilegedPatternEncoder::encode(arena, 1);
    assert(encoded.economy.responsibility_chickens[3][0] == 1);
    assert(encoded.economy.chongfeng_tile_masks[3] == 0);
    assert(encoded.economy.settlement_tile_counts[3][YAO_JI_TILE_ID] == 0);
    assert(encoded.economy.settlement_tile_counts[0][YAO_JI_TILE_ID] == 3);
    assert(encoded.progress.opening_hand_changed[0] == 1);
    assert(encoded.progress.must_discard_after_call == 1);
}

void test_daiminkan_records_owner_source_and_replacement_context() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {0, 0, 0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 26};
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {25}, 24))
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        0
    ));
    if (arena.phase() == RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Daiminkan,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }

    assert(arena.current_decision().player == 1);
    const auto encoded = PrivilegedPatternEncoder::encode(arena, 1);
    assert(encoded.economy.bean_source_counts[0][3][2] == 1);
    assert(encoded.economy.settlement_tile_counts[0][0] == 4);
    assert(encoded.progress.last_draw_kind == DecisionDrawKind::Replacement);
    assert(encoded.progress.hot_discard_player == 0);
    assert(encoded.progress.must_discard_after_call == 0);
}

void test_ankan_uses_owner_as_the_payer_source() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {
        0, 0, 0, 0,
        1, 2, 3,
        9, 10, 11,
        18, 19, 20,
    };
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {26}, 25))
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::AnKan,
        0
    ));

    const auto encoded = PrivilegedPatternEncoder::encode(arena, 0);
    assert(encoded.economy.bean_source_counts[0][0][0] == 1);
    for (uint8_t source = 1; source < PLAYER_COUNT; ++source) {
        assert(encoded.economy.bean_source_counts[0][source][0] == 0);
    }
    assert(encoded.economy.settlement_tile_counts[0][0] == 4);
    assert(encoded.progress.last_draw_kind ==
           DecisionDrawKind::Replacement);
}

void test_kakan_uses_owner_instead_of_the_original_pon_source() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    const std::array<std::vector<uint8_t>, 4> hands{{
        {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},
        {0, 0, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8},
        {9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13},
        {14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18},
    }};
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(
            hands,
            {8, 19, 20, 21, 0},
            22
        ))
    );
    advance_to_player_one_kakan(arena, 8, {19, 20, 21});
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Kakan,
        0
    ));
    for (int i = 0; i < 3; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }

    assert(arena.current_decision().player == 1);
    const auto encoded = PrivilegedPatternEncoder::encode(arena, 1);
    assert(encoded.economy.bean_source_counts[0][0][1] == 1);
    assert(encoded.economy.bean_source_counts[0][3][1] == 0);
    assert(encoded.economy.settlement_tile_counts[0][0] == 4);
    assert(encoded.progress.last_draw_kind ==
           DecisionDrawKind::Replacement);
}

void test_wrong_actor_is_rejected() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{80008, 0});
    start_round_passing_initial_ready(arena);
    const uint8_t actor = arena.current_decision().player;
    bool wrong_turn_threw = false;
    try {
        PrivilegedPatternEncoder::encode(
            arena,
            static_cast<uint8_t>((actor + 1) % PLAYER_COUNT)
        );
    } catch (const std::invalid_argument&) {
        wrong_turn_threw = true;
    }
    assert(wrong_turn_threw);

    bool range_threw = false;
    try {
        PrivilegedPatternEncoder::encode(arena, PLAYER_COUNT);
    } catch (const std::out_of_range&) {
        range_threw = true;
    }
    assert(range_threw);
}

int main() {
    static_assert(GYMJ::RL::PRIVILEGED_PATTERN_VERSION == 2);
    test_privileged_patterns_match_all_true_hands_and_wall();
    test_responsibility_chicken_moves_to_the_true_owner();
    test_daiminkan_records_owner_source_and_replacement_context();
    test_ankan_uses_owner_as_the_payer_source();
    test_kakan_uses_owner_instead_of_the_original_pon_source();
    test_wrong_actor_is_rejected();
    return 0;
}
