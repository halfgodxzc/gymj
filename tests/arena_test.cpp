#include "arena.h"
#include "observation_encoder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

std::vector<GYMJ::Tile> ordered_standard_wall() {
    std::vector<GYMJ::Tile> tiles;
    for (uint8_t id = 0; id < 27; ++id) {
        for (int copy = 0; copy < 4; ++copy) {
            tiles.push_back(GYMJ::Tile{id});
        }
    }
    return tiles;
}

std::vector<GYMJ::Tile> scripted_wall(
    std::array<std::vector<uint8_t>, 4> hands,
    const std::vector<uint8_t>& front_draws,
    std::optional<uint8_t> replacement_draw = std::nullopt
) {
    std::array<int, 27> remaining;
    remaining.fill(4);

    auto reserve = [&](uint8_t tile_id) {
        assert(tile_id < 27);
        assert(remaining[tile_id] > 0);
        remaining[tile_id]--;
    };

    for (const auto& hand : hands) {
        assert(hand.size() <= 13);
        for (uint8_t tile_id : hand) reserve(tile_id);
    }
    for (uint8_t tile_id : front_draws) reserve(tile_id);
    if (replacement_draw.has_value()) reserve(*replacement_draw);

    for (auto& hand : hands) {
        for (uint8_t tile_id = 0; hand.size() < 13 && tile_id < 27; ++tile_id) {
            while (hand.size() < 13 && remaining[tile_id] > 0) {
                hand.push_back(tile_id);
                remaining[tile_id]--;
            }
        }
        assert(hand.size() == 13);
    }

    std::vector<GYMJ::Tile> wall;
    wall.reserve(108);
    for (size_t tile_index = 0; tile_index < 13; ++tile_index) {
        for (uint8_t player_id = 0; player_id < 4; ++player_id) {
            wall.push_back(GYMJ::Tile{hands[player_id][tile_index]});
        }
    }
    for (uint8_t tile_id : front_draws) {
        wall.push_back(GYMJ::Tile{tile_id});
    }
    for (uint8_t tile_id = 0; tile_id < 27; ++tile_id) {
        while (remaining[tile_id] > 0) {
            wall.push_back(GYMJ::Tile{tile_id});
            remaining[tile_id]--;
        }
    }
    if (replacement_draw.has_value()) {
        wall.push_back(GYMJ::Tile{*replacement_draw});
    }

    assert(wall.size() == 108);
    return wall;
}

GYMJ::Action find_action(
    const GYMJ::DecisionContext& context,
    GYMJ::ActionType type,
    int tile_id = -1
) {
    const auto it = std::find_if(
        context.legal_actions.begin(),
        context.legal_actions.end(),
        [&](const GYMJ::Action& action) {
            return action.type == type &&
                   (tile_id < 0 || action.tile.id == tile_id);
        }
    );
    assert(it != context.legal_actions.end());
    return *it;
}

void decline_ready_if_offered(GYMJ::Arena& arena) {
    if (arena.phase() != GYMJ::RoundPhase::AwaitReadyDeclaration) return;
    const auto ready = arena.current_decision();
    arena.submit_action(find_action(ready, GYMJ::ActionType::Pass));
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

void pass_all_responses(GYMJ::Arena& arena) {
    decline_ready_if_offered(arena);
    for (int i = 0; i < 3; ++i) {
        const auto context = arena.current_decision();
        arena.submit_action(find_action(context, GYMJ::ActionType::Pass));
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
    pass_all_responses(arena);

    const std::array<uint8_t, 3> players{2, 3, 0};
    for (size_t i = 0; i < players.size(); ++i) {
        assert(arena.current_player() == players[i]);
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Discard,
            turn_discards[i]
        ));
        pass_all_responses(arena);
    }

    assert(arena.current_player() == 1);
    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
}

} // namespace

void test_start_deals_and_draws_for_dealer() {
    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(ordered_standard_wall())
    );
    start_round_passing_initial_ready(arena);

    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.current_player() == 0);
    assert(arena.wall_remaining() == 55);

    int dealer_tiles = 0;
    for (uint8_t count : arena.player(0).tehai) dealer_tiles += count;
    assert(dealer_tiles == 14);

    for (uint8_t player_id = 1; player_id < 4; ++player_id) {
        int tiles = 0;
        for (uint8_t count : arena.player(player_id).tehai) tiles += count;
        assert(tiles == 13);
    }
}

void test_discard_passes_and_next_draw() {
    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(ordered_standard_wall())
    );
    start_round_passing_initial_ready(arena);

    const auto turn = arena.current_decision();
    const GYMJ::Action discard = find_action(turn, GYMJ::ActionType::Discard);
    arena.submit_action(discard);
    decline_ready_if_offered(arena);

    assert(arena.phase() == GYMJ::RoundPhase::AwaitDiscardResponses);
    assert(arena.current_decision().player == 1);
    pass_all_responses(arena);

    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.current_player() == 1);
    assert(arena.wall_remaining() == 54);
    assert(arena.player(0).kawa[0].back() == discard.tile);
    assert(arena.player(2).kawa[2].back() == discard.tile);
}

void test_pon_takes_response_and_caller_must_discard() {
    auto tiles = ordered_standard_wall();
    std::swap(tiles[2], tiles[5]);

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pon, 0));

    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.current_player() == 1);
    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.player(1).melds[0].size() == 1);

    const auto after_pon = arena.current_decision();
    assert(!after_pon.legal_actions.empty());
    for (const auto& action : after_pon.legal_actions) {
        assert(action.type == GYMJ::ActionType::Discard);
    }
}

void test_global_discard_history_converts_to_player_views() {
    auto tiles = ordered_standard_wall();
    std::swap(tiles[2], tiles[5]);

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);

    assert(arena.discard_history().size() == 1);
    assert(arena.discard_history()[0].actor == 0);
    assert(arena.discard_history()[0].tile.id == 0);
    assert(!arena.discard_history()[0].was_called);

    const uint8_t expected_relative_actors[4] = {0, 3, 2, 1};
    for (uint8_t observer = 0; observer < 4; ++observer) {
        const auto observed = arena.discard_history_for(observer);
        assert(observed.size() == 1);
        assert(observed[0].relative_actor == expected_relative_actors[observer]);
        assert(observed[0].tile.id == 0);
        assert(!observed[0].was_called);
    }

    auto response = arena.current_decision();
    arena.submit_action(find_action(response, GYMJ::ActionType::Pon, 0));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.discard_history().size() == 1);
    assert(arena.discard_history()[0].was_called);

    const auto next_discard = find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard
    );
    arena.submit_action(next_discard);

    assert(arena.discard_history().size() == 2);
    assert(arena.discard_history()[0].was_called);
    assert(arena.discard_history()[1].actor == 1);
    assert(arena.discard_history()[1].tile == next_discard.tile);
    assert(!arena.discard_history()[1].was_called);

    const auto p3_view = arena.discard_history_for(3);
    assert(p3_view[0].relative_actor == 1);
    assert(p3_view[1].relative_actor == 2);
}

void test_illegal_action_is_rejected() {
    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(ordered_standard_wall())
    );
    start_round_passing_initial_ready(arena);

    bool threw = false;
    try {
        arena.submit_action({
            GYMJ::ActionType::Ron,
            1,
            GYMJ::Tile{26},
            0,
        });
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_unreported_dihu_after_pon_tracks_responsibility_chicken() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {GYMJ::YAO_JI_TILE_ID};
    hands[1] = {GYMJ::YAO_JI_TILE_ID, GYMJ::YAO_JI_TILE_ID, 7};
    hands[2] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    const auto tiles = scripted_wall(hands, {25, 26});

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        GYMJ::YAO_JI_TILE_ID
    ));
    decline_ready_if_offered(arena);

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(
        response,
        GYMJ::ActionType::Pon,
        GYMJ::YAO_JI_TILE_ID
    ));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.current_player() == 1);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        7
    ));
    pass_all_responses(arena);

    assert(arena.current_player() == 2);
    assert(arena.player(2).shanten == -1);
    const size_t wall_before_hu = arena.wall_remaining();
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Tsumo,
        26
    ));

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::Tsumo);
    assert(arena.result().hu_players == std::vector<uint8_t>{2});
    assert(arena.result().hu_tile.id == 26);
    const auto& settlement = arena.settlement_snapshot();
    assert(settlement.player(2).hand_count(GYMJ::Tile{26}) == 2);
    const auto& chickens = arena.chicken_result();
    assert(chickens.indicator.has_value());
    assert(chickens.flipped_chicken.has_value());
    assert((std::array<int, GYMJ::PLAYER_COUNT>{
        chickens.player(0).total_points,
        chickens.player(1).total_points,
        chickens.player(2).total_points,
        chickens.player(3).total_points,
    } == std::array<int, GYMJ::PLAYER_COUNT>{0, 6, 2, 0}));
    assert(arena.discard_history().front().called_by == 1);
    assert(chickens.responsibility_chickens.size() == 1);
    assert(chickens.responsibility_chickens.front().source_player == 0);
    assert(chickens.responsibility_chickens.front().owner_player == 1);
    const auto& round_settlement = arena.round_settlement();
    assert(round_settlement.hu_settlements.size() == 1);
    assert(round_settlement.hu_settlements.front().player_id == 2);
    assert(round_settlement.hu_settlements.front().features.ready_kind ==
           GYMJ::ReadyKind::None);
    assert(round_settlement.hu_settlements.front().features.timing ==
           GYMJ::HuTiming::DiHu);
    assert(round_settlement.hu_settlements.front().score.total_fan == 6);
    assert(round_settlement.hu_settlements.front().score.hu_points == 30);
    assert((round_settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{-43, -9, 90, -38}));
    assert(round_settlement.ledger.net_total() == 0);
    assert(arena.wall_remaining() == wall_before_hu - 1);
}

void test_daiminkan_replacement_draw_gangkai_settles_all_players() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {
        0, 0, 0,
        1, 2, 3,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    const auto tiles = scripted_wall(hands, {25}, 26);

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Daiminkan, 0));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.current_player() == 1);
    assert(arena.player(1).melds[0].size() == 1);
    assert(arena.player(1).melds[0][0].type == GYMJ::MeldType::Daiminkan);
    assert(arena.discard_history().size() == 1);
    assert(arena.discard_history()[0].was_called);
    assert(arena.player(1).tehai[26] == 2);
    assert(arena.player(1).shanten == -1);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Tsumo,
        26
    ));
    assert(arena.result().reason == GYMJ::RoundEndReason::Tsumo);
    assert(arena.result().hu_players == std::vector<uint8_t>{1});
    const auto& round_settlement = arena.round_settlement();
    assert(round_settlement.hu_settlements.size() == 1);
    assert(round_settlement.hu_settlements.front().features.special ==
           GYMJ::SpecialHu::GangKai);
    assert(round_settlement.hu_settlements.front().score.total_fan == 2);
    assert(round_settlement.hu_settlements.front().score.hu_points == 10);
    assert((std::array<int, GYMJ::PLAYER_COUNT>{
        arena.chicken_result().player(0).total_points,
        arena.chicken_result().player(1).total_points,
        arena.chicken_result().player(2).total_points,
        arena.chicken_result().player(3).total_points,
    } == std::array<int, GYMJ::PLAYER_COUNT>{0, 2, 0, 0}));
    assert(round_settlement.beans.items.size() == 1);
    assert(round_settlement.beans.items.front().kind ==
           GYMJ::BeanKind::Daiminkan);
    assert(round_settlement.beans.items.front().owner == 1);
    assert(round_settlement.beans.items.front().source == 0);
    assert((round_settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{-17, 41, -12, -12}));
    assert(round_settlement.ledger.net_total() == 0);
}

void test_qing_qidui_ron_settles_only_with_discarder() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {
        0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6,
    };
    const auto tiles = scripted_wall(hands, {25});

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Ron, 0));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::Ron);
    assert(arena.result().hu_players == std::vector<uint8_t>{1});
    assert(arena.result().fangpao_player == 0);
    assert(arena.result().hu_tile.id == 0);
    const auto& round_settlement = arena.round_settlement();
    assert(round_settlement.hu_settlements.size() == 1);
    assert(round_settlement.hu_settlements.front().features.pattern ==
           GYMJ::HuPattern::SevenPairs);
    assert(round_settlement.hu_settlements.front().features.is_flush);
    assert(round_settlement.hu_settlements.front().score.total_fan == 6);
    assert(round_settlement.hu_settlements.front().score.hu_points == 30);
    assert((std::array<int, GYMJ::PLAYER_COUNT>{
        arena.chicken_result().player(0).total_points,
        arena.chicken_result().player(1).total_points,
        arena.chicken_result().player(2).total_points,
        arena.chicken_result().player(3).total_points,
    } == std::array<int, GYMJ::PLAYER_COUNT>{0, 0, 0, 0}));
    assert(round_settlement.beans.items.empty());
    assert((round_settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{-30, 30, 0, 0}));
    assert(round_settlement.ledger.net_total() == 0);
}

void test_last_replacement_draw_allows_repao_and_settles_reverse_bean() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0, 0, 0, 26};
    hands[1] = {
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6,
        26,
    };
    auto tiles = scripted_wall(hands, {0});
    tiles.resize(54);

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);
    assert(arena.wall_remaining() == 1);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::AnKan,
        0
    ));

    assert(arena.wall_remaining() == 0);
    assert(!arena.is_finished());
    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.current_player() == 0);
    const auto after_kan = arena.current_decision();
    assert(!after_kan.legal_actions.empty());
    for (const auto& action : after_kan.legal_actions) {
        assert(action.type == GYMJ::ActionType::Discard);
    }

    arena.submit_action(find_action(
        after_kan,
        GYMJ::ActionType::Discard,
        26
    ));
    assert(!arena.is_finished());
    assert(arena.phase() == GYMJ::RoundPhase::AwaitDiscardResponses);

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Ron, 26));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::Ron);
    assert(arena.result().hu_players == std::vector<uint8_t>{1});
    assert(arena.result().fangpao_player == 0);
    assert(arena.result().is_hot_cannon);
    assert(arena.chicken_result().player(0).is_burned);
    assert(!arena.chicken_result().indicator.has_value());
    assert(!arena.chicken_result().flipped_chicken.has_value());
    assert((std::array<int, GYMJ::PLAYER_COUNT>{
        arena.chicken_result().player(0).total_points,
        arena.chicken_result().player(1).total_points,
        arena.chicken_result().player(2).total_points,
        arena.chicken_result().player(3).total_points,
    } == std::array<int, GYMJ::PLAYER_COUNT>{0, 0, 0, 0}));

    const auto& round_settlement = arena.round_settlement();
    assert(round_settlement.hu_settlements.size() == 1);
    assert(round_settlement.hu_settlements.front().features.pattern ==
           GYMJ::HuPattern::SevenPairs);
    assert(round_settlement.hu_settlements.front().features.special ==
           GYMJ::SpecialHu::RePao);
    assert(round_settlement.hu_settlements.front().score.total_fan == 5);
    assert(round_settlement.hu_settlements.front().score.hu_points == 25);
    assert(round_settlement.beans.items.size() == 1);
    assert(round_settlement.beans.items.front().kind ==
           GYMJ::BeanKind::AnKan);
    assert(round_settlement.beans.items.front().owner == 0);
    assert(!round_settlement.beans.items.front().owner_is_jiao);
    assert((round_settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{-40, 30, 5, 5}));
    assert(round_settlement.ledger.net_total() == 0);
}

void test_last_live_draw_does_not_offer_ankan_or_kakan() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0, 0, 5, 5, 5, 5, 6};
    hands[3] = {0};
    auto tiles = scripted_wall(hands, {26, 25, 24, 23, 0});
    tiles.resize(57);

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 3},
        GYMJ::Wall::from_tiles(tiles)
    );
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
        6
    ));
    pass_all_responses(arena);

    const std::array<uint8_t, 3> drawn_tiles{25, 24, 23};
    for (uint8_t tile_id : drawn_tiles) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Discard,
            tile_id
        ));
        pass_all_responses(arena);
    }

    assert(arena.current_player() == 0);
    assert(arena.wall_remaining() == 0);
    const auto decision = arena.current_decision();
    assert(std::none_of(
        decision.legal_actions.begin(),
        decision.legal_actions.end(),
        [](const GYMJ::Action& action) {
            return action.type == GYMJ::ActionType::AnKan ||
                   action.type == GYMJ::ActionType::Kakan;
        }
    ));
}

void test_last_discard_allows_only_ron_or_pass() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {0, 0, 0};
    auto tiles = scripted_wall(hands, {26});
    tiles.resize(53);

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);
    assert(arena.wall_remaining() == 0);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);

    const auto response = arena.current_decision();
    assert(response.player == 1);
    assert(std::all_of(
        response.legal_actions.begin(),
        response.legal_actions.end(),
        [](const GYMJ::Action& action) {
            return action.type == GYMJ::ActionType::Ron ||
                   action.type == GYMJ::ActionType::Pass;
        }
    ));
    assert(std::none_of(
        response.legal_actions.begin(),
        response.legal_actions.end(),
        [](const GYMJ::Action& action) {
            return action.type == GYMJ::ActionType::Pon ||
                   action.type == GYMJ::ActionType::Daiminkan;
        }
    ));
}

void test_repao_burns_only_source_chickens_and_settles_all_players() {
    const std::array<std::vector<uint8_t>, 4> hands{{
        {
            0, 0, 0, 0,
            3, 4, 5,
            9, 10, 11,
            18, 19, 20,
        },
        {
            1, 2, 3,
            4, 5, 6,
            9, 10, 11,
            12, 13, 14,
            26,
        },
        {1, 1, 1, 2, 2, 2, 6, 6, 6, 12, 12, 12, 13},
        {14, 14, 14, 15, 15, 15, 17, 17, 17, 21, 21, 21, 22},
    }};
    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(scripted_wall(
            hands,
            {26, 7},
            25
        ))
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::AnKan,
        0
    ));
    assert(arena.current_player() == 0);
    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        26
    ));
    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Ron, 26));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::Ron);
    assert(arena.result().hu_players == std::vector<uint8_t>{1});
    assert(arena.result().fangpao_player == 0);
    assert(arena.result().is_hot_cannon);

    const auto& chickens = arena.chicken_result();
    assert(chickens.indicator.has_value());
    assert(chickens.indicator->id == 7);
    assert(chickens.flipped_chicken.has_value());
    assert(chickens.flipped_chicken->id == 8);
    assert(chickens.player(0).is_burned);
    assert(chickens.player(0).raw_total_points == 2);
    assert(chickens.player(0).total_points == 0);
    for (uint8_t player_id = 1;
         player_id < GYMJ::PLAYER_COUNT;
         ++player_id) {
        assert(!chickens.player(player_id).is_burned);
        assert(chickens.player(player_id).total_points == 0);
    }

    const auto& settlement = arena.round_settlement();
    assert(settlement.hu_settlements.size() == 1);
    assert(settlement.hu_settlements.front().features.pattern ==
           GYMJ::HuPattern::Standard);
    assert(settlement.hu_settlements.front().features.special ==
           GYMJ::SpecialHu::RePao);
    assert(settlement.hu_settlements.front().score.total_fan == 2);
    assert(settlement.hu_settlements.front().score.hu_points == 10);
    assert(settlement.beans.items.size() == 1);
    assert(settlement.beans.items.front().kind == GYMJ::BeanKind::AnKan);
    assert(settlement.beans.items.front().owner == 0);
    assert(settlement.beans.items.front().owner_is_jiao);
    assert((settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{5, 5, -5, -5}));
    assert(settlement.ledger.net_total() == 0);
}

void test_ron_has_priority_over_pon() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {0, 0};
    hands[2] = {
        1, 2,
        1, 2, 3,
        3, 4, 5,
        5, 6, 7,
        8, 8,
    };
    const auto tiles = scripted_wall(hands, {25});

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pon, 0));

    response = arena.current_decision();
    assert(response.player == 2);
    arena.submit_action(find_action(response, GYMJ::ActionType::Ron, 0));

    response = arena.current_decision();
    assert(response.player == 3);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pass));

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::Ron);
    assert(arena.result().hu_players == std::vector<uint8_t>{2});
    assert(arena.result().fangpao_player == 0);
    const auto& player_one_melds = arena.player(1).melds[0];
    assert(std::none_of(
        player_one_melds.begin(),
        player_one_melds.end(),
        [](const GYMJ::Meld& meld) {
            return meld.type == GYMJ::MeldType::Pon &&
                   meld.tile.id == 0 &&
                   meld.source == 0;
        }
    ));
    assert(!arena.discard_history()[0].was_called);
    const auto& settlement = arena.settlement_snapshot();
    assert(settlement.player(0).discard_count(GYMJ::Tile{0}) == 0);
    assert(settlement.player(2).hand_count(GYMJ::Tile{0}) == 1);
    const auto& round_settlement = arena.round_settlement();
    assert(round_settlement.hu_settlements.size() == 1);
    assert(round_settlement.hu_settlements.front().features.pattern ==
           GYMJ::HuPattern::Standard);
    assert(round_settlement.hu_settlements.front().features.is_flush);
    assert(!round_settlement.hu_settlements.front().features.is_single_wait);
    assert(round_settlement.hu_settlements.front().score.total_fan == 3);
    assert(round_settlement.hu_settlements.front().score.hu_points == 15);
    assert((std::array<int, GYMJ::PLAYER_COUNT>{
        arena.chicken_result().player(0).total_points,
        arena.chicken_result().player(1).total_points,
        arena.chicken_result().player(2).total_points,
        arena.chicken_result().player(3).total_points,
    } == std::array<int, GYMJ::PLAYER_COUNT>{0, 0, 0, 0}));
    assert(round_settlement.beans.items.empty());
    assert((round_settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{-15, 0, 15, 0}));
    assert(round_settlement.ledger.net_total() == 0);
}

void test_tianting_player_kills_dealer_baoting_and_settles() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    hands[1] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        7,
    };
    const auto tiles = scripted_wall(hands, {7});

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    arena.start_round();

    assert(arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration);
    assert(arena.wall_remaining() == 56);
    assert(arena.current_decision().player == 1);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::DeclareReady
    ));
    assert(arena.ready_state(1).kind == GYMJ::ReadyKind::TianTing);
    while (arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
    assert(arena.current_player() == 0);
    assert(arena.wall_remaining() == 55);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        7
    ));
    assert(arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration);
    assert(arena.current_decision().player == 0);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::DeclareReady
    ));
    assert(arena.ready_state(0).kind == GYMJ::ReadyKind::BaoTing);
    assert(arena.phase() == GYMJ::RoundPhase::AwaitDiscardResponses);

    auto response = arena.current_decision();
    assert(response.player == 1);
    const auto ron = find_action(response, GYMJ::ActionType::Ron, 7);
    arena.submit_action(ron);
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::Ron);
    assert(arena.result().hu_players == std::vector<uint8_t>{1});
    assert(arena.result().fangpao_player == 0);
    assert(arena.result().hu_tile.id == 7);
    const auto& round_settlement = arena.round_settlement();
    assert(round_settlement.hu_settlements.size() == 1);
    assert(round_settlement.hu_settlements.front().features.ready_kind ==
           GYMJ::ReadyKind::TianTing);
    assert(round_settlement.hu_settlements.front().features.special ==
           GYMJ::SpecialHu::ShaBao);
    assert(round_settlement.hu_settlements.front().score.ready_fan == 5);
    assert(round_settlement.hu_settlements.front().score.special_fan == 3);
    assert(round_settlement.hu_settlements.front().score.total_fan == 8);
    assert(round_settlement.hu_settlements.front().score.hu_points == 40);
    assert((std::array<int, GYMJ::PLAYER_COUNT>{
        arena.chicken_result().player(0).total_points,
        arena.chicken_result().player(1).total_points,
        arena.chicken_result().player(2).total_points,
        arena.chicken_result().player(3).total_points,
    } == std::array<int, GYMJ::PLAYER_COUNT>{2, 2, 0, 0}));
    assert(round_settlement.beans.items.empty());
    assert((round_settlement.ledger.score_deltas() ==
            std::array<int, GYMJ::PLAYER_COUNT>{-36, 44, -4, -4}));
    assert(round_settlement.ledger.net_total() == 0);
}

void test_consecutive_pons_do_not_close_untouched_players_ready_window() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {0, 0, 1};
    hands[2] = {1, 1, 2};
    hands[3] = {
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        7, 7,
        23,
        26,
    };
    const auto tiles = scripted_wall(hands, {8, 24});

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    start_round_passing_initial_ready(arena);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    if (arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }

    auto response = arena.current_decision();
    assert(response.player == 1);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pon, 0));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        1
    ));
    response = arena.current_decision();
    assert(response.player == 2);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pon, 1));
    for (int i = 0; i < 2; ++i) {
        response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        2
    ));
    pass_all_responses(arena);

    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.current_player() == 3);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        26
    ));

    assert(arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration);
    assert(arena.current_decision().player == 3);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::DeclareReady
    ));

    assert(arena.ready_state(3).kind == GYMJ::ReadyKind::BaoTing);
    assert((arena.ready_state(3).locked_waits & (uint32_t{1} << 22)) != 0);
    assert((arena.ready_state(3).locked_waits & (uint32_t{1} << 25)) != 0);
    assert(arena.phase() == GYMJ::RoundPhase::AwaitDiscardResponses);
}

void test_tianting_locks_hand_to_the_drawn_tile() {
    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {7};
    hands[1] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    const auto tiles = scripted_wall(hands, {0, 8});

    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(tiles)
    );
    arena.start_round();

    assert(arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration);
    assert(arena.current_decision().player == 1);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::DeclareReady
    ));
    while (arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
    assert(arena.ready_state(1).kind == GYMJ::ReadyKind::TianTing);

    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        7
    ));
    decline_ready_if_offered(arena);

    for (int i = 0; i < 3; ++i) {
        const auto response = arena.current_decision();
        arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    }

    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.current_player() == 1);

    const auto locked_turn = arena.current_decision();
    assert(locked_turn.legal_actions.size() == 1);
    assert(locked_turn.legal_actions[0].type == GYMJ::ActionType::Discard);
    assert(locked_turn.legal_actions[0].tile.id == 8);
}

void test_kakan_always_queries_all_opponents_before_it_applies() {
    const std::array<std::vector<uint8_t>, 4> hands{{
        {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},
        {0, 0, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8},
        {9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13},
        {14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18},
    }};
    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(scripted_wall(
            hands,
            {8, 19, 20, 21, 0}
        ))
    );
    advance_to_player_one_kakan(arena, 8, {19, 20, 21});

    const size_t wall_before = arena.wall_remaining();
    assert(arena.player(1).tehai[0] == 1);
    assert(arena.player(1).melds[0][0].type == GYMJ::MeldType::Pon);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Kakan,
        0
    ));

    assert(arena.phase() == GYMJ::RoundPhase::AwaitKakanResponses);
    assert(arena.wall_remaining() == wall_before);
    assert(arena.player(1).tehai[0] == 1);
    assert(arena.player(1).melds[0][0].type == GYMJ::MeldType::Pon);
    assert(arena.history().back().action.type == GYMJ::ActionType::Kakan);
    assert(!arena.history().back().applied);
    for (uint8_t observer = 0; observer < 4; ++observer) {
        const auto observation = GYMJ::RL::ObservationEncoder::encode(
            arena,
            observer
        );
        assert(observation.visible_counts[0] == 4);
        assert(observation.current_discard == 0);
        assert(observation.current_discard_actor ==
               static_cast<uint8_t>((1 - observer + 4) % 4));
    }

    const std::array<uint8_t, 3> responders{2, 3, 0};
    for (uint8_t responder : responders) {
        const auto decision = arena.current_decision();
        assert(decision.player == responder);
        assert(decision.legal_actions.size() == 1);
        assert(decision.legal_actions[0].type == GYMJ::ActionType::Pass);
        arena.submit_action(decision.legal_actions[0]);
    }

    assert(arena.phase() == GYMJ::RoundPhase::AwaitTurnAction);
    assert(arena.current_player() == 1);
    assert(arena.wall_remaining() == wall_before - 1);
    assert(arena.player(1).tehai[0] == 0);
    assert(arena.player(1).melds[0][0].type == GYMJ::MeldType::Kakan);
    std::vector<const GYMJ::GameEvent*> kakan_events;
    for (const auto& event : arena.history()) {
        if (event.action.type == GYMJ::ActionType::Kakan) {
            kakan_events.push_back(&event);
        }
    }
    assert(kakan_events.size() == 2);
    assert(!kakan_events[0]->applied);
    assert(kakan_events[0]->phase_after ==
           GYMJ::RoundPhase::AwaitKakanResponses);
    assert(kakan_events[1]->applied);
    assert(kakan_events[1]->phase_before ==
           GYMJ::RoundPhase::AwaitKakanResponses);
    assert(kakan_events[1]->phase_after ==
           GYMJ::RoundPhase::ReplacementDraw);
}

void test_qianggang_waits_for_all_responses_and_cancels_kakan() {
    const std::array<std::vector<uint8_t>, 4> hands{{
        {0, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17},
        {0, 0, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24},
        {1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 26, 26},
        {6, 6, 6, 7, 7, 7, 8, 8, 8, 12, 12, 12, 13},
    }};
    GYMJ::Arena arena(
        GYMJ::ArenaConfig{0, 0},
        GYMJ::Wall::from_tiles(scripted_wall(
            hands,
            {25, 6, 13, 24, 0}
        ))
    );
    advance_to_player_one_kakan(arena, 24, {6, 13, 24});

    const size_t wall_before = arena.wall_remaining();
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Kakan,
        0
    ));

    auto response = arena.current_decision();
    assert(response.player == 2);
    arena.submit_action(find_action(response, GYMJ::ActionType::Ron, 0));
    assert(!arena.is_finished());
    assert(arena.phase() == GYMJ::RoundPhase::AwaitKakanResponses);

    response = arena.current_decision();
    assert(response.player == 3);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pass));
    assert(!arena.is_finished());

    response = arena.current_decision();
    assert(response.player == 0);
    arena.submit_action(find_action(response, GYMJ::ActionType::Pass));

    assert(arena.is_finished());
    assert(arena.result().reason == GYMJ::RoundEndReason::QiangGang);
    assert(arena.result().hu_players == std::vector<uint8_t>{2});
    assert(arena.result().fangpao_player == 1);
    assert(arena.result().hu_tile.id == 0);
    assert(arena.wall_remaining() == wall_before - 1);
    assert(arena.player(1).tehai[0] == 1);
    assert(arena.player(1).melds[0][0].type == GYMJ::MeldType::Pon);
    const auto& settlement = arena.settlement_snapshot();
    assert(settlement.player(1).hand_count(GYMJ::Tile{0}) == 0);
    assert(settlement.player(1).meld_tile_count(GYMJ::Tile{0}) == 3);
    assert(settlement.player(2).hand_count(GYMJ::Tile{0}) == 1);
    assert(arena.chicken_result().indicator.has_value());
    for (const auto& bean : arena.round_settlement().beans.items) {
        assert(bean.kind != GYMJ::BeanKind::Kakan);
    }
    assert(arena.round_settlement().ledger.net_total() == 0);
    int kakan_event_count = 0;
    for (const auto& event : arena.history()) {
        if (event.action.type != GYMJ::ActionType::Kakan) continue;
        kakan_event_count++;
        assert(!event.applied);
    }
    assert(kakan_event_count == 1);
}

void test_passed_pon_until_own_draw() {
    using namespace GYMJ;
    // P2 passes P0's tile. P1 either repeats it immediately or keeps it
    // until P2 has drawn; a different pair remains callable throughout.
    for (int scenario = 0; scenario < 3; ++scenario) {
        Arena arena({0, 0}, Wall::from_tiles(scripted_wall(
            {{{25}, {25, 26}, {25, 25, 26, 26}, {}}}, {19, 20, 21, 22, 23, 24}
        )));
        start_round_passing_initial_ready(arena);
        auto discard = [&](int tile) {
            arena.submit_action(find_action(arena.current_decision(), ActionType::Discard, tile));
            decline_ready_if_offered(arena);
        };
        auto pass = [&]() { arena.submit_action(find_action(arena.current_decision(), ActionType::Pass)); };
        discard(25);
        pass();
        find_action(arena.current_decision(), ActionType::Pon, 25);
        pass();
        pass();
        if (scenario == 2) {
            for (int tile : {20, 21, 22, 23}) {
                discard(tile);
                pass_all_responses(arena);
            }
        }
        discard(scenario == 1 ? 26 : 25);
        const auto ctx = arena.current_decision();
        const bool pon = std::any_of(ctx.legal_actions.begin(), ctx.legal_actions.end(),
            [](const Action& a) { return a.type == ActionType::Pon; });
        assert(pon == (scenario != 0));
    }
}

int main() {
    test_passed_pon_until_own_draw();
    test_start_deals_and_draws_for_dealer();
    test_discard_passes_and_next_draw();
    test_pon_takes_response_and_caller_must_discard();
    test_global_discard_history_converts_to_player_views();
    test_illegal_action_is_rejected();
    test_unreported_dihu_after_pon_tracks_responsibility_chicken();
    test_daiminkan_replacement_draw_gangkai_settles_all_players();
    test_qing_qidui_ron_settles_only_with_discarder();
    test_last_replacement_draw_allows_repao_and_settles_reverse_bean();
    test_last_live_draw_does_not_offer_ankan_or_kakan();
    test_last_discard_allows_only_ron_or_pass();
    test_repao_burns_only_source_chickens_and_settles_all_players();
    test_ron_has_priority_over_pon();
    test_tianting_player_kills_dealer_baoting_and_settles();
    test_consecutive_pons_do_not_close_untouched_players_ready_window();
    test_tianting_locks_hand_to_the_drawn_tile();
    test_kakan_always_queries_all_opponents_before_it_applies();
    test_qianggang_waits_for_all_responses_and_cancels_kakan();
    return 0;
}
