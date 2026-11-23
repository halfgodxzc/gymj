#include "game.h"

#include <array>
#include <cassert>
#include <numeric>
#include <utility>
#include <vector>

namespace {

std::vector<GYMJ::Tile> scripted_wall(
    const std::array<std::vector<uint8_t>, GYMJ::PLAYER_COUNT>& hands,
    const std::vector<uint8_t>& front_draws
) {
    std::array<int, GYMJ::TILE_TYPE_COUNT> remaining;
    remaining.fill(GYMJ::MAX_TILE_COPIES);

    auto reserve = [&](uint8_t tile_id) {
        assert(tile_id < GYMJ::TILE_TYPE_COUNT);
        assert(remaining[tile_id] > 0);
        remaining[tile_id]--;
    };
    for (const auto& hand : hands) {
        assert(hand.size() == 13);
        for (uint8_t tile_id : hand) reserve(tile_id);
    }
    for (uint8_t tile_id : front_draws) reserve(tile_id);

    std::vector<GYMJ::Tile> wall;
    wall.reserve(GYMJ::TILE_TYPE_COUNT * GYMJ::MAX_TILE_COPIES);
    for (size_t tile_index = 0; tile_index < 13; ++tile_index) {
        for (uint8_t player_id = 0;
             player_id < GYMJ::PLAYER_COUNT;
             ++player_id) {
            wall.push_back(GYMJ::Tile{hands[player_id][tile_index]});
        }
    }
    for (uint8_t tile_id : front_draws) {
        wall.push_back(GYMJ::Tile{tile_id});
    }
    for (uint8_t tile_id = 0;
         tile_id < GYMJ::TILE_TYPE_COUNT;
         ++tile_id) {
        while (remaining[tile_id] > 0) {
            wall.push_back(GYMJ::Tile{tile_id});
            remaining[tile_id]--;
        }
    }
    assert(wall.size() ==
           GYMJ::TILE_TYPE_COUNT * GYMJ::MAX_TILE_COPIES);
    return wall;
}

void test_single_and_multi_ron_hu_player_dealer_rotation() {
    using namespace GYMJ;
    std::array<bool, PLAYER_COUNT> is_jiao{};

    RoundResult result;
    result.reason = RoundEndReason::Tsumo;
    result.hu_players = {2};
    assert(DealerRotation::next_dealer(0, result, is_jiao) == 2);

    result.reason = RoundEndReason::Ron;
    result.hu_players = {3, 1};
    result.fangpao_player = 2;
    assert(DealerRotation::next_dealer(0, result, is_jiao) == 2);
    assert(DealerRotation::next_dealer(3, result, is_jiao) == 2);

    result.reason = RoundEndReason::QiangGang;
    assert(DealerRotation::next_dealer(1, result, is_jiao) == 2);

    result.hu_players = {0, 1, 3};
    result.fangpao_player = 2;
    assert(DealerRotation::next_dealer(0, result, is_jiao) == 2);

    for (uint8_t dealer = 0; dealer < PLAYER_COUNT; ++dealer) {
        for (uint8_t source = 0; source < PLAYER_COUNT; ++source) {
            result.reason = RoundEndReason::Ron;
            result.fangpao_player = source;
            result.hu_players.clear();
            for (uint8_t player = 0; player < PLAYER_COUNT; ++player) {
                if (player != source) result.hu_players.push_back(player);
            }
            assert(DealerRotation::next_dealer(dealer, result, is_jiao) == source);
            result.hu_players.pop_back();
            assert(DealerRotation::next_dealer(dealer, result, is_jiao) == source);
            std::swap(result.hu_players[0], result.hu_players[1]);
            assert(DealerRotation::next_dealer(dealer, result, is_jiao) == source);
            result.hu_players.pop_back();
            assert(DealerRotation::next_dealer(dealer, result, is_jiao) == result.hu_players[0]);
        }
    }
}

void test_exhaustive_draw_continuation_rules() {
    using namespace GYMJ;
    RoundResult result;
    result.reason = RoundEndReason::ExhaustiveDraw;
    std::array<bool, PLAYER_COUNT> is_jiao{};

    assert(DealerRotation::next_dealer(1, result, is_jiao) == 1);
    is_jiao.fill(true);
    assert(DealerRotation::next_dealer(1, result, is_jiao) == 1);

    is_jiao.fill(false);
    is_jiao[1] = true;
    is_jiao[3] = true;
    assert(DealerRotation::next_dealer(1, result, is_jiao) == 1);

    is_jiao[1] = false;
    assert(DealerRotation::next_dealer(1, result, is_jiao) == 2);
}

const GYMJ::Action& first_action(
    const GYMJ::DecisionContext& decision,
    GYMJ::ActionType type,
    int tile_id = -1
) {
    for (const auto& action : decision.legal_actions) {
        if (action.type == type &&
            (tile_id < 0 || action.tile.id == tile_id)) {
            return action;
        }
    }
    assert(false);
    return decision.legal_actions.front();
}

void pass_ready_declarations(GYMJ::Game& game) {
    while (game.arena().phase() ==
           GYMJ::RoundPhase::AwaitReadyDeclaration) {
        game.submit_action(first_action(
            game.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
}

void play_passive_hand(GYMJ::Game& game) {
    using namespace GYMJ;
    while (!game.arena().is_finished()) {
        const auto decision = game.current_decision();
        const bool pass_phase =
            decision.phase == RoundPhase::AwaitReadyDeclaration ||
            decision.phase == RoundPhase::AwaitDiscardResponses ||
            decision.phase == RoundPhase::AwaitKakanResponses;
        game.submit_action(first_action(
            decision,
            pass_phase ? ActionType::Pass : ActionType::Discard
        ));
    }
}

void test_game_runs_bounded_reproducible_hand_lifecycle() {
    using namespace GYMJ;
    Game game(GameConfig{42, 0, 2, {100, 100, 100, 100}});
    assert(game.phase() == GamePhase::NotStarted);

    game.start();
    assert(game.phase() == GamePhase::PlayingHand);
    play_passive_hand(game);
    game.finish_hand();
    assert(game.phase() == GamePhase::BetweenHands);
    assert(game.history().size() == 1);
    assert(game.history().front().hand_index == 0);
    assert(std::accumulate(game.scores().begin(), game.scores().end(), 0) == 400);

    game.start_next_hand();
    play_passive_hand(game);
    game.finish_hand();
    assert(game.is_finished());
    assert(game.hand_index() == 2);
    assert(game.history().size() == 2);
    assert(std::accumulate(game.scores().begin(), game.scores().end(), 0) == 400);
}

void test_game_accumulates_nonzero_scores_across_scripted_hands() {
    using namespace GYMJ;

    const std::array<std::vector<uint8_t>, PLAYER_COUNT> first_hands{{
        {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 26},
        {1, 1, 1, 2, 2, 2, 6, 6, 6, 12, 12, 12, 13},
        {14, 14, 14, 15, 15, 15, 17, 17, 17, 21, 21, 21, 22},
        {3, 3, 3, 4, 4, 4, 5, 5, 5, 9, 9, 9, 10},
    }};
    const std::array<std::vector<uint8_t>, PLAYER_COUNT> second_hands{{
        {0, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15},
        {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},
        {5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9},
        {10, 10, 10, 11, 11, 11, 17, 17, 17, 19, 19, 19, 20},
    }};
    std::vector<Wall> walls;
    walls.push_back(Wall::from_tiles(scripted_wall(
        first_hands,
        {26, 7}
    )));
    walls.push_back(Wall::from_tiles(scripted_wall(
        second_hands,
        {24, 25}
    )));

    Game game(
        GameConfig{42, 0, 2, {100, 100, 100, 100}},
        std::move(walls)
    );
    game.start();
    pass_ready_declarations(game);
    game.submit_action(first_action(
        game.current_decision(),
        ActionType::Tsumo,
        26
    ));
    assert(game.arena().round_settlement().ledger.score_deltas() ==
           (std::array<int, PLAYER_COUNT>{135, -45, -45, -45}));

    game.finish_hand();
    assert((game.scores() ==
            std::array<int, PLAYER_COUNT>{235, 55, 55, 55}));
    assert(game.history().size() == 1);
    assert((game.history()[0].scores_after ==
            std::array<int, PLAYER_COUNT>{235, 55, 55, 55}));
    assert(game.history()[0].next_dealer == 0);
    assert(game.history()[0].dealer_continues);
    assert(game.dealer() == 0);
    assert(game.dealer_continuation() == 1);

    game.start_next_hand();
    pass_ready_declarations(game);
    game.submit_action(first_action(
        game.current_decision(),
        ActionType::Discard,
        0
    ));
    if (game.arena().phase() == RoundPhase::AwaitReadyDeclaration) {
        game.submit_action(first_action(
            game.current_decision(),
            ActionType::Pass
        ));
    }
    game.submit_action(first_action(
        game.current_decision(),
        ActionType::Ron,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        game.submit_action(first_action(
            game.current_decision(),
            ActionType::Pass
        ));
    }

    const auto& second_settlement = game.arena().round_settlement();
    assert(second_settlement.hu_settlements.size() == 1);
    assert(second_settlement.hu_settlements.front().features.pattern ==
           HuPattern::AllTriplets);
    assert(second_settlement.hu_settlements.front().features.is_flush);
    assert(second_settlement.hu_settlements.front().score.total_fan == 5);
    assert(second_settlement.hu_settlements.front().score.hu_points == 25);
    assert((second_settlement.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{-25, 25, 0, 0}));

    game.finish_hand();
    assert(game.is_finished());
    assert(game.hand_index() == 2);
    assert((game.scores() ==
            std::array<int, PLAYER_COUNT>{210, 80, 55, 55}));
    assert(game.history().size() == 2);
    assert(game.history()[1].dealer == 0);
    assert(game.history()[1].dealer_continuation == 1);
    assert((game.history()[1].scores_after ==
            std::array<int, PLAYER_COUNT>{210, 80, 55, 55}));
    assert(game.history()[1].next_dealer == 1);
    assert(!game.history()[1].dealer_continues);
    assert(game.dealer() == 1);
    assert(game.dealer_continuation() == 0);
    assert(std::accumulate(game.scores().begin(), game.scores().end(), 0) ==
           400);
}

} // namespace

int main() {
    test_single_and_multi_ron_hu_player_dealer_rotation();
    test_exhaustive_draw_continuation_rules();
    test_game_runs_bounded_reproducible_hand_lifecycle();
    test_game_accumulates_nonzero_scores_across_scripted_hands();
    return 0;
}
