#include "round_settlement.h"

#include <array>
#include <cassert>
#include <stdexcept>
#include <string>

namespace {

GYMJ::HuSettlement hu_settlement(uint8_t player_id, int hu_points) {
    GYMJ::HuSettlement result;
    result.player_id = player_id;
    result.score.total_fan = hu_points / GYMJ::GUIYANG_HU_BASE_POINTS;
    result.score.hu_points = hu_points;
    return result;
}

GYMJ::ChickenResult make_chicken_result() {
    GYMJ::ChickenResult result;
    for (uint8_t id = 0; id < GYMJ::PLAYER_COUNT; ++id) {
        result.players[id].player_id = id;
    }
    return result;
}

void test_tsumo_hu_chicken_and_bean_are_zero_sum() {
    using namespace GYMJ;

    RoundResult round;
    round.reason = RoundEndReason::Tsumo;
    round.hu_players = {0};
    round.hu_tile = Tile{4};

    auto chicken = make_chicken_result();
    chicken.players[0].is_jiao = true;
    chicken.players[0].total_points = 2;
    chicken.players[1].total_points = -3;

    BeanResult beans;
    beans.ledger.transfer(
        ScoreCategory::AnKanBean,
        2,
        3,
        5,
        Tile{7}
    );

    const auto result = RoundSettlementCalculator::calculate(
        round,
        {hu_settlement(0, 5)},
        {},
        chicken,
        beans
    );

    // Hu: +15/-5 each. P0 chicken: +6/-2 each.
    // P1 no-jiao chicken: P1 pays P0 3. Bean: P2 pays P3 5.
    assert((result.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{24, -10, -12, -2}));
    assert(result.ledger.net_total() == 0);
}

void test_called_chicken_settles_all_four_players() {
    using namespace GYMJ;

    RoundResult round;
    round.reason = RoundEndReason::Tsumo;
    round.hu_players = {0};
    round.hu_tile = Tile{2};

    auto chicken = make_chicken_result();
    chicken.players[0].is_jiao = true;
    chicken.players[3].is_jiao = true;

    // P1 discarded YaoJi and P2 called it. P2 has no jiao, so the three
    // exposed YaoJi are negative public chickens paid to every jiao player.
    chicken.players[2].total_points = -6;
    chicken.responsibility_chickens.push_back({1, 2, Tile{18}, false, 5});

    const auto result = RoundSettlementCalculator::calculate(
        round,
        {hu_settlement(0, 5)},
        {},
        chicken,
        BeanResult{}
    );

    // Hu: P1/P2/P3 each pay P0 5.
    // Chicken: P2 pays P0 and P3 6 each. P1 no longer owns the called tile.
    // Responsibility is inactive because neither P1 nor P2 has jiao.
    assert((result.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{21, -5, -17, 1}));

    int package_transfer_count = 0;
    int responsibility_transfer_count = 0;
    for (const auto& transfer : result.ledger.transfers()) {
        if (transfer.category == ScoreCategory::Chicken &&
            transfer.from == 2 && transfer.points == 6 &&
            (transfer.to == 0 || transfer.to == 3)) {
            ++package_transfer_count;
        }
        if (transfer.category == ScoreCategory::ResponsibilityChicken) {
            ++responsibility_transfer_count;
        }
    }
    assert(package_transfer_count == 2);
    assert(responsibility_transfer_count == 0);
    assert(result.ledger.net_total() == 0);
}

void test_multi_ron_charges_the_source_for_each_hu_score() {
    using namespace GYMJ;

    RoundResult round;
    round.reason = RoundEndReason::Ron;
    round.hu_players = {1, 3};
    round.fangpao_player = 0;
    round.hu_tile = Tile{9};

    const auto result = RoundSettlementCalculator::calculate(
        round,
        {hu_settlement(1, 5), hu_settlement(3, 15)},
        {},
        std::nullopt,
        BeanResult{}
    );
    assert((result.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{-20, 5, 0, 15}));
    assert(result.ledger.transfers().size() == 2);
}

void test_no_jiao_public_chicken_only_pays_jiao_players() {
    using namespace GYMJ;

    RoundResult round;
    round.reason = RoundEndReason::Tsumo;
    round.hu_players = {0};
    round.hu_tile = Tile{1};

    auto chicken = make_chicken_result();
    chicken.players[0].is_jiao = true;
    chicken.players[2].is_jiao = true;
    chicken.players[1].total_points = -4;

    const auto result = RoundSettlementCalculator::calculate(
        round,
        {hu_settlement(0, 5)},
        {},
        chicken,
        BeanResult{}
    );
    assert((result.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{19, -13, -1, -5}));
}

void test_exhaustive_draw_charges_only_non_jiao_players() {
    using namespace GYMJ;

    RoundResult round;
    round.reason = RoundEndReason::ExhaustiveDraw;

    ChaJiaoSettlement p0;
    p0.player_id = 0;
    p0.best_wait = Tile{4};
    p0.score = GuiyangScoring::calculate(HuFeatures{});

    HuFeatures all_triplets;
    all_triplets.pattern = HuPattern::AllTriplets;
    ChaJiaoSettlement p1;
    p1.player_id = 1;
    p1.best_wait = Tile{22};
    p1.features = all_triplets;
    p1.score = GuiyangScoring::calculate(all_triplets);

    const auto result = RoundSettlementCalculator::calculate(
        round,
        {},
        {p0, p1},
        std::nullopt,
        BeanResult{}
    );
    assert((result.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{10, 20, -15, -15}));
    assert(result.ledger.transfers().size() == 4);
    for (const auto& transfer : result.ledger.transfers()) {
        assert(transfer.category == ScoreCategory::ChaJiao);
        assert(transfer.from == 2 || transfer.from == 3);
        assert(transfer.to == 0 || transfer.to == 1);
    }
}

void test_cha_jiao_uses_highest_scoring_wait() {
    using namespace GYMJ;

    std::array<PlayerState, PLAYER_COUNT> players{
        PlayerState(0),
        PlayerState(1),
        PlayerState(2),
        PlayerState(3),
    };
    for (uint8_t tile_id = 0; tile_id < 6; ++tile_id) {
        players[0].tehai[tile_id] = 2;
    }
    players[0].tehai[6] = 1;

    players[1].tehai[0] = 1;
    players[1].tehai[1] = 1;
    players[1].tehai[2] = 1;
    players[1].tehai[9] = 1;
    players[1].tehai[10] = 1;
    players[1].tehai[11] = 1;
    players[1].tehai[18] = 1;
    players[1].tehai[19] = 1;
    players[1].tehai[20] = 1;
    players[1].tehai[21] = 1;
    players[1].tehai[22] = 1;
    players[1].tehai[23] = 1;
    players[1].tehai[7] = 1;
    constexpr std::array<uint8_t, 13> non_tenpai_hand{
        2, 3, 4, 8, 12, 14, 15, 15, 18, 20, 24, 24, 25,
    };
    for (uint8_t tile_id : non_tenpai_hand) {
        players[2].tehai[tile_id]++;
        players[3].tehai[tile_id]++;
    }

    const auto settlements =
        RoundSettlementCalculator::evaluate_cha_jiao(
            players,
            std::array<ReadyState, PLAYER_COUNT>{}
        );

    assert(settlements.size() == 2);
    assert(settlements[0].player_id == 0);
    assert(settlements[0].best_wait == Tile{6});
    assert(settlements[0].features.pattern == HuPattern::SevenPairs);
    assert(settlements[0].features.is_flush);
    assert(settlements[0].score.total_fan == 6);
    assert(settlements[0].score.hu_points == 30);
    assert(settlements[1].player_id == 1);
    assert(settlements[1].score.hu_points == 5);
}

void test_cha_jiao_rejects_impossible_fifth_copy_wait() {
    using namespace GYMJ;

    std::array<PlayerState, PLAYER_COUNT> players{
        PlayerState(0),
        PlayerState(1),
        PlayerState(2),
        PlayerState(3),
    };
    constexpr std::array<uint8_t, 13> impossible_wait_hand{
        2, 3, 4,
        14, 14, 14, 14,
        19, 20, 21,
        24, 24, 24,
    };
    constexpr std::array<uint8_t, 13> non_tenpai_hand{
        2, 3, 4, 8, 12, 14, 15, 15, 18, 20, 24, 24, 25,
    };

    for (uint8_t tile_id : impossible_wait_hand) {
        players[0].tehai[tile_id]++;
    }
    for (uint8_t player_id = 1; player_id < PLAYER_COUNT; ++player_id) {
        for (uint8_t tile_id : non_tenpai_hand) {
            players[player_id].tehai[tile_id]++;
        }
    }

    const auto settlements =
        RoundSettlementCalculator::evaluate_cha_jiao(
            players,
            std::array<ReadyState, PLAYER_COUNT>{}
        );

    assert(settlements.empty());
}

void test_cha_jiao_rejects_invalid_concealed_tile_count() {
    using namespace GYMJ;

    std::array<PlayerState, PLAYER_COUNT> players{
        PlayerState(0),
        PlayerState(1),
        PlayerState(2),
        PlayerState(3),
    };
    players[0].tehai[5] = 3;
    players[0].tehai[19] = 1;
    players[0].tehai[20] = 1;
    players[0].tehai[21] = 1;
    players[0].melds[0].push_back({MeldType::Pon, Tile{16}, 1});
    players[0].melds[0].push_back({MeldType::Daiminkan, Tile{2}, 2});

    bool rejected = false;
    try {
        static_cast<void>(RoundSettlementCalculator::evaluate_cha_jiao(
            players,
            std::array<ReadyState, PLAYER_COUNT>{}
        ));
    } catch (const std::logic_error& error) {
        rejected = std::string(error.what()) ==
                   "exhaustive draw player has invalid concealed tile count";
    }
    assert(rejected);
}

void test_responsibility_chicken_reverses_when_only_source_has_jiao() {
    using namespace GYMJ;

    RoundResult round;
    round.reason = RoundEndReason::Tsumo;
    round.hu_players = {0};
    round.hu_tile = Tile{2};

    auto chicken = make_chicken_result();
    chicken.players[0].is_jiao = true;
    chicken.players[2].total_points = -6;
    chicken.responsibility_chickens.push_back({0, 2, Tile{18}, false, 5});

    const auto result = RoundSettlementCalculator::calculate(
        round,
        {hu_settlement(0, 5)},
        {},
        chicken,
        BeanResult{}
    );
    assert((result.ledger.score_deltas() ==
            std::array<int, PLAYER_COUNT>{26, -5, -16, -5}));

    const auto& transfer = result.ledger.transfers().back();
    assert(transfer.category == ScoreCategory::ResponsibilityChicken);
    assert(transfer.from == 2);
    assert(transfer.to == 0);
}

} // namespace

int main() {
    test_tsumo_hu_chicken_and_bean_are_zero_sum();
    test_called_chicken_settles_all_four_players();
    test_multi_ron_charges_the_source_for_each_hu_score();
    test_no_jiao_public_chicken_only_pays_jiao_players();
    test_exhaustive_draw_charges_only_non_jiao_players();
    test_cha_jiao_uses_highest_scoring_wait();
    test_cha_jiao_rejects_impossible_fifth_copy_wait();
    test_cha_jiao_rejects_invalid_concealed_tile_count();
    test_responsibility_chicken_reverses_when_only_source_has_jiao();
    return 0;
}
