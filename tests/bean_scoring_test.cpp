#include "bean_scoring.h"

#include <array>
#include <cassert>

namespace {

using Players = std::array<GYMJ::PlayerState, GYMJ::PLAYER_COUNT>;

Players make_players() {
    return {
        GYMJ::PlayerState(0),
        GYMJ::PlayerState(1),
        GYMJ::PlayerState(2),
        GYMJ::PlayerState(3),
    };
}

void assert_deltas(
    const GYMJ::BeanResult& result,
    const std::array<int, GYMJ::PLAYER_COUNT>& expected
) {
    assert(result.ledger.score_deltas() == expected);
    assert(result.ledger.net_total() == 0);
}

void test_jiao_ankan_is_paid_by_all_three_players() {
    using namespace GYMJ;
    auto players = make_players();
    players[1].melds[0].push_back({MeldType::AnKan, Tile{4}, 1});
    std::array<bool, PLAYER_COUNT> is_jiao{};
    is_jiao[1] = true;

    const auto result = BeanScoring::calculate(players, is_jiao);
    assert(result.items.size() == 1);
    assert(result.items.front().kind == BeanKind::AnKan);
    assert(result.ledger.transfers().size() == 3);
    assert_deltas(result, {-5, 15, -5, -5});
}

void test_no_jiao_kakan_pays_all_three_players() {
    using namespace GYMJ;
    auto players = make_players();
    players[2].melds[0].push_back({MeldType::Kakan, Tile{8}, 0});
    std::array<bool, PLAYER_COUNT> is_jiao{};

    const auto result = BeanScoring::calculate(players, is_jiao);
    assert(result.items.front().kind == BeanKind::Kakan);
    assert_deltas(result, {5, 5, -15, 5});
}

void test_daiminkan_only_settles_with_its_source() {
    using namespace GYMJ;
    auto players = make_players();
    players[3].melds[0].push_back({MeldType::Daiminkan, Tile{12}, 1});
    std::array<bool, PLAYER_COUNT> is_jiao{};
    is_jiao[3] = true;

    auto result = BeanScoring::calculate(players, is_jiao);
    assert(result.ledger.transfers().size() == 1);
    assert_deltas(result, {0, -5, 0, 5});

    is_jiao[3] = false;
    result = BeanScoring::calculate(players, is_jiao);
    assert_deltas(result, {0, 5, 0, -5});
}

void test_beans_accumulate_and_pon_is_ignored() {
    using namespace GYMJ;
    auto players = make_players();
    players[0].melds[0].push_back({MeldType::Pon, Tile{1}, 2});
    players[0].melds[0].push_back({MeldType::AnKan, Tile{2}, 0});
    players[0].melds[0].push_back({MeldType::Kakan, Tile{3}, 2});
    std::array<bool, PLAYER_COUNT> is_jiao{};
    is_jiao[0] = true;

    const auto result = BeanScoring::calculate(players, is_jiao);
    assert(result.items.size() == 2);
    assert(result.ledger.transfers().size() == 6);
    assert_deltas(result, {30, -10, -10, -10});
}

} // namespace

int main() {
    test_jiao_ankan_is_paid_by_all_three_players();
    test_no_jiao_kakan_pays_all_three_players();
    test_daiminkan_only_settles_with_its_source();
    test_beans_accumulate_and_pon_is_ignored();
    return 0;
}
