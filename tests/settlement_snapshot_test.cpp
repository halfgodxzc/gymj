#include "settlement_snapshot.h"

#include <array>
#include <cassert>
#include <stdexcept>

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

void expect_invalid(
    const Players& players,
    const GYMJ::HuTileClaim& claim
) {
    bool threw = false;
    try {
        GYMJ::SettlementSnapshotBuilder::build(players, claim);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_self_draw_freezes_all_four_player_states() {
    using namespace GYMJ;

    auto players = make_players();
    players[2].tehai[7] = 1;
    players[0].kawa[0].push_back(Tile{3});
    players[1].melds[0].push_back({MeldType::Pon, Tile{9}, 0});

    const HuTileClaim claim{
        HuTileOrigin::SelfDraw,
        Tile{7},
        2,
        {2},
    };
    const auto snapshot = SettlementSnapshotBuilder::build(players, claim);

    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        assert(snapshot.player(player_id).player_id == player_id);
    }
    assert(snapshot.player(2).hand_count(Tile{7}) == 1);
    assert(snapshot.player(2).hand_counts[7] == 1);
    assert(snapshot.player(0).discard_count(Tile{3}) == 1);
    assert(snapshot.player(1).meld_tile_count(Tile{9}) == 3);

    players[2].tehai[7] = 0;
    players[0].kawa[0].clear();
    players[1].melds[0].clear();
    assert(snapshot.player(2).hand_count(Tile{7}) == 1);
    assert(snapshot.player(0).discard_count(Tile{3}) == 1);
    assert(snapshot.player(1).meld_tile_count(Tile{9}) == 3);
}

void test_two_ron_hu_players_each_receive_the_same_discard() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].kawa[0] = {Tile{4}, Tile{5}};
    players[1].tehai[5] = 2;

    const HuTileClaim claim{
        HuTileOrigin::Discard,
        Tile{5},
        0,
        {1, 2},
    };
    const auto snapshot = SettlementSnapshotBuilder::build(players, claim);

    const auto& source = snapshot.player(0);
    assert(source.discards.size() == 1);
    assert(source.discards.back() == Tile{4});
    assert(source.discard_count(Tile{4}) == 1);
    assert(source.discard_count(Tile{5}) == 0);

    assert(snapshot.player(1).hand_count(Tile{5}) == 3);
    assert(snapshot.player(2).hand_count(Tile{5}) == 1);
    assert(snapshot.player(3).hand_count(Tile{5}) == 0);

    assert(players[0].kawa[0].size() == 2);
    assert(players[0].kawa[0].back() == Tile{5});
    assert(players[1].tehai[5] == 2);
    assert(players[2].tehai[5] == 0);
}

void test_two_qianggang_hu_players_keep_pon_and_receive_fourth_tile() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].melds[0].push_back({MeldType::Pon, Tile{5}, 3});
    players[0].tehai[5] = 1;
    players[1].tehai[5] = 2;

    const HuTileClaim claim{
        HuTileOrigin::Kakan,
        Tile{5},
        0,
        {1, 2},
    };
    const auto snapshot = SettlementSnapshotBuilder::build(players, claim);

    const auto& source = snapshot.player(0);
    assert(source.melds.size() == 1);
    assert(source.melds.front().type == MeldType::Pon);
    assert(source.hand_count(Tile{5}) == 0);
    assert(source.meld_tile_count(Tile{5}) == 3);
    assert(source.settlement_tile_count(Tile{5}) == 3);

    assert(snapshot.player(1).hand_count(Tile{5}) == 3);
    assert(snapshot.player(2).hand_count(Tile{5}) == 1);
    assert(snapshot.player(3).hand_count(Tile{5}) == 0);

    assert(players[0].melds[0].front().type == MeldType::Pon);
    assert(players[0].tehai[5] == 1);
    assert(players[1].tehai[5] == 2);
    assert(players[2].tehai[5] == 0);
}

void test_invalid_claims_are_rejected() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].kawa[0].push_back(Tile{4});
    players[0].tehai[5] = 1;

    expect_invalid(players, {
        HuTileOrigin::Discard,
        Tile{5},
        0,
        {1},
    });
    expect_invalid(players, {
        HuTileOrigin::Kakan,
        Tile{5},
        0,
        {1},
    });
    expect_invalid(players, {
        HuTileOrigin::Discard,
        Tile{4},
        0,
        {1, 1},
    });
    expect_invalid(players, {
        HuTileOrigin::SelfDraw,
        Tile{6},
        2,
        {2},
    });
}

} // namespace

int main() {
    test_self_draw_freezes_all_four_player_states();
    test_two_ron_hu_players_each_receive_the_same_discard();
    test_two_qianggang_hu_players_keep_pon_and_receive_fourth_tile();
    test_invalid_claims_are_rejected();
    return 0;
}
