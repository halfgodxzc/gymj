#include "chicken.h"

#include <array>
#include <cassert>
#include <optional>
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

GYMJ::SettlementSnapshot make_snapshot(Players players) {
    players[0].tehai[0] = 1;
    return GYMJ::SettlementSnapshotBuilder::build(
        players,
        {
            GYMJ::HuTileOrigin::SelfDraw,
            GYMJ::Tile{0},
            0,
            {0},
        }
    );
}

GYMJ::ChickenContext all_jiao_context(
    std::optional<GYMJ::Tile> indicator = std::nullopt
) {
    GYMJ::ChickenContext context;
    context.indicator = indicator;
    context.is_jiao.fill(true);
    return context;
}

const GYMJ::ChickenItem& find_item(
    const GYMJ::PlayerChickenResult& result,
    GYMJ::ChickenKind kind
) {
    for (const auto& item : result.items) {
        if (item.kind == kind) return item;
    }
    assert(false);
    return result.items.front();
}

void test_indicator_advances_and_wraps_within_suit() {
    using namespace GYMJ;

    assert(Tile{YAO_JI_TILE_ID}.to_string() == "1s");
    assert(Tile{BA_TONG_TILE_ID}.to_string() == "8p");
    assert(ChickenResolver::flipped_chicken_for(Tile{0}) == Tile{1});
    assert(ChickenResolver::flipped_chicken_for(Tile{8}) == Tile{0});
    assert(ChickenResolver::flipped_chicken_for(Tile{17}) == Tile{9});
    assert(ChickenResolver::flipped_chicken_for(Tile{26}) == Tile{18});
}

void test_chongfeng_uses_global_first_and_opening_follow_discards() {
    using namespace GYMJ;

    ChongfengTracker tracker;
    assert(!tracker.observe_discard(0, Tile{0}));
    assert(!tracker.observe_discard(1, Tile{1}));

    // YaoJi and BaTong each have an independent global-first marker.
    assert(tracker.observe_discard(0, Tile{YAO_JI_TILE_ID}));
    assert(tracker.observe_discard(1, Tile{BA_TONG_TILE_ID}));

    // P2 can follow BaTong because this is P2's opening discard.
    assert(tracker.observe_discard(2, Tile{BA_TONG_TILE_ID}));
    assert(!tracker.observe_discard(2, Tile{YAO_JI_TILE_ID}));

    // P3 gives up the opening follow chance by discarding another tile.
    assert(!tracker.observe_discard(3, Tile{2}));
    assert(!tracker.observe_discard(3, Tile{YAO_JI_TILE_ID}));
}

void test_one_player_can_own_both_chongfeng_chicken_types() {
    using namespace GYMJ;

    ChongfengTracker tracker;
    assert(tracker.observe_discard(0, Tile{YAO_JI_TILE_ID}));
    assert(tracker.observe_discard(0, Tile{BA_TONG_TILE_ID}));

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[0].kawa[0].push_back(Tile{YAO_JI_TILE_ID});
    players[0].kawa[0].push_back(Tile{BA_TONG_TILE_ID});

    ChickenContext context;
    context.chongfeng_tile_masks[0] =
        CHONGFENG_YAO_JI_MASK | CHONGFENG_BA_TONG_MASK;
    context.is_jiao[0] = true;
    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );

    const auto& player = result.player(0);
    assert(find_item(player, ChickenKind::ChongfengYaoJi).total_points() == 5);
    assert(find_item(player, ChickenKind::ChongfengBaTong).total_points() == 5);
    assert(player.total_tile_count == 2);
    assert(player.total_points == 10);
}

void test_ba_tong_can_be_a_gold_chongfeng_chicken() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[2].kawa[0].push_back(Tile{BA_TONG_TILE_ID});

    ChickenContext context;
    context.indicator = Tile{15};
    context.chongfeng_tile_masks[2] = CHONGFENG_BA_TONG_MASK;
    context.is_jiao[2] = true;
    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );
    const auto& player = result.player(2);
    const auto& chongfeng = find_item(
        player,
        ChickenKind::ChongfengBaTong
    );

    assert(result.flipped_chicken ==
           std::optional<Tile>{Tile{BA_TONG_TILE_ID}});
    assert(chongfeng.is_gold);
    assert(chongfeng.points_per_tile == 10);
    assert(player.total_points == 10);
}

void test_counts_hand_discards_and_melds_without_double_counting() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[0].tehai[5] = 2;
    players[0].kawa[0].push_back(Tile{YAO_JI_TILE_ID});
    players[0].melds[0].push_back({
        MeldType::Pon,
        Tile{BA_TONG_TILE_ID},
        1,
    });

    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        all_jiao_context(Tile{4})
    );
    const auto& player = result.player(0);

    assert(result.flipped_chicken == std::optional<Tile>{Tile{5}});
    assert(player.total_tile_count == 6);
    assert(player.total_points == 10);

    const auto& flipped = find_item(player, ChickenKind::Flipped);
    assert(flipped.counts.hand == 2);
    assert(flipped.points_per_tile == 1);

    const auto& yao_ji = find_item(player, ChickenKind::YaoJi);
    assert(yao_ji.counts.discards == 1);
    assert(yao_ji.points_per_tile == 2);

    const auto& ba_tong = find_item(player, ChickenKind::BaTong);
    assert(ba_tong.counts.melds == 3);
    assert(ba_tong.points_per_tile == 2);
}

void test_fixed_chicken_becomes_gold_without_duplicate_flipped_item() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[0].tehai[YAO_JI_TILE_ID] = 1;
    players[0].kawa[0].push_back(Tile{YAO_JI_TILE_ID});

    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        all_jiao_context(Tile{26})
    );
    const auto& player = result.player(0);

    assert(result.flipped_chicken ==
           std::optional<Tile>{Tile{YAO_JI_TILE_ID}});
    assert(player.items.size() == 1);
    assert(player.items.front().kind == ChickenKind::YaoJi);
    assert(player.items.front().is_gold);
    assert(player.items.front().points_per_tile == 4);
    assert(player.items.front().tile_count() == 2);
    assert(player.total_points == 8);
}

void test_chongfeng_upgrades_exactly_one_discard_and_can_be_gold() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[0].tehai[YAO_JI_TILE_ID] = 1;
    players[0].kawa[0].push_back(Tile{YAO_JI_TILE_ID});

    ChickenContext context;
    context.indicator = Tile{26};
    context.chongfeng_tile_masks[0] = CHONGFENG_YAO_JI_MASK;
    context.is_jiao[0] = true;
    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );
    const auto& player = result.player(0);

    const auto& ordinary = find_item(player, ChickenKind::YaoJi);
    assert(ordinary.counts.hand == 1);
    assert(ordinary.counts.discards == 0);
    assert(ordinary.is_gold);
    assert(ordinary.total_points() == 4);

    const auto& chongfeng = find_item(
        player,
        ChickenKind::ChongfengYaoJi
    );
    assert(chongfeng.counts.discards == 1);
    assert(chongfeng.is_gold);
    assert(chongfeng.points_per_tile == 10);
    assert(player.total_points == 14);
}

void test_fixed_chickens_are_counted_without_an_indicator() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[1].tehai[BA_TONG_TILE_ID] = 2;

    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        all_jiao_context()
    );

    assert(!result.indicator.has_value());
    assert(!result.flipped_chicken.has_value());
    assert(result.player(0).total_points == 0);
    assert(result.player(1).total_tile_count == 2);
    assert(result.player(1).total_points == 4);
}

void test_virtual_hu_tile_is_counted_for_each_ron_hu_player() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].kawa[0].push_back(Tile{5});
    const auto snapshot = SettlementSnapshotBuilder::build(
        players,
        {
            HuTileOrigin::Discard,
            Tile{5},
            0,
            {1, 2},
        }
    );
    const auto result = ChickenResolver::resolve(
        snapshot,
        all_jiao_context(Tile{4})
    );

    assert(result.player(0).total_points == 0);
    assert(result.player(1).total_points == 1);
    assert(result.player(2).total_points == 1);
}

void test_invalid_chongfeng_marker_is_rejected() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    ChickenContext context;
    context.chongfeng_tile_masks[0] = CHONGFENG_YAO_JI_MASK;
    context.is_jiao[0] = true;

    bool threw = false;
    try {
        ChickenResolver::resolve(make_snapshot(players), context);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_player_without_jiao_ignores_hand_and_loses_public_chickens() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[1].tehai[YAO_JI_TILE_ID] = 1;
    players[1].kawa[0].push_back(Tile{BA_TONG_TILE_ID});
    players[1].melds[0].push_back({MeldType::Pon, Tile{5}, 0});

    ChickenContext context;
    context.indicator = Tile{4};
    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );

    const auto& player = result.player(1);
    assert(!player.is_jiao);
    assert(player.total_tile_count == 4);
    assert(player.total_points == -5);

    const auto& ba_tong = find_item(player, ChickenKind::BaTong);
    assert(ba_tong.counts.hand == 0);
    assert(ba_tong.counts.discards == 1);
    assert(ba_tong.points_per_tile == -2);

    const auto& flipped = find_item(player, ChickenKind::Flipped);
    assert(flipped.counts.hand == 0);
    assert(flipped.counts.melds == 3);
    assert(flipped.points_per_tile == -1);

    for (const auto& item : player.items) {
        assert(item.kind != ChickenKind::YaoJi);
    }
}

void test_player_without_jiao_loses_gold_chongfeng_at_equal_value() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[0] = 1;
    players[1].kawa[0].push_back(Tile{YAO_JI_TILE_ID});

    ChickenContext context;
    context.indicator = Tile{26};
    context.chongfeng_tile_masks[1] = CHONGFENG_YAO_JI_MASK;
    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );
    const auto& chongfeng = find_item(
        result.player(1),
        ChickenKind::ChongfengYaoJi
    );

    assert(!result.player(1).is_jiao);
    assert(chongfeng.is_gold);
    assert(chongfeng.points_per_tile == -10);
    assert(result.player(1).total_points == -10);
}

void test_hot_cannon_burns_only_the_source_players_chickens() {
    using namespace GYMJ;

    auto players = make_players();
    players[0].tehai[YAO_JI_TILE_ID] = 1;
    players[1].tehai[BA_TONG_TILE_ID] = 1;
    ChickenContext context = all_jiao_context();
    context.burned_players[0] = true;

    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );
    assert(result.player(0).is_burned);
    assert(result.player(0).raw_total_points == 2);
    assert(result.player(0).total_points == 0);
    assert(!result.player(1).is_burned);
    assert(result.player(1).total_points == 2);
}

void test_responsibility_chicken_keeps_normal_meld_count_and_adds_extra() {
    using namespace GYMJ;

    auto players = make_players();
    players[2].melds[0].push_back({
        MeldType::Pon,
        Tile{YAO_JI_TILE_ID},
        0,
    });
    ChickenContext context = all_jiao_context(Tile{26});
    context.responsibility_chickens.push_back({
        0,
        2,
        Tile{YAO_JI_TILE_ID},
    });

    const auto result = ChickenResolver::resolve(
        make_snapshot(players),
        context
    );
    assert(result.player(2).total_points == 12);
    assert(result.responsibility_chickens.size() == 1);
    const auto& responsibility = result.responsibility_chickens.front();
    assert(responsibility.source_player == 0);
    assert(responsibility.owner_player == 2);
    assert(responsibility.is_gold);
    assert(responsibility.points == 10);
}

} // namespace

int main() {
    test_indicator_advances_and_wraps_within_suit();
    test_chongfeng_uses_global_first_and_opening_follow_discards();
    test_one_player_can_own_both_chongfeng_chicken_types();
    test_ba_tong_can_be_a_gold_chongfeng_chicken();
    test_counts_hand_discards_and_melds_without_double_counting();
    test_fixed_chicken_becomes_gold_without_duplicate_flipped_item();
    test_chongfeng_upgrades_exactly_one_discard_and_can_be_gold();
    test_fixed_chickens_are_counted_without_an_indicator();
    test_virtual_hu_tile_is_counted_for_each_ron_hu_player();
    test_invalid_chongfeng_marker_is_rejected();
    test_player_without_jiao_ignores_hand_and_loses_public_chickens();
    test_player_without_jiao_loses_gold_chongfeng_at_equal_value();
    test_hot_cannon_burns_only_the_source_players_chickens();
    test_responsibility_chicken_keeps_normal_meld_count_and_adds_extra();
    return 0;
}
