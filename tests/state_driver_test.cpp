#include "guiyang_mahjong.h"

#include <cassert>
#include <cstdint>

namespace {

void add_tiles(uint8_t hand[27], std::initializer_list<int> tiles) {
    for (int tile : tiles) {
        hand[tile]++;
    }
}

bool has_action(const std::vector<GYMJ::Action>& actions, GYMJ::ActionType type, uint8_t tile_id) {
    for (const auto& action : actions) {
        if (action.type == type && action.tile.id == tile_id) {
            return true;
        }
    }
    return false;
}

} // namespace

void test_self_draw_and_discard_updates_hand_and_river() {
    using namespace GYMJ;

    PlayerState ps(0);
    add_tiles(ps.tehai, {
        0, 1, 2,
        9, 10, 11,
        18, 19, 20,
        3, 4, 5,
        6,
    });

    ps.apply_action({ActionType::Draw, 0, Tile{7}, 0});
    assert(ps.tehai[7] == 1);

    ps.apply_action({ActionType::Discard, 0, Tile{7}, 0});
    assert(ps.tehai[7] == 0);
    assert(ps.kawa[0].size() == 1);
    assert(ps.kawa[0].back().id == 7);
}

void test_opponent_pon_updates_public_melds_and_wait_visibility() {
    using namespace GYMJ;

    PlayerState ps(0);
    add_tiles(ps.tehai, {
        1, 2,
        9, 10, 11,
        18, 19, 20,
        21, 22, 23,
        15, 15,
    });

    ps.apply_action({ActionType::Discard, 1, Tile{0}, 1});
    assert(ps.kawa[1].size() == 1);
    assert(ps.kawa[1].back().id == 0);

    ps.apply_action({ActionType::Pon, 2, Tile{0}, 1});
    assert(ps.kawa[1].empty());
    assert(ps.melds[2].size() == 1);
    assert(ps.melds[2][0].type == MeldType::Pon);
    assert(ps.melds[2][0].tile.id == 0);

    ps.apply_action({ActionType::Discard, 3, Tile{0}, 3});
    assert(ps.kawa[3].size() == 1);
    assert(ps.kawa[3].back().id == 0);

    assert(!ps.waits[0]);
    assert(ps.waits[3]);
}

void test_self_pon_updates_hand_and_self_melds() {
    using namespace GYMJ;

    PlayerState ps(0);
    add_tiles(ps.tehai, {
        0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4,
    });

    auto actions = ps.get_legal_actions_on_opponent_discard(1, Tile{0});
    assert(has_action(actions, ActionType::Pon, 0));

    ps.apply_action({ActionType::Pon, 0, Tile{0}, 1});
    assert(ps.tehai[0] == 0);
    assert(ps.melds[0].size() == 1);
    assert(ps.melds[0][0].type == MeldType::Pon);
}

void test_base_draw_actions_leave_hu_decisions_to_arena() {
    using namespace GYMJ;

    PlayerState ps(0);
    add_tiles(ps.tehai, {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    });

    ps.apply_action({ActionType::Draw, 0, Tile{26}, 0});
    assert(ps.tehai[26] == 2);
    assert(ps.shanten == -1);

    const auto actions = ps.get_legal_actions_on_draw(Tile{26});
    assert(!has_action(actions, ActionType::Tsumo, 26));

    ps.apply_action({ActionType::Tsumo, 0, Tile{26}, 0});
    assert(ps.tehai[26] == 2);
}

void test_kakan_requires_the_newly_drawn_fourth_tile() {
    using namespace GYMJ;

    PlayerState retained(0);
    add_tiles(retained.tehai, {0, 0, 0});

    retained.apply_action({ActionType::Pon, 0, Tile{0}, 1});
    retained.apply_action({ActionType::Draw, 0, Tile{1}, 0});
    assert(retained.tehai[0] == 1);
    const auto unrelated_draw_actions =
        retained.get_legal_actions_on_draw(Tile{1});
    assert(!has_action(unrelated_draw_actions, ActionType::Kakan, 0));

    PlayerState ps(0);
    add_tiles(ps.tehai, {0, 0});

    ps.apply_action({ActionType::Pon, 0, Tile{0}, 1});
    assert(ps.tehai[0] == 0);
    assert(ps.melds[0].size() == 1);
    assert(ps.melds[0][0].type == MeldType::Pon);

    ps.apply_action({ActionType::Draw, 0, Tile{0}, 0});
    assert(ps.tehai[0] == 1);
    const auto actions = ps.get_legal_actions_on_draw(Tile{0});
    assert(has_action(actions, ActionType::Kakan, 0));

    ps.apply_action({ActionType::Kakan, 0, Tile{0}, 0});
    assert(ps.tehai[0] == 0);
    assert(ps.melds[0][0].type == MeldType::Kakan);
    assert(ps.melds[0][0].source == 1);
}

void test_action_refreshes_real_shanten() {
    using namespace GYMJ;

    PlayerState ps(0);
    add_tiles(ps.tehai, {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    });

    ps.apply_action({ActionType::Pass, 1, Tile{0}, 0});
    assert(ps.shanten == 0);
    assert(ps.waits[26]);

    PlayerState distant(0);
    add_tiles(distant.tehai, {
        0, 2, 4, 6,
        9, 11, 13, 15,
        18, 20, 22, 24,
        26,
    });
    distant.apply_action({ActionType::Pass, 1, Tile{0}, 0});
    assert(distant.shanten > 0);
    for (bool wait : distant.waits) {
        assert(!wait);
    }
}

int main() {
    test_self_draw_and_discard_updates_hand_and_river();
    test_opponent_pon_updates_public_melds_and_wait_visibility();
    test_self_pon_updates_hand_and_self_melds();
    test_base_draw_actions_leave_hu_decisions_to_arena();
    test_kakan_requires_the_newly_drawn_fourth_tile();
    test_action_refreshes_real_shanten();

    return 0;
}
