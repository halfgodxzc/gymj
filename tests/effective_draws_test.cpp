#include "effective_draws.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace {

using TileCounts = std::array<uint8_t, GYMJ::TILE_TYPE_COUNT>;

TileCounts make_counts(std::initializer_list<uint8_t> tiles) {
    TileCounts counts{};
    for (uint8_t tile_id : tiles) {
        ++counts[tile_id];
    }
    return counts;
}

bool throws_invalid_argument(
    const TileCounts& hand,
    const TileCounts& visible,
    int meld_count
) {
    try {
        GYMJ::Algo::analyze_effective_draws(
            hand.data(),
            visible.data(),
            meld_count
        );
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

void test_normal_single_wait_counts_remaining_copies() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 1, 2,
        6, 6, 7, 8,
        9, 10, 11,
        18, 19, 20,
    });
    TileCounts visible = hand;
    const TileCounts original_hand = hand;

    auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        0
    );
    assert(analysis.shanten.normal == 0);
    assert(analysis.normal.kinds == 1);
    assert(analysis.normal.copies == 2);
    assert(analysis.normal.remaining_by_tile[6] == 2);
    for (size_t tile_id = 0; tile_id < GYMJ::TILE_TYPE_COUNT; ++tile_id) {
        if (tile_id != 6) {
            assert(analysis.normal.remaining_by_tile[tile_id] == 0);
        }
    }

    visible[6] = 3;
    analysis = analyze_effective_draws(hand.data(), visible.data(), 0);
    assert(analysis.normal.kinds == 1);
    assert(analysis.normal.copies == 1);
    assert(analysis.normal.remaining_by_tile[6] == 1);

    visible[6] = 4;
    analysis = analyze_effective_draws(hand.data(), visible.data(), 0);
    assert(analysis.normal.kinds == 0);
    assert(analysis.normal.copies == 0);
    assert(analysis.normal.remaining_by_tile[6] == 0);
    assert(hand == original_hand);
}

void test_seven_pairs_effective_draws_are_route_specific() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 0,
        1, 1,
        2, 2,
        9, 9,
        10, 10,
        18, 18,
        26,
    });
    const TileCounts visible = hand;
    const auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        0
    );

    assert(analysis.shanten.seven_pairs == 0);
    assert(analysis.seven_pairs.kinds == 1);
    assert(analysis.seven_pairs.copies == 3);
    assert(analysis.seven_pairs.remaining_by_tile[26] == 3);
}

void test_all_triplets_effective_draws_are_route_specific() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 0, 0,
        1, 1, 1,
        9, 9, 9,
        18, 18, 18,
        26,
    });
    const TileCounts visible = hand;
    const auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        0
    );

    assert(analysis.shanten.all_triplets == 0);
    assert(analysis.all_triplets.kinds == 1);
    assert(analysis.all_triplets.copies == 3);
    assert(analysis.all_triplets.remaining_by_tile[26] == 3);
}

void test_open_hand_disables_seven_pairs_effective_draws() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 0, 0,
        1, 1, 1,
        9, 9, 9,
        26,
    });
    TileCounts visible = hand;
    visible[18] = 3;
    const auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        1
    );

    assert(analysis.shanten.seven_pairs == 8);
    assert(analysis.seven_pairs.kinds == 0);
    assert(analysis.seven_pairs.copies == 0);
}

void test_two_sided_normal_wait_sums_kinds_and_copies() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 1, 2,
        3, 4, 5,
        15, 16, 17,
        19, 20,
        22, 22,
    });
    const TileCounts visible = hand;
    const auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        0
    );

    assert(analysis.shanten.normal == 0);
    assert(analysis.normal.kinds == 2);
    assert(analysis.normal.copies == 8);
}

void test_four_open_melds_leave_a_single_pair_wait() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({26});
    const TileCounts visible = hand;
    const auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        4
    );

    assert(analysis.shanten.normal == 0);
    assert(analysis.normal.kinds == 1);
    assert(analysis.normal.copies == 3);
    assert(analysis.shanten.seven_pairs == 8);
    assert(analysis.seven_pairs.kinds == 0);
    assert(analysis.seven_pairs.copies == 0);
    assert(analysis.shanten.all_triplets == 0);
    assert(analysis.all_triplets.kinds == 1);
    assert(analysis.all_triplets.copies == 3);
}

void test_quad_contributes_two_pair_units_to_seven_pairs() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 0, 0, 0,
        1, 1,
        2, 2,
        9, 9,
        10, 10,
        18,
    });
    const TileCounts visible = hand;
    const auto analysis = analyze_effective_draws(
        hand.data(),
        visible.data(),
        0
    );

    assert(analysis.shanten.seven_pairs == 0);
    assert(analysis.seven_pairs.kinds == 1);
    assert(analysis.seven_pairs.copies == 3);
}

void test_invalid_inputs_are_rejected() {
    TileCounts hand = make_counts({
        0, 1, 2,
        6, 6, 7, 8,
        9, 10, 11,
        18, 19, 20,
    });
    TileCounts visible = hand;

    TileCounts too_few_visible = visible;
    --too_few_visible[0];
    assert(throws_invalid_argument(hand, too_few_visible, 0));

    TileCounts too_many_visible = visible;
    too_many_visible[3] = 5;
    assert(throws_invalid_argument(hand, too_many_visible, 0));

    TileCounts wrong_size = hand;
    --wrong_size[0];
    assert(throws_invalid_argument(wrong_size, visible, 0));
    assert(throws_invalid_argument(hand, visible, -1));
    assert(throws_invalid_argument(hand, visible, 5));

    TileCounts excessive_hand = hand;
    excessive_hand[0] = 5;
    TileCounts excessive_hand_visible = excessive_hand;
    assert(throws_invalid_argument(
        excessive_hand,
        excessive_hand_visible,
        0
    ));
}

int main() {
    test_normal_single_wait_counts_remaining_copies();
    test_seven_pairs_effective_draws_are_route_specific();
    test_all_triplets_effective_draws_are_route_specific();
    test_open_hand_disables_seven_pairs_effective_draws();
    test_two_sided_normal_wait_sums_kinds_and_copies();
    test_four_open_melds_leave_a_single_pair_wait();
    test_quad_contributes_two_pair_units_to_seven_pairs();
    test_invalid_inputs_are_rejected();
    return 0;
}
