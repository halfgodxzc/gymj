#include "flush_progress.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace {

using TileCounts = std::array<uint8_t, GYMJ::TILE_TYPE_COUNT>;

TileCounts make_counts(std::initializer_list<uint8_t> tiles) {
    TileCounts counts{};
    for (uint8_t tile_id : tiles) ++counts[tile_id];
    return counts;
}

} // namespace

void test_off_suit_tiles_are_counted_for_each_target_suit() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 0, 1, 2, 3, 4, 5, 6,
        9, 10, 11,
        18, 19,
    });
    const TileCounts original = hand;
    const FlushProgress progress = analyze_flush_progress(hand.data(), 0);

    assert(progress.off_suit_tiles[0] == 5);
    assert(progress.off_suit_tiles[1] == 10);
    assert(progress.off_suit_tiles[2] == 11);
    assert(progress.meld_compatible[0] == 1);
    assert(progress.meld_compatible[1] == 1);
    assert(progress.meld_compatible[2] == 1);
    assert(hand == original);
}

void test_fixed_melds_gate_flush_suits() {
    using namespace GYMJ::Algo;

    const TileCounts hand{};
    const FlushProgress wan_only = analyze_flush_progress(
        hand.data(),
        uint8_t{1} << 0
    );
    assert(wan_only.meld_compatible[0] == 1);
    assert(wan_only.meld_compatible[1] == 0);
    assert(wan_only.meld_compatible[2] == 0);

    const FlushProgress mixed = analyze_flush_progress(
        hand.data(),
        (uint8_t{1} << 0) | (uint8_t{1} << 1)
    );
    assert(mixed.meld_compatible[0] == 0);
    assert(mixed.meld_compatible[1] == 0);
    assert(mixed.meld_compatible[2] == 0);
}

void test_each_single_suit_meld_mask_selects_exactly_one_flush() {
    using namespace GYMJ::Algo;

    const TileCounts hand{};
    for (size_t fixed_suit = 0;
         fixed_suit < FLUSH_SUIT_COUNT;
         ++fixed_suit) {
        const FlushProgress progress = analyze_flush_progress(
            hand.data(),
            flush_suit_bit(fixed_suit)
        );
        for (size_t target_suit = 0;
             target_suit < FLUSH_SUIT_COUNT;
             ++target_suit) {
            assert(progress.meld_compatible[target_suit] ==
                   (target_suit == fixed_suit ? 1 : 0));
        }
    }
}

void test_empty_concealed_hand_has_no_off_suit_tiles() {
    using namespace GYMJ::Algo;

    const TileCounts hand{};
    const FlushProgress progress = analyze_flush_progress(hand.data(), 0);
    for (size_t suit = 0; suit < FLUSH_SUIT_COUNT; ++suit) {
        assert(progress.off_suit_tiles[suit] == 0);
        assert(progress.meld_compatible[suit] == 1);
    }
}

void test_maximum_physical_counts_do_not_overflow() {
    using namespace GYMJ::Algo;

    TileCounts hand{};
    hand.fill(GYMJ::MAX_TILE_COPIES);
    const FlushProgress progress = analyze_flush_progress(hand.data(), 0);
    for (size_t suit = 0; suit < FLUSH_SUIT_COUNT; ++suit) {
        assert(progress.off_suit_tiles[suit] == 72);
        assert(progress.meld_compatible[suit] == 1);
    }
}

void test_invalid_hand_and_suit_mask_are_rejected() {
    TileCounts hand{};
    hand[0] = 5;
    bool hand_threw = false;
    try {
        GYMJ::Algo::analyze_flush_progress(hand.data(), 0);
    } catch (const std::invalid_argument&) {
        hand_threw = true;
    }
    assert(hand_threw);

    hand[0] = 0;
    bool mask_threw = false;
    try {
        GYMJ::Algo::analyze_flush_progress(
            hand.data(),
            uint8_t{1} << GYMJ::Algo::FLUSH_SUIT_COUNT
        );
    } catch (const std::invalid_argument&) {
        mask_threw = true;
    }
    assert(mask_threw);
}

void test_flush_target_prefers_lower_shanten_before_tile_count() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        1, 1, 2, 4, 5, 6, 8,
        18, 19, 20, 22, 23, 24,
    });
    const FlushTargetAnalysis analysis = analyze_flush_targets(
        hand.data(), hand.data(), 0, 0
    );

    assert(analysis.suits[WAN_FLUSH_INDEX].concealed_tiles == 7);
    assert(analysis.suits[TIAO_FLUSH_INDEX].concealed_tiles == 6);
    assert(analysis.suits[WAN_FLUSH_INDEX].off_suit_tiles == 6);
    assert(analysis.suits[TIAO_FLUSH_INDEX].off_suit_tiles == 7);
    assert(analysis.suits[WAN_FLUSH_INDEX].shanten == 5);
    assert(analysis.suits[TIAO_FLUSH_INDEX].shanten == 4);
    assert(analysis.best_suit_mask == flush_suit_bit(TIAO_FLUSH_INDEX));
}

void test_flush_target_uses_tile_count_after_shanten_tie() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 4, 5, 7,
        11, 14, 14, 17,
        18, 19, 20, 22, 23,
    });
    const FlushTargetAnalysis analysis = analyze_flush_targets(
        hand.data(), hand.data(), 0, 0
    );

    assert(analysis.suits[TONG_FLUSH_INDEX].shanten == 5);
    assert(analysis.suits[TIAO_FLUSH_INDEX].shanten == 5);
    assert(analysis.suits[TONG_FLUSH_INDEX].concealed_tiles == 4);
    assert(analysis.suits[TIAO_FLUSH_INDEX].concealed_tiles == 5);
    assert(analysis.best_suit_mask == flush_suit_bit(TIAO_FLUSH_INDEX));
}

void test_flush_target_uses_effective_copies_after_count_tie() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 0, 1, 3, 8,
        10, 11, 14,
        18, 18, 23, 24, 25,
    });
    const FlushTargetAnalysis analysis = analyze_flush_targets(
        hand.data(), hand.data(), 0, 0
    );

    assert(analysis.suits[WAN_FLUSH_INDEX].shanten == 5);
    assert(analysis.suits[TIAO_FLUSH_INDEX].shanten == 5);
    assert(analysis.suits[WAN_FLUSH_INDEX].concealed_tiles == 5);
    assert(analysis.suits[TIAO_FLUSH_INDEX].concealed_tiles == 5);
    assert(analysis.suits[WAN_FLUSH_INDEX].effective_draws.copies == 9);
    assert(analysis.suits[TIAO_FLUSH_INDEX].effective_draws.copies == 11);
    uint8_t wan_copy_sum = 0;
    uint8_t tiao_copy_sum = 0;
    for (size_t tile_id = 0; tile_id < GYMJ::TILE_TYPE_COUNT; ++tile_id) {
        wan_copy_sum = static_cast<uint8_t>(
            wan_copy_sum + analysis.suits[WAN_FLUSH_INDEX]
                .effective_draws.remaining_by_tile[tile_id]
        );
        tiao_copy_sum = static_cast<uint8_t>(
            tiao_copy_sum + analysis.suits[TIAO_FLUSH_INDEX]
                .effective_draws.remaining_by_tile[tile_id]
        );
    }
    assert(wan_copy_sum == 9);
    assert(tiao_copy_sum == 11);
    assert(analysis.best_suit_mask == flush_suit_bit(TIAO_FLUSH_INDEX));
}

void test_flush_target_uses_seed_for_exact_ties() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        1, 2, 2, 4, 5,
        12, 16, 17,
        18, 18, 20, 23, 26,
    });
    const FlushTargetAnalysis analysis = analyze_flush_targets(
        hand.data(), hand.data(), 0, 0
    );

    const uint8_t expected_mask =
        flush_suit_bit(WAN_FLUSH_INDEX) |
        flush_suit_bit(TIAO_FLUSH_INDEX);
    assert(analysis.best_suit_mask == expected_mask);
    assert(select_flush_target_suit(analysis, 0) == TIAO_FLUSH_INDEX);
    assert(select_flush_target_suit(analysis, 0) == TIAO_FLUSH_INDEX);
    assert(select_flush_target_suit(analysis, 2) == WAN_FLUSH_INDEX);
}

void test_flush_target_respects_fixed_meld_suit() {
    using namespace GYMJ::Algo;

    const TileCounts hand = make_counts({
        0, 1, 2,
        9, 10, 11, 12, 13, 14, 15,
    });
    const FlushTargetAnalysis analysis = analyze_flush_targets(
        hand.data(),
        hand.data(),
        1,
        flush_suit_bit(WAN_FLUSH_INDEX)
    );

    assert(analysis.suits[WAN_FLUSH_INDEX].meld_compatible == 1);
    assert(analysis.suits[TONG_FLUSH_INDEX].meld_compatible == 0);
    assert(analysis.suits[TIAO_FLUSH_INDEX].meld_compatible == 0);
    assert(analysis.best_suit_mask == flush_suit_bit(WAN_FLUSH_INDEX));
}

void test_flush_target_rejects_non_post_discard_hand() {
    TileCounts hand = make_counts({
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    });
    bool threw = false;
    try {
        GYMJ::Algo::analyze_flush_targets(
            hand.data(), hand.data(), 0, 0
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_off_suit_tiles_are_counted_for_each_target_suit();
    test_fixed_melds_gate_flush_suits();
    test_each_single_suit_meld_mask_selects_exactly_one_flush();
    test_empty_concealed_hand_has_no_off_suit_tiles();
    test_maximum_physical_counts_do_not_overflow();
    test_invalid_hand_and_suit_mask_are_rejected();
    test_flush_target_prefers_lower_shanten_before_tile_count();
    test_flush_target_uses_tile_count_after_shanten_tie();
    test_flush_target_uses_effective_copies_after_count_tie();
    test_flush_target_uses_seed_for_exact_ties();
    test_flush_target_respects_fixed_meld_suit();
    test_flush_target_rejects_non_post_discard_hand();
    return 0;
}
