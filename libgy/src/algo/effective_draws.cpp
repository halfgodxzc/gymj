#include "effective_draws.h"

#include <algorithm>
#include <stdexcept>

namespace GYMJ {
namespace Algo {
namespace {

void validate_input(
    const uint8_t hand[TILE_TYPE_COUNT],
    const uint8_t visible_counts[TILE_TYPE_COUNT],
    int meld_count
) {
    if (meld_count < 0 || meld_count > COMPLETE_HAND_MELD_COUNT) {
        throw std::invalid_argument("meld count must be in range 0..4");
    }

    int concealed_tiles = 0;
    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (hand[tile_id] > MAX_TILE_COPIES) {
            throw std::invalid_argument(
                "hand count exceeds physical tile copies"
            );
        }
        if (visible_counts[tile_id] > MAX_TILE_COPIES) {
            throw std::invalid_argument(
                "visible count exceeds physical tile copies"
            );
        }
        if (visible_counts[tile_id] < hand[tile_id]) {
            throw std::invalid_argument(
                "visible count cannot be lower than the concealed count"
            );
        }
        concealed_tiles += hand[tile_id];
    }

    const int expected_tiles =
        COMPLETE_HAND_TILE_COUNT - 1 - meld_count * TILES_PER_MELD;
    if (concealed_tiles != expected_tiles) {
        throw std::invalid_argument(
            "effective draw analysis requires a post-discard hand"
        );
    }
}

void add_if_improved(
    int8_t before,
    int8_t after,
    size_t tile_id,
    uint8_t remaining,
    EffectiveDrawCount& count
) {
    if (after >= before) return;
    ++count.kinds;
    count.copies = static_cast<uint8_t>(count.copies + remaining);
    count.remaining_by_tile[tile_id] = remaining;
}

} // namespace

EffectiveDrawAnalysis analyze_effective_draws(
    const uint8_t hand[TILE_TYPE_COUNT],
    const uint8_t visible_counts[TILE_TYPE_COUNT],
    int meld_count
) {
    validate_input(hand, visible_counts, meld_count);

    EffectiveDrawAnalysis result{};
    result.shanten = calculate_shanten(hand, meld_count);

    uint8_t after_draw[TILE_TYPE_COUNT];
    std::copy(hand, hand + TILE_TYPE_COUNT, after_draw);
    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        const uint8_t remaining = static_cast<uint8_t>(
            MAX_TILE_COPIES - visible_counts[tile_id]
        );
        if (remaining == 0) continue;

        ++after_draw[tile_id];
        const ShantenResult drawn = calculate_shanten(
            after_draw,
            meld_count
        );
        --after_draw[tile_id];

        add_if_improved(
            result.shanten.normal,
            drawn.normal,
            tile_id,
            remaining,
            result.normal
        );
        add_if_improved(
            result.shanten.seven_pairs,
            drawn.seven_pairs,
            tile_id,
            remaining,
            result.seven_pairs
        );
        add_if_improved(
            result.shanten.all_triplets,
            drawn.all_triplets,
            tile_id,
            remaining,
            result.all_triplets
        );
    }
    return result;
}

} // namespace Algo
} // namespace GYMJ
