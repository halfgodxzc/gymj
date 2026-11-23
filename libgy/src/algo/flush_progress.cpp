#include "flush_progress.h"

#include "shanten.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace GYMJ {
namespace Algo {
namespace {

constexpr uint8_t valid_suit_mask() {
    return flush_suit_bit(FLUSH_SUIT_COUNT) - 1;
}

void validate_suit_mask(uint8_t fixed_meld_suit_mask) {
    if ((fixed_meld_suit_mask & ~valid_suit_mask()) != 0) {
        throw std::invalid_argument("fixed meld suit mask has invalid bits");
    }
}

void validate_target_input(
    const uint8_t hand[TILE_TYPE_COUNT],
    const uint8_t visible_counts[TILE_TYPE_COUNT],
    int meld_count,
    uint8_t fixed_meld_suit_mask
) {
    validate_suit_mask(fixed_meld_suit_mask);
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
            "flush target analysis requires a post-discard hand"
        );
    }
}

uint64_t splitmix64(uint64_t value) {
    value += UINT64_C(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

} // namespace

FlushProgress analyze_flush_progress(
    const uint8_t hand[TILE_TYPE_COUNT],
    uint8_t fixed_meld_suit_mask
) {
    static_assert(FLUSH_SUIT_COUNT == 3);
    validate_suit_mask(fixed_meld_suit_mask);

    FlushProgress result;
    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (hand[tile_id] > MAX_TILE_COPIES) {
            throw std::invalid_argument(
                "hand count exceeds physical tile copies"
            );
        }

        const size_t tile_suit = tile_id / TILES_PER_SUIT;
        for (size_t target_suit = 0;
             target_suit < FLUSH_SUIT_COUNT;
             ++target_suit) {
            if (tile_suit != target_suit) {
                result.off_suit_tiles[target_suit] =
                    static_cast<uint8_t>(
                        result.off_suit_tiles[target_suit] + hand[tile_id]
                    );
            }
        }
    }

    for (size_t target_suit = 0;
         target_suit < FLUSH_SUIT_COUNT;
         ++target_suit) {
        const uint8_t target_mask = flush_suit_bit(target_suit);
        result.meld_compatible[target_suit] =
            fixed_meld_suit_mask == 0 ||
            fixed_meld_suit_mask == target_mask;
    }
    return result;
}

FlushTargetAnalysis analyze_flush_targets(
    const uint8_t hand[TILE_TYPE_COUNT],
    const uint8_t visible_counts[TILE_TYPE_COUNT],
    int meld_count,
    uint8_t fixed_meld_suit_mask
) {
    validate_target_input(
        hand,
        visible_counts,
        meld_count,
        fixed_meld_suit_mask
    );

    const FlushProgress progress = analyze_flush_progress(
        hand,
        fixed_meld_suit_mask
    );
    FlushTargetAnalysis result;

    for (size_t suit = 0; suit < FLUSH_SUIT_COUNT; ++suit) {
        FlushSuitCandidate& candidate = result.suits[suit];
        candidate.meld_compatible = progress.meld_compatible[suit];
        candidate.off_suit_tiles = progress.off_suit_tiles[suit];
        if (!candidate.meld_compatible) continue;

        std::array<uint8_t, TILE_TYPE_COUNT> flush_hand{};
        const size_t first_tile = suit * TILES_PER_SUIT;
        const size_t tile_end = first_tile + TILES_PER_SUIT;
        for (size_t tile_id = first_tile; tile_id < tile_end; ++tile_id) {
            flush_hand[tile_id] = hand[tile_id];
            candidate.concealed_tiles = static_cast<uint8_t>(
                candidate.concealed_tiles + hand[tile_id]
            );
        }

        candidate.shanten = calculate_shanten(
            flush_hand.data(),
            meld_count
        ).best;
        for (size_t tile_id = first_tile; tile_id < tile_end; ++tile_id) {
            const uint8_t remaining = static_cast<uint8_t>(
                MAX_TILE_COPIES - visible_counts[tile_id]
            );
            if (remaining == 0) continue;

            ++flush_hand[tile_id];
            const int8_t after_draw = calculate_shanten(
                flush_hand.data(),
                meld_count
            ).best;
            --flush_hand[tile_id];
            if (after_draw >= candidate.shanten) continue;

            ++candidate.effective_draws.kinds;
            candidate.effective_draws.copies = static_cast<uint8_t>(
                candidate.effective_draws.copies + remaining
            );
            candidate.effective_draws.remaining_by_tile[tile_id] = remaining;
        }
    }

    int8_t best_shanten = std::numeric_limits<int8_t>::max();
    uint8_t most_tiles = 0;
    uint8_t most_effective_copies = 0;
    for (size_t suit = 0; suit < FLUSH_SUIT_COUNT; ++suit) {
        const FlushSuitCandidate& candidate = result.suits[suit];
        if (!candidate.meld_compatible) continue;

        const bool better =
            candidate.shanten < best_shanten ||
            (candidate.shanten == best_shanten &&
             candidate.concealed_tiles > most_tiles) ||
            (candidate.shanten == best_shanten &&
             candidate.concealed_tiles == most_tiles &&
             candidate.effective_draws.copies > most_effective_copies);
        const bool tied =
            candidate.shanten == best_shanten &&
            candidate.concealed_tiles == most_tiles &&
            candidate.effective_draws.copies == most_effective_copies;

        if (better) {
            best_shanten = candidate.shanten;
            most_tiles = candidate.concealed_tiles;
            most_effective_copies = candidate.effective_draws.copies;
            result.best_suit_mask = flush_suit_bit(suit);
        } else if (tied) {
            result.best_suit_mask |= flush_suit_bit(suit);
        }
    }
    return result;
}

uint8_t select_flush_target_suit(
    const FlushTargetAnalysis& analysis,
    uint64_t seed
) {
    std::array<uint8_t, FLUSH_SUIT_COUNT> candidates{};
    size_t candidate_count = 0;
    for (size_t suit = 0; suit < FLUSH_SUIT_COUNT; ++suit) {
        if ((analysis.best_suit_mask & flush_suit_bit(suit)) == 0) continue;
        candidates[candidate_count++] = static_cast<uint8_t>(suit);
    }
    if (candidate_count == 0) return NO_FLUSH_SUIT;

    const size_t selected = static_cast<size_t>(
        splitmix64(seed) % candidate_count
    );
    return candidates[selected];
}

} // namespace Algo
} // namespace GYMJ
