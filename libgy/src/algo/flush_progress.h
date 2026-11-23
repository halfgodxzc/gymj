#pragma once

#include "effective_draws.h"
#include "guiyang_mahjong.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace Algo {

constexpr size_t FLUSH_SUIT_COUNT = TILE_TYPE_COUNT / TILES_PER_SUIT;
constexpr size_t WAN_FLUSH_INDEX = 0;
constexpr size_t TONG_FLUSH_INDEX = 1;
constexpr size_t TIAO_FLUSH_INDEX = 2;
constexpr uint8_t NO_FLUSH_SUIT =
    static_cast<uint8_t>(FLUSH_SUIT_COUNT);

constexpr uint8_t flush_suit_bit(size_t suit_index) {
    return static_cast<uint8_t>(uint8_t{1} << suit_index);
}

struct FlushProgress {
    std::array<uint8_t, FLUSH_SUIT_COUNT> off_suit_tiles{};
    std::array<uint8_t, FLUSH_SUIT_COUNT> meld_compatible{};
};

struct FlushSuitCandidate {
    int8_t shanten = 8;
    uint8_t concealed_tiles = 0;
    uint8_t off_suit_tiles = 0;
    EffectiveDrawCount effective_draws{};
    uint8_t meld_compatible = 0;
};

struct FlushTargetAnalysis {
    std::array<FlushSuitCandidate, FLUSH_SUIT_COUNT> suits{};
    uint8_t best_suit_mask = 0;
};

FlushProgress analyze_flush_progress(
    const uint8_t hand[TILE_TYPE_COUNT],
    uint8_t fixed_meld_suit_mask
);

// The hand must contain the post-discard concealed tile count. A flush may
// finish as a standard hand, seven pairs, or all triplets.
FlushTargetAnalysis analyze_flush_targets(
    const uint8_t hand[TILE_TYPE_COUNT],
    const uint8_t visible_counts[TILE_TYPE_COUNT],
    int meld_count,
    uint8_t fixed_meld_suit_mask
);

// Returns NO_FLUSH_SUIT when fixed melds make every flush impossible.
uint8_t select_flush_target_suit(
    const FlushTargetAnalysis& analysis,
    uint64_t seed
);

} // namespace Algo
} // namespace GYMJ
