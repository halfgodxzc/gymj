#pragma once

#include "guiyang_mahjong.h"
#include "shanten.h"

#include <array>
#include <cstdint>

namespace GYMJ {
namespace Algo {

struct EffectiveDrawCount {
    uint8_t kinds = 0;
    uint8_t copies = 0;
    // Zero means the tile does not improve this route. Otherwise this stores
    // the number of physically unseen copies that may still be drawn.
    std::array<uint8_t, TILE_TYPE_COUNT> remaining_by_tile{};
};

struct EffectiveDrawAnalysis {
    ShantenResult shanten;
    EffectiveDrawCount normal;
    EffectiveDrawCount seven_pairs;
    EffectiveDrawCount all_triplets;
};

EffectiveDrawAnalysis analyze_effective_draws(
    const uint8_t hand[TILE_TYPE_COUNT],
    const uint8_t visible_counts[TILE_TYPE_COUNT],
    int meld_count
);

} // namespace Algo
} // namespace GYMJ
