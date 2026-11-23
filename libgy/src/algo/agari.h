#pragma once

#include "guiyang_mahjong.h"
#include <cstdint>
#include <vector>

namespace GYMJ {
namespace Algo {

void ensure_agari_table_initialized();

bool is_standard_hu(const uint8_t tehai[TILE_TYPE_COUNT], int meld_count);
bool is_seven_pairs(const uint8_t tehai[TILE_TYPE_COUNT]);
bool is_flush(
    const uint8_t tehai[TILE_TYPE_COUNT],
    const std::vector<Meld>& melds
);
bool is_all_triplets(
    const uint8_t tehai[TILE_TYPE_COUNT],
    const std::vector<Meld>& melds
);

std::vector<Tile> get_waits(
    const uint8_t tehai[TILE_TYPE_COUNT],
    int meld_count
);

} // namespace Algo
} // namespace GYMJ
