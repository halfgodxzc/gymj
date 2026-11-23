#pragma once

#include <cstdint>
#include <vector>

namespace GYMJ {
namespace Algo {

struct SuitPattern {
    uint8_t melds;
    uint8_t taatsu;
    uint8_t pairs;
};

struct ShantenResult {
    int8_t normal;
    int8_t seven_pairs;
    int8_t all_triplets;
    int8_t best;
};

struct ShantenAnalysis {
    ShantenResult result;
    SuitPattern wan;
    SuitPattern tong;
    SuitPattern tiao;
};

void set_shanten_logging(bool enabled);
bool is_shanten_logging_enabled();

const std::vector<SuitPattern>& get_suit_patterns(const uint8_t suit_tehai[9]);

int8_t calculate_normal_shanten(const uint8_t tehai[27], int meld_count);
int8_t calculate_seven_pairs_shanten(const uint8_t tehai[27], int meld_count);
int8_t calculate_all_triplets_shanten(const uint8_t tehai[27], int meld_count);
ShantenAnalysis analyze_shanten(const uint8_t tehai[27], int meld_count);
ShantenResult calculate_shanten(const uint8_t tehai[27], int meld_count);

} // namespace Algo
} // namespace GYMJ
