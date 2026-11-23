#pragma once

#include "effective_draws.h"
#include "flush_progress.h"
#include "hu_evaluator.h"
#include "observation_encoder.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace RL {

constexpr uint32_t PATTERN_FEATURE_VERSION = 2;
constexpr size_t PATTERN_ROUTE_COUNT = 3;
constexpr size_t MELD_TYPE_COUNT = 4;
constexpr size_t BEAN_TYPE_COUNT = 3;

enum class PatternRoute : size_t {
    Normal = 0,
    SevenPairs = 1,
    AllTriplets = 2,
};

enum HuPatternFeatureBit : uint16_t {
    PatternStandard = uint16_t{1} << 0,
    PatternAllTriplets = uint16_t{1} << 1,
    PatternSevenPairs = uint16_t{1} << 2,
    PatternDragonSevenPairs = uint16_t{1} << 3,
    PatternDiLong = uint16_t{1} << 4,
    PatternFlush = uint16_t{1} << 5,
    PatternSingleWait = uint16_t{1} << 6,
};

struct RoutePatternFeatures {
    int8_t shanten = 8;
    Algo::EffectiveDrawCount effective_draws{};
};

struct FlushPatternFeatures {
    int8_t shanten = 8;
    uint8_t concealed_tiles = 0;
    uint8_t off_suit_tiles = 0;
    uint8_t meld_compatible = 0;
    Algo::EffectiveDrawCount effective_draws{};
};

struct WaitValueFeatures {
    uint16_t pattern_mask = 0;
    int8_t base_fan = 0;
    int16_t base_hu_points = 0;
    uint8_t remaining_copies = 0;
};

struct HandPatternFeatures {
    uint8_t post_discard_state = 0;
    std::array<RoutePatternFeatures, PATTERN_ROUTE_COUNT> routes{};
    std::array<FlushPatternFeatures, Algo::FLUSH_SUIT_COUNT> flushes{};
    uint8_t best_flush_suit_mask = 0;

    uint32_t shape_wait_mask = 0;
    uint32_t live_wait_mask = 0;
    uint8_t live_wait_kinds = 0;
    uint8_t live_wait_copies = 0;
    std::array<WaitValueFeatures, TILE_TYPE_COUNT> wait_values{};
};

struct PublicPlayerFeatures {
    std::array<uint8_t, TILE_TYPE_COUNT> river_counts{};
    std::array<uint8_t, TILE_TYPE_COUNT> called_discard_counts{};
    std::array<uint8_t, TILE_TYPE_COUNT> meld_tile_counts{};
    std::array<uint8_t, TILE_TYPE_COUNT> public_owned_tile_counts{};
    std::array<uint8_t, MELD_TYPE_COUNT> meld_type_counts{};
    std::array<uint8_t, BEAN_TYPE_COUNT> bean_type_counts{};

    uint8_t discard_count = 0;
    uint8_t meld_count = 0;
    uint8_t fixed_chicken_count = 0;
    // -1 means known not tenpai, 0 means unknown, 1 means known tenpai.
    int8_t known_tenpai_status = 0;
    // Current fixed chickens are positive for known tenpai and negative for
    // known not-tenpai. Unknown opponents stay zero.
    int8_t signed_fixed_chicken_count = 0;
    uint8_t chongfeng_chicken_count = 0;
    uint8_t responsibility_chicken_given_count = 0;
    uint8_t responsibility_chicken_received_count = 0;
};

struct PublicPatternFeatures {
    HandPatternFeatures hand;
    std::array<PublicPlayerFeatures, PLAYER_COUNT> players{};
};

uint16_t hu_pattern_feature_mask(const HuEvaluation& evaluation);

class PatternFeatureEncoder {
public:
    static HandPatternFeatures encode_hand(
        const CompactObservation& observation
    );
    static PublicPatternFeatures encode(
        const CompactObservation& observation
    );
};

} // namespace RL
} // namespace GYMJ
