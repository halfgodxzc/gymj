#pragma once

#include "decision_feature_encoder.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace RL {

constexpr uint32_t PRIVILEGED_PATTERN_VERSION = 2;

struct PrivilegedEconomyFeatures {
    std::array<std::array<uint8_t, TILE_TYPE_COUNT>, PLAYER_COUNT>
        settlement_tile_counts{};
    // [owner][payer source][bean kind], relative to the acting player.
    // Self-drawn beans use owner as source; daiminkan uses its discarder.
    std::array<
        std::array<std::array<uint8_t, BEAN_TYPE_COUNT>, PLAYER_COUNT>,
        PLAYER_COUNT
    > bean_source_counts{};
    std::array<uint8_t, PLAYER_COUNT> chongfeng_tiles{
        NO_TILE, NO_TILE, NO_TILE, NO_TILE,
    };
    // Bit 0 is YaoJi; bit 1 is BaTong. The legacy tile field above keeps
    // the first marker for checkpoint and binding compatibility.
    std::array<uint8_t, PLAYER_COUNT> chongfeng_tile_masks{};
    // [source][owner] counts fixed chickens called from source to owner.
    std::array<std::array<uint8_t, PLAYER_COUNT>, PLAYER_COUNT>
        responsibility_chickens{};
    std::array<uint8_t, PLAYER_COUNT> player_tenpai{};
    // Signed current fixed-chicken ownership using true tenpai states.
    std::array<int8_t, PLAYER_COUNT> signed_fixed_chicken_counts{};
};

struct PrivilegedPatternFeatures {
    // Slots 0..3 correspond to relative players 0..3.
    std::array<HandPatternFeatures, PLAYER_COUNT> player_patterns{};
    std::array<uint32_t, PLAYER_COUNT> true_shape_wait_masks{};
    std::array<uint8_t, TILE_TYPE_COUNT> remaining_wall_counts{};
    PrivilegedEconomyFeatures economy;
    DecisionProgressFeatures progress;
};

class PrivilegedPatternEncoder {
public:
    static PrivilegedPatternFeatures encode(
        const Arena& arena,
        uint8_t actor_id
    );
};

} // namespace RL
} // namespace GYMJ
