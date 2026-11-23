#pragma once

#include "arena.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace RL {

constexpr uint32_t PRIVILEGED_STATE_VERSION = 3;
constexpr size_t MAX_PRIVILEGED_WALL_TILES =
    TILE_TYPE_COUNT * 4 - PLAYER_COUNT * (COMPLETE_HAND_TILE_COUNT - 1);
constexpr size_t PRIVILEGED_FUTURE_DRAW_COUNT = 3;
constexpr uint8_t PRIVILEGED_TILE_PAD = static_cast<uint8_t>(TILE_TYPE_COUNT);

struct PrivilegedState {
    // Slots 0..3 correspond to relative players 0..3.
    std::array<uint8_t, PLAYER_COUNT> player_tenpai{};
    // Slots 0..2 correspond to relative players 1..3.
    std::array<std::array<uint8_t, TILE_TYPE_COUNT>, PLAYER_COUNT - 1>
        opponent_hands{};
    // Front draw is first; replacement draw is last.
    std::array<uint8_t, MAX_PRIVILEGED_WALL_TILES> wall_tiles{};
    uint8_t wall_count = 0;
    // Nominal front draws if no calls or kans change turn order.
    std::array<
        std::array<uint8_t, PRIVILEGED_FUTURE_DRAW_COUNT>,
        PLAYER_COUNT
    > future_draw_tiles{};
    // Back draws used by future kans, nearest replacement first.
    std::array<uint8_t, PRIVILEGED_FUTURE_DRAW_COUNT>
        replacement_draw_tiles{};
};

class PrivilegedStateEncoder {
public:
    static PrivilegedState encode(const Arena& arena, uint8_t actor_id);
};

} // namespace RL
} // namespace GYMJ
