#pragma once

#include "action_encoder.h"
#include "arena.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace RL {

constexpr uint32_t OBSERVATION_VERSION = 3;
constexpr size_t MAX_DISCARD_EVENTS = 72;
constexpr size_t MAX_MELDS = 16;
constexpr uint8_t NO_TILE = 27;
constexpr uint8_t NO_PLAYER = 4;

struct DiscardObservation {
    uint8_t tile = NO_TILE;
    uint8_t relative_actor = NO_PLAYER;
    uint8_t was_called = 0;
};

struct MeldObservation {
    uint8_t tile = NO_TILE;
    uint8_t relative_owner = NO_PLAYER;
    uint8_t relative_source = NO_PLAYER;
    MeldType type = MeldType::Pon;
    uint8_t valid = 0;
};

struct CompactObservation {
    std::array<uint8_t, 27> hand_counts{};

    std::array<DiscardObservation, MAX_DISCARD_EVENTS> discards{};
    uint8_t discard_count = 0;

    std::array<MeldObservation, MAX_MELDS> melds{};
    uint8_t meld_count = 0;

    std::array<uint8_t, 27> visible_counts{};
    uint32_t waits = 0;
    int8_t shanten = 14;

    RoundPhase phase = RoundPhase::NotStarted;
    uint8_t current_actor = NO_PLAYER;
    uint8_t current_discard = NO_TILE;
    uint8_t current_discard_actor = NO_PLAYER;
    uint8_t dealer = NO_PLAYER;
    uint8_t wall_remaining = 0;

    std::array<uint8_t, 4> ready_kinds{};
    uint32_t locked_ready_waits = 0;

    std::array<uint64_t, 2> legal_mask{};
};

class ObservationEncoder {
public:
    static CompactObservation encode(
        const Arena& arena,
        uint8_t observer_id
    );
};

} // namespace RL
} // namespace GYMJ
