#pragma once

#include "action_encoder.h"
#include "pattern_feature_encoder.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace RL {

constexpr uint32_t DECISION_FEATURE_VERSION = 3;

enum class ActionFamily : uint8_t {
    Turn,
    Response,
    Ready,
    Unsupported,
};

enum class DecisionDrawKind : uint8_t {
    None,
    Opening,
    Normal,
    Replacement,
};

struct ActionSemantics {
    ActionType type = ActionType::Pass;
    uint8_t tile = NO_TILE;
    // Source supplies the claimed tile; target is the acting receiver.
    uint8_t relative_source = NO_PLAYER;
    uint8_t relative_target = NO_PLAYER;
    ActionFamily family = ActionFamily::Unsupported;
};

struct HuActionFeatures {
    uint8_t available = 0;
    uint8_t is_tsumo = 0;
    uint8_t is_qianggang = 0;
    uint16_t pattern_mask = 0;
    int8_t shape_fan = 0;
    int8_t ready_fan = 0;
    int8_t timing_fan = 0;
    int8_t special_fan = 0;
    int8_t total_fan = 0;
    int16_t hu_points = 0;
};

struct ReadyActionFeatures {
    uint8_t available = 0;
    ReadyKind kind = ReadyKind::None;
    uint32_t wait_mask = 0;
    uint8_t wait_kinds = 0;
    uint8_t wait_copies = 0;
    int16_t max_base_hu_points = 0;
};

struct PassActionFeatures {
    uint8_t available = 0;
    uint8_t declines_pon = 0;
    uint8_t declines_daiminkan = 0;
    uint8_t declines_ron = 0;
    uint8_t declines_ready = 0;
};

struct CandidateActionFeatures {
    uint8_t legal = 0;
    uint8_t supported_by_v4 = 0;
    uint8_t post_hand_known = 0;
    ActionSemantics semantics;
    HandPatternFeatures post_hand;
    HuActionFeatures hu;
    ReadyActionFeatures ready;
    PassActionFeatures pass;
};

struct PonFollowUpFeatures {
    uint8_t available = 0;
    std::array<uint8_t, TILE_TYPE_COUNT> discard_mask{};
    std::array<HandPatternFeatures, TILE_TYPE_COUNT> post_discard_hands{};
};

struct DecisionProgressFeatures {
    uint32_t decision_sequence = 0;
    std::array<uint8_t, PLAYER_COUNT> has_normal_drawn{};
    std::array<uint8_t, PLAYER_COUNT> has_discarded{};
    std::array<uint8_t, PLAYER_COUNT> opening_hand_changed{};
    std::array<uint8_t, PLAYER_COUNT> ready_window_closed{};
    DecisionDrawKind last_draw_kind = DecisionDrawKind::None;
    uint8_t must_discard_after_call = 0;
    uint8_t hot_discard_player = NO_PLAYER;
    uint8_t pending_discard_is_hot = 0;
    uint8_t pending_kakan = 0;
};

struct DecisionFeatureSet {
    CompactObservation observation;
    PublicPatternFeatures public_state;
    DecisionProgressFeatures progress;
    std::array<CandidateActionFeatures, ACTION_SPACE_SIZE> actions{};
    PonFollowUpFeatures pon_follow_up;
};

class DecisionFeatureEncoder {
public:
    static DecisionFeatureSet encode(
        const Arena& arena,
        uint8_t actor_id
    );
};

} // namespace RL
} // namespace GYMJ
