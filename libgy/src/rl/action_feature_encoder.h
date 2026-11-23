#pragma once

#include "action_encoder.h"
#include "observation_encoder.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace GYMJ {
namespace RL {

// Bump this version if feature order or meaning changes after consumers use it.
constexpr uint32_t ACTION_FEATURE_VERSION = 2;
constexpr size_t ACTION_FEATURE_COUNT = 20;

enum class ActionFeatureIndex : size_t {
    PostStateKnown = 0,
    RequiresFollowUpDiscard = 1,
    MeldCountDelta = 2,
    NormalShanten = 3,
    SevenPairsShanten = 4,
    AllTripletsShanten = 5,
    NormalEffectiveDrawKinds = 6,
    SevenPairsEffectiveDrawKinds = 7,
    AllTripletsEffectiveDrawKinds = 8,
    NormalEffectiveDrawCopies = 9,
    SevenPairsEffectiveDrawCopies = 10,
    AllTripletsEffectiveDrawCopies = 11,
    WanFlushOffSuitTiles = 12,
    TongFlushOffSuitTiles = 13,
    TiaoFlushOffSuitTiles = 14,
    // Only fixed meld suits are checked; concealed off-suit tiles may be discarded.
    WanFlushMeldCompatible = 15,
    TongFlushMeldCompatible = 16,
    TiaoFlushMeldCompatible = 17,
    ConcealedTilesConsumed = 18,
    TakesReplacementDraw = 19,
};

using ActionFeatures = std::array<float, ACTION_FEATURE_COUNT>;
using ActionFeatureMatrix = std::array<ActionFeatures, ACTION_SPACE_SIZE>;

class ActionFeatureEncoder {
public:
    static ActionFeatureMatrix encode(
        const CompactObservation& observation,
        const std::array<uint64_t, 2>& legal_mask
    );
};

} // namespace RL
} // namespace GYMJ
