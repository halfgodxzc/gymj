#include "action_feature_encoder.h"

#include "effective_draws.h"
#include "flush_progress.h"

#include <stdexcept>

namespace GYMJ {
namespace RL {
namespace {

constexpr size_t feature_index(ActionFeatureIndex index) {
    return static_cast<size_t>(index);
}

constexpr size_t ANKAN_FIRST = TILE_TYPE_COUNT;
constexpr size_t KAKAN_FIRST = 2 * TILE_TYPE_COUNT;
constexpr size_t PON_ACTION_ID = 3 * TILE_TYPE_COUNT;
constexpr size_t DAIMINKAN_ACTION_ID = PON_ACTION_ID + 1;

bool is_legal(
    const std::array<uint64_t, 2>& legal_mask,
    size_t action_id
) {
    const size_t word = action_id / 64;
    const size_t bit = action_id % 64;
    return (legal_mask[word] & (uint64_t{1} << bit)) != 0;
}

struct OwnMeldSummary {
    int count = 0;
    uint8_t suit_mask = 0;
};

OwnMeldSummary summarize_own_melds(
    const CompactObservation& observation
) {
    if (observation.meld_count > observation.melds.size()) {
        throw std::invalid_argument("observation meld count exceeds capacity");
    }

    OwnMeldSummary summary;
    for (size_t index = 0; index < observation.meld_count; ++index) {
        const MeldObservation& meld = observation.melds[index];
        if (meld.valid && meld.relative_owner == 0) {
            if (meld.tile >= TILE_TYPE_COUNT) {
                throw std::invalid_argument(
                    "own meld tile must be in range 0..26"
                );
            }
            ++summary.count;
            const size_t suit = meld.tile / TILES_PER_SUIT;
            summary.suit_mask |= Algo::flush_suit_bit(suit);
        }
    }
    if (summary.count > COMPLETE_HAND_MELD_COUNT) {
        throw std::invalid_argument("player cannot have more than four melds");
    }
    return summary;
}

void validate_hand(const CompactObservation& observation) {
    for (uint8_t count : observation.hand_counts) {
        if (count > MAX_TILE_COPIES) {
            throw std::invalid_argument(
                "observation hand count exceeds physical tile copies"
            );
        }
    }
}

} // namespace

ActionFeatureMatrix ActionFeatureEncoder::encode(
    const CompactObservation& observation,
    const std::array<uint64_t, 2>& legal_mask
) {
    validate_hand(observation);
    const OwnMeldSummary own_melds = summarize_own_melds(observation);
    ActionFeatureMatrix features{};

    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (!is_legal(legal_mask, tile_id)) continue;
        if (observation.hand_counts[tile_id] == 0) {
            throw std::invalid_argument(
                "legal discard action is absent from the concealed hand"
            );
        }

        auto after_discard = observation.hand_counts;
        --after_discard[tile_id];
        const Algo::EffectiveDrawAnalysis analysis =
            Algo::analyze_effective_draws(
                after_discard.data(),
                observation.visible_counts.data(),
                own_melds.count
            );
        const Algo::FlushProgress flush = Algo::analyze_flush_progress(
            after_discard.data(),
            own_melds.suit_mask
        );

        ActionFeatures& action = features[tile_id];
        action[feature_index(ActionFeatureIndex::PostStateKnown)] = 1.0F;
        action[feature_index(ActionFeatureIndex::NormalShanten)] =
            static_cast<float>(analysis.shanten.normal);
        action[feature_index(ActionFeatureIndex::SevenPairsShanten)] =
            static_cast<float>(analysis.shanten.seven_pairs);
        action[feature_index(ActionFeatureIndex::AllTripletsShanten)] =
            static_cast<float>(analysis.shanten.all_triplets);
        action[feature_index(ActionFeatureIndex::NormalEffectiveDrawKinds)] =
            static_cast<float>(analysis.normal.kinds);
        action[feature_index(
            ActionFeatureIndex::SevenPairsEffectiveDrawKinds
        )] = static_cast<float>(analysis.seven_pairs.kinds);
        action[feature_index(
            ActionFeatureIndex::AllTripletsEffectiveDrawKinds
        )] = static_cast<float>(analysis.all_triplets.kinds);
        action[feature_index(ActionFeatureIndex::NormalEffectiveDrawCopies)] =
            static_cast<float>(analysis.normal.copies);
        action[feature_index(
            ActionFeatureIndex::SevenPairsEffectiveDrawCopies
        )] = static_cast<float>(analysis.seven_pairs.copies);
        action[feature_index(
            ActionFeatureIndex::AllTripletsEffectiveDrawCopies
        )] = static_cast<float>(analysis.all_triplets.copies);
        action[feature_index(ActionFeatureIndex::WanFlushOffSuitTiles)] =
            static_cast<float>(
                flush.off_suit_tiles[Algo::WAN_FLUSH_INDEX]
            );
        action[feature_index(ActionFeatureIndex::TongFlushOffSuitTiles)] =
            static_cast<float>(
                flush.off_suit_tiles[Algo::TONG_FLUSH_INDEX]
            );
        action[feature_index(ActionFeatureIndex::TiaoFlushOffSuitTiles)] =
            static_cast<float>(
                flush.off_suit_tiles[Algo::TIAO_FLUSH_INDEX]
            );
        action[feature_index(ActionFeatureIndex::WanFlushMeldCompatible)] =
            static_cast<float>(flush.meld_compatible[Algo::WAN_FLUSH_INDEX]);
        action[feature_index(ActionFeatureIndex::TongFlushMeldCompatible)] =
            static_cast<float>(flush.meld_compatible[Algo::TONG_FLUSH_INDEX]);
        action[feature_index(ActionFeatureIndex::TiaoFlushMeldCompatible)] =
            static_cast<float>(flush.meld_compatible[Algo::TIAO_FLUSH_INDEX]);
    }

    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (is_legal(legal_mask, ANKAN_FIRST + tile_id)) {
            if (observation.hand_counts[tile_id] != MAX_TILE_COPIES ||
                own_melds.count >= COMPLETE_HAND_MELD_COUNT) {
                throw std::invalid_argument("legal concealed kan has invalid hand or meld count");
            }
            ActionFeatures& action = features[ANKAN_FIRST + tile_id];
            action[feature_index(ActionFeatureIndex::MeldCountDelta)] = 1.0F;
            action[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] =
                static_cast<float>(MAX_TILE_COPIES);
            action[feature_index(ActionFeatureIndex::TakesReplacementDraw)] = 1.0F;
        }
        if (is_legal(legal_mask, KAKAN_FIRST + tile_id)) {
            if (observation.hand_counts[tile_id] == 0) {
                throw std::invalid_argument("legal added kan tile is absent from the concealed hand");
            }
            ActionFeatures& action = features[KAKAN_FIRST + tile_id];
            action[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] = 1.0F;
            action[feature_index(ActionFeatureIndex::TakesReplacementDraw)] = 1.0F;
        }
    }

    if (is_legal(legal_mask, PON_ACTION_ID) ||
        is_legal(legal_mask, DAIMINKAN_ACTION_ID)) {
        if (observation.current_discard >= TILE_TYPE_COUNT ||
            own_melds.count >= COMPLETE_HAND_MELD_COUNT) {
            throw std::invalid_argument("legal call needs a current discard and an open meld slot");
        }
        const uint8_t held = observation.hand_counts[observation.current_discard];
        if (is_legal(legal_mask, PON_ACTION_ID)) {
            if (held < 2) {
                throw std::invalid_argument("legal pon needs two matching concealed tiles");
            }
            ActionFeatures& action = features[PON_ACTION_ID];
            action[feature_index(ActionFeatureIndex::RequiresFollowUpDiscard)] = 1.0F;
            action[feature_index(ActionFeatureIndex::MeldCountDelta)] = 1.0F;
            action[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] = 2.0F;
        }
        if (is_legal(legal_mask, DAIMINKAN_ACTION_ID)) {
            if (held < 3) {
                throw std::invalid_argument("legal open kan needs three matching concealed tiles");
            }
            ActionFeatures& action = features[DAIMINKAN_ACTION_ID];
            action[feature_index(ActionFeatureIndex::MeldCountDelta)] = 1.0F;
            action[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] = 3.0F;
            action[feature_index(ActionFeatureIndex::TakesReplacementDraw)] = 1.0F;
        }
    }

    return features;
}

} // namespace RL
} // namespace GYMJ
