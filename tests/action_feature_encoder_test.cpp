#include "action_feature_encoder.h"
#include "effective_draws.h"
#include "flush_progress.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace {

using GYMJ::RL::ActionFeatureIndex;
using GYMJ::RL::ActionFeatures;

constexpr size_t PASS_ACTION_ID = 85;
constexpr size_t ANKAN_FIRST = GYMJ::TILE_TYPE_COUNT;
constexpr size_t KAKAN_FIRST = 2 * GYMJ::TILE_TYPE_COUNT;
constexpr size_t PON_ACTION_ID = 3 * GYMJ::TILE_TYPE_COUNT;
constexpr size_t DAIMINKAN_ACTION_ID = PON_ACTION_ID + 1;

constexpr size_t feature_index(ActionFeatureIndex index) {
    return static_cast<size_t>(index);
}

std::array<uint64_t, 2> mask_for(size_t action_id) {
    std::array<uint64_t, 2> mask{};
    mask[action_id / 64] |= uint64_t{1} << (action_id % 64);
    return mask;
}

void add_to_mask(std::array<uint64_t, 2>& mask, size_t action_id) {
    mask[action_id / 64] |= uint64_t{1} << (action_id % 64);
}

bool all_zero(const ActionFeatures& features) {
    for (float value : features) {
        if (value != 0.0F) return false;
    }
    return true;
}

void add_concealed_tile(
    GYMJ::RL::CompactObservation& observation,
    uint8_t tile_id
) {
    ++observation.hand_counts[tile_id];
    ++observation.visible_counts[tile_id];
}

GYMJ::RL::CompactObservation observation_with_base_hand() {
    GYMJ::RL::CompactObservation observation;
    const std::array<uint8_t, 13> tiles = {
        0, 0, 0,
        1, 2, 3,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    for (uint8_t tile_id : tiles) {
        add_concealed_tile(observation, tile_id);
    }
    return observation;
}

bool encode_throws_invalid_argument(
    const GYMJ::RL::CompactObservation& observation,
    const std::array<uint64_t, 2>& legal_mask
) {
    try {
        GYMJ::RL::ActionFeatureEncoder::encode(observation, legal_mask);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

void test_feature_layout_is_stable() {
    using namespace GYMJ::RL;

    static_assert(ACTION_FEATURE_VERSION == 2);
    static_assert(ACTION_FEATURE_COUNT == 20);
    static_assert(feature_index(ActionFeatureIndex::PostStateKnown) == 0);
    static_assert(feature_index(ActionFeatureIndex::RequiresFollowUpDiscard) == 1);
    static_assert(feature_index(ActionFeatureIndex::MeldCountDelta) == 2);
    static_assert(feature_index(ActionFeatureIndex::NormalShanten) == 3);
    static_assert(feature_index(ActionFeatureIndex::SevenPairsShanten) == 4);
    static_assert(feature_index(ActionFeatureIndex::AllTripletsShanten) == 5);
    static_assert(feature_index(
        ActionFeatureIndex::NormalEffectiveDrawKinds
    ) == 6);
    static_assert(feature_index(
        ActionFeatureIndex::SevenPairsEffectiveDrawKinds
    ) == 7);
    static_assert(feature_index(
        ActionFeatureIndex::AllTripletsEffectiveDrawKinds
    ) == 8);
    static_assert(feature_index(
        ActionFeatureIndex::NormalEffectiveDrawCopies
    ) == 9);
    static_assert(feature_index(
        ActionFeatureIndex::SevenPairsEffectiveDrawCopies
    ) == 10);
    static_assert(feature_index(
        ActionFeatureIndex::AllTripletsEffectiveDrawCopies
    ) == 11);
    static_assert(feature_index(ActionFeatureIndex::WanFlushOffSuitTiles) == 12);
    static_assert(feature_index(ActionFeatureIndex::TongFlushOffSuitTiles) == 13);
    static_assert(feature_index(ActionFeatureIndex::TiaoFlushOffSuitTiles) == 14);
    static_assert(feature_index(ActionFeatureIndex::WanFlushMeldCompatible) == 15);
    static_assert(feature_index(ActionFeatureIndex::TongFlushMeldCompatible) == 16);
    static_assert(feature_index(ActionFeatureIndex::TiaoFlushMeldCompatible) == 17);
    static_assert(feature_index(ActionFeatureIndex::ConcealedTilesConsumed) == 18);
    static_assert(feature_index(ActionFeatureIndex::TakesReplacementDraw) == 19);
}

void test_all_27_discards_match_the_shanten_calculator() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        CompactObservation observation = observation_with_base_hand();
        add_concealed_tile(observation, static_cast<uint8_t>(tile_id));
        const auto original_hand = observation.hand_counts;
        const auto original_visible = observation.visible_counts;

        const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
            observation,
            mask_for(tile_id)
        );

        auto expected_hand = observation.hand_counts;
        --expected_hand[tile_id];
        const Algo::EffectiveDrawAnalysis expected =
            Algo::analyze_effective_draws(
                expected_hand.data(),
                observation.visible_counts.data(),
                0
            );
        const Algo::FlushProgress expected_flush =
            Algo::analyze_flush_progress(expected_hand.data(), 0);
        const ActionFeatures& action = encoded[tile_id];
        assert(action[feature_index(ActionFeatureIndex::NormalShanten)] ==
               expected.shanten.normal);
        assert(action[feature_index(ActionFeatureIndex::SevenPairsShanten)] ==
               expected.shanten.seven_pairs);
        assert(action[feature_index(ActionFeatureIndex::AllTripletsShanten)] ==
               expected.shanten.all_triplets);
        assert(action[feature_index(
            ActionFeatureIndex::NormalEffectiveDrawKinds
        )] == expected.normal.kinds);
        assert(action[feature_index(
            ActionFeatureIndex::SevenPairsEffectiveDrawKinds
        )] == expected.seven_pairs.kinds);
        assert(action[feature_index(
            ActionFeatureIndex::AllTripletsEffectiveDrawKinds
        )] == expected.all_triplets.kinds);
        assert(action[feature_index(
            ActionFeatureIndex::NormalEffectiveDrawCopies
        )] == expected.normal.copies);
        assert(action[feature_index(
            ActionFeatureIndex::SevenPairsEffectiveDrawCopies
        )] == expected.seven_pairs.copies);
        assert(action[feature_index(
            ActionFeatureIndex::AllTripletsEffectiveDrawCopies
        )] == expected.all_triplets.copies);
        assert(action[feature_index(ActionFeatureIndex::WanFlushOffSuitTiles)] ==
               expected_flush.off_suit_tiles[0]);
        assert(action[feature_index(ActionFeatureIndex::TongFlushOffSuitTiles)] ==
               expected_flush.off_suit_tiles[1]);
        assert(action[feature_index(ActionFeatureIndex::TiaoFlushOffSuitTiles)] ==
               expected_flush.off_suit_tiles[2]);
        assert(action[feature_index(ActionFeatureIndex::WanFlushMeldCompatible)] ==
               expected_flush.meld_compatible[0]);
        assert(action[feature_index(ActionFeatureIndex::TongFlushMeldCompatible)] ==
               expected_flush.meld_compatible[1]);
        assert(action[feature_index(ActionFeatureIndex::TiaoFlushMeldCompatible)] ==
               expected_flush.meld_compatible[2]);

        assert(action[feature_index(ActionFeatureIndex::PostStateKnown)] == 1.0F);
        assert(action[feature_index(
            ActionFeatureIndex::RequiresFollowUpDiscard
        )] == 0.0F);
        assert(action[feature_index(ActionFeatureIndex::MeldCountDelta)] == 0.0F);
        for (size_t action_id = 0; action_id < ACTION_SPACE_SIZE; ++action_id) {
            if (action_id != tile_id) assert(all_zero(encoded[action_id]));
        }
        assert(observation.hand_counts == original_hand);
        assert(observation.visible_counts == original_visible);
    }
}

void test_flush_features_follow_the_candidate_discard() {
    using namespace GYMJ::RL;

    CompactObservation observation;
    const std::array<uint8_t, 14> tiles = {
        0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10,
        18, 19,
    };
    for (uint8_t tile_id : tiles) {
        add_concealed_tile(observation, tile_id);
    }
    auto legal_mask = mask_for(0);
    legal_mask[0] |= uint64_t{1} << 9;

    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation,
        legal_mask
    );
    const ActionFeatures& discard_wan = encoded[0];
    const ActionFeatures& discard_tong = encoded[9];

    assert(discard_wan[feature_index(
        ActionFeatureIndex::WanFlushOffSuitTiles
    )] == 4.0F);
    assert(discard_tong[feature_index(
        ActionFeatureIndex::WanFlushOffSuitTiles
    )] == 3.0F);
    assert(discard_tong[feature_index(
        ActionFeatureIndex::TongFlushOffSuitTiles
    )] == 12.0F);
    assert(discard_tong[feature_index(
        ActionFeatureIndex::TiaoFlushOffSuitTiles
    )] == 11.0F);
    assert(discard_tong[feature_index(
        ActionFeatureIndex::WanFlushMeldCompatible
    )] == 1.0F);
    assert(discard_tong[feature_index(
        ActionFeatureIndex::TongFlushMeldCompatible
    )] == 1.0F);
    assert(discard_tong[feature_index(
        ActionFeatureIndex::TiaoFlushMeldCompatible
    )] == 1.0F);
}

void test_multiple_legal_discards_are_encoded_independently() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = observation_with_base_hand();
    add_concealed_tile(observation, 3);
    auto legal_mask = mask_for(0);
    add_to_mask(legal_mask, 3);
    add_to_mask(legal_mask, 26);

    const auto original_hand = observation.hand_counts;
    const auto original_visible = observation.visible_counts;
    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation,
        legal_mask
    );

    for (size_t tile_id : {size_t{0}, size_t{3}, size_t{26}}) {
        auto after_discard = observation.hand_counts;
        --after_discard[tile_id];
        const Algo::EffectiveDrawAnalysis expected =
            Algo::analyze_effective_draws(
                after_discard.data(),
                observation.visible_counts.data(),
                0
            );
        assert(encoded[tile_id][feature_index(
            ActionFeatureIndex::NormalShanten
        )] == expected.shanten.normal);
        assert(encoded[tile_id][feature_index(
            ActionFeatureIndex::NormalEffectiveDrawCopies
        )] == expected.normal.copies);
    }

    for (size_t action_id = 0; action_id < ACTION_SPACE_SIZE; ++action_id) {
        if (action_id == 0 || action_id == 3 || action_id == 26) continue;
        assert(all_zero(encoded[action_id]));
    }
    assert(observation.hand_counts == original_hand);
    assert(observation.visible_counts == original_visible);
}

void test_effective_draw_features_use_public_remaining_counts() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation;
    const std::array<uint8_t, 14> tiles = {
        0, 1, 2,
        6, 6, 7, 8,
        9, 10, 11,
        13,
        18, 19, 20,
    };
    for (uint8_t tile_id : tiles) {
        add_concealed_tile(observation, tile_id);
    }
    observation.visible_counts[6] = 3;

    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation,
        mask_for(13)
    );
    const ActionFeatures& action = encoded[13];
    assert(action[feature_index(ActionFeatureIndex::NormalShanten)] == 0.0F);
    assert(action[feature_index(
        ActionFeatureIndex::NormalEffectiveDrawKinds
    )] == 1.0F);
    assert(action[feature_index(
        ActionFeatureIndex::NormalEffectiveDrawCopies
    )] == 1.0F);
}

void test_fully_visible_draw_is_not_counted_as_effective() {
    using namespace GYMJ::RL;

    CompactObservation observation;
    const std::array<uint8_t, 14> tiles = {
        0, 1, 2,
        6, 6, 7, 8,
        9, 10, 11,
        13,
        18, 19, 20,
    };
    for (uint8_t tile_id : tiles) {
        add_concealed_tile(observation, tile_id);
    }
    observation.visible_counts[6] = 4;

    const ActionFeatures action = ActionFeatureEncoder::encode(
        observation,
        mask_for(13)
    )[13];
    assert(action[feature_index(ActionFeatureIndex::NormalShanten)] == 0.0F);
    assert(action[feature_index(
        ActionFeatureIndex::NormalEffectiveDrawKinds
    )] == 0.0F);
    assert(action[feature_index(
        ActionFeatureIndex::NormalEffectiveDrawCopies
    )] == 0.0F);
}

void test_route_specific_effective_draw_features_do_not_leak() {
    using namespace GYMJ::RL;

    CompactObservation seven_pairs;
    for (uint8_t tile_id : {
             0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 18, 18, 26, 8,
         }) {
        add_concealed_tile(seven_pairs, tile_id);
    }
    const ActionFeatures seven_pairs_action = ActionFeatureEncoder::encode(
        seven_pairs,
        mask_for(8)
    )[8];
    assert(seven_pairs_action[feature_index(
        ActionFeatureIndex::SevenPairsShanten
    )] == 0.0F);
    assert(seven_pairs_action[feature_index(
        ActionFeatureIndex::SevenPairsEffectiveDrawKinds
    )] == 1.0F);
    assert(seven_pairs_action[feature_index(
        ActionFeatureIndex::SevenPairsEffectiveDrawCopies
    )] == 3.0F);

    CompactObservation all_triplets;
    for (uint8_t tile_id : {
             0, 0, 0, 1, 1, 1, 9, 9, 9,
             18, 18, 18, 26, 8,
         }) {
        add_concealed_tile(all_triplets, tile_id);
    }
    const ActionFeatures all_triplets_action = ActionFeatureEncoder::encode(
        all_triplets,
        mask_for(8)
    )[8];
    assert(all_triplets_action[feature_index(
        ActionFeatureIndex::AllTripletsShanten
    )] == 0.0F);
    assert(all_triplets_action[feature_index(
        ActionFeatureIndex::AllTripletsEffectiveDrawKinds
    )] == 1.0F);
    assert(all_triplets_action[feature_index(
        ActionFeatureIndex::AllTripletsEffectiveDrawCopies
    )] == 3.0F);
}

void test_only_own_melds_disable_seven_pairs() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation opponent_meld_only;
    for (uint8_t tile_id = 0; tile_id < 7; ++tile_id) {
        add_concealed_tile(opponent_meld_only, tile_id);
        add_concealed_tile(opponent_meld_only, tile_id);
    }
    opponent_meld_only.meld_count = 1;
    opponent_meld_only.melds[0] = {
        12, 1, 2, MeldType::Pon, 1,
    };
    opponent_meld_only.visible_counts[12] = 3;
    const ActionFeatureMatrix opponent_meld_features =
        ActionFeatureEncoder::encode(
            opponent_meld_only,
            mask_for(6)
        );
    assert(opponent_meld_features[6][feature_index(
        ActionFeatureIndex::SevenPairsShanten
    )] == 0.0F);
    assert(opponent_meld_features[6][feature_index(
        ActionFeatureIndex::WanFlushMeldCompatible
    )] == 1.0F);
    assert(opponent_meld_features[6][feature_index(
        ActionFeatureIndex::TongFlushMeldCompatible
    )] == 1.0F);
    assert(opponent_meld_features[6][feature_index(
        ActionFeatureIndex::TiaoFlushMeldCompatible
    )] == 1.0F);

    CompactObservation observation;
    const std::array<uint8_t, 11> concealed_tiles = {
        0, 0, 0,
        1, 2, 3,
        5,
        9, 10, 11,
        18,
    };
    for (uint8_t tile_id : concealed_tiles) {
        add_concealed_tile(observation, tile_id);
    }
    observation.meld_count = 2;
    observation.melds[0] = {
        4, 1, 2, MeldType::Pon, 1,
    };
    observation.melds[1] = {
        7, 0, 3, MeldType::Pon, 1,
    };
    observation.visible_counts[4] = 3;
    observation.visible_counts[7] = 3;

    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation,
        mask_for(5)
    );
    auto expected_hand = observation.hand_counts;
    --expected_hand[5];
    const Algo::EffectiveDrawAnalysis expected =
        Algo::analyze_effective_draws(
            expected_hand.data(),
            observation.visible_counts.data(),
            1
        );

    const ActionFeatures& action = encoded[5];
    assert(expected.shanten.seven_pairs == 8);
    assert(action[feature_index(ActionFeatureIndex::NormalShanten)] ==
           expected.shanten.normal);
    assert(action[feature_index(ActionFeatureIndex::SevenPairsShanten)] == 8.0F);
    assert(action[feature_index(ActionFeatureIndex::AllTripletsShanten)] ==
           expected.shanten.all_triplets);
    assert(action[feature_index(
        ActionFeatureIndex::SevenPairsEffectiveDrawKinds
    )] == 0.0F);
    assert(action[feature_index(
        ActionFeatureIndex::SevenPairsEffectiveDrawCopies
    )] == 0.0F);
    assert(action[feature_index(ActionFeatureIndex::WanFlushMeldCompatible)] == 1.0F);
    assert(action[feature_index(ActionFeatureIndex::TongFlushMeldCompatible)] == 0.0F);
    assert(action[feature_index(ActionFeatureIndex::TiaoFlushMeldCompatible)] == 0.0F);
}

void test_fixed_meld_can_rule_out_an_otherwise_pure_flush() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation;
    for (uint8_t tile_id : {
             0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
         }) {
        add_concealed_tile(observation, tile_id);
    }
    observation.meld_count = 1;
    observation.melds[0] = {9, 0, 1, MeldType::Pon, 1};
    observation.visible_counts[9] = 3;

    const ActionFeatures action = ActionFeatureEncoder::encode(
        observation,
        mask_for(8)
    )[8];
    assert(action[feature_index(
        ActionFeatureIndex::WanFlushOffSuitTiles
    )] == 0.0F);
    assert(action[feature_index(
        ActionFeatureIndex::WanFlushMeldCompatible
    )] == 0.0F);
    assert(action[feature_index(
        ActionFeatureIndex::TongFlushMeldCompatible
    )] == 1.0F);
    assert(action[feature_index(
        ActionFeatureIndex::TiaoFlushMeldCompatible
    )] == 0.0F);
}

void test_mixed_own_meld_suits_rule_out_every_flush() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation;
    for (uint8_t tile_id : {0, 0, 0, 1, 2, 3, 18, 18}) {
        add_concealed_tile(observation, tile_id);
    }
    observation.meld_count = 2;
    observation.melds[0] = {4, 0, 1, MeldType::Pon, 1};
    observation.melds[1] = {13, 0, 2, MeldType::Daiminkan, 1};
    observation.visible_counts[4] = 3;
    observation.visible_counts[13] = 4;

    const ActionFeatures action = ActionFeatureEncoder::encode(
        observation,
        mask_for(18)
    )[18];
    assert(action[feature_index(ActionFeatureIndex::SevenPairsShanten)] == 8.0F);
    assert(action[feature_index(ActionFeatureIndex::WanFlushMeldCompatible)] == 0.0F);
    assert(action[feature_index(ActionFeatureIndex::TongFlushMeldCompatible)] == 0.0F);
    assert(action[feature_index(ActionFeatureIndex::TiaoFlushMeldCompatible)] == 0.0F);
}

void test_invalid_meld_slots_do_not_affect_features() {
    using namespace GYMJ::RL;

    CompactObservation observation = observation_with_base_hand();
    add_concealed_tile(observation, 3);
    observation.meld_count = 1;
    observation.melds[0].relative_owner = 0;
    observation.melds[0].tile = NO_TILE;
    observation.melds[0].valid = 0;

    const ActionFeatures action = ActionFeatureEncoder::encode(
        observation,
        mask_for(3)
    )[3];
    assert(action[feature_index(ActionFeatureIndex::SevenPairsShanten)] < 8.0F);
    assert(action[feature_index(ActionFeatureIndex::WanFlushMeldCompatible)] == 1.0F);
    assert(action[feature_index(ActionFeatureIndex::TongFlushMeldCompatible)] == 1.0F);
    assert(action[feature_index(ActionFeatureIndex::TiaoFlushMeldCompatible)] == 1.0F);
}

void test_illegal_and_unimplemented_actions_stay_empty() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = observation_with_base_hand();
    add_concealed_tile(observation, 3);
    auto legal_mask = mask_for(3);
    add_to_mask(legal_mask, PASS_ACTION_ID);

    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation,
        legal_mask
    );
    assert(!all_zero(encoded[3]));
    assert(all_zero(encoded[4]));
    assert(all_zero(encoded[PASS_ACTION_ID]));

    for (size_t action_id = DAIMINKAN_ACTION_ID + 1;
         action_id < ACTION_SPACE_SIZE;
         ++action_id) {
        const ActionFeatureMatrix unimplemented = ActionFeatureEncoder::encode(
            observation,
            mask_for(action_id)
        );
        for (const ActionFeatures& action : unimplemented) {
            assert(all_zero(action));
        }
    }
}

void test_pon_and_open_kan_encode_only_known_effects() {
    using namespace GYMJ::RL;

    CompactObservation observation = observation_with_base_hand();
    observation.current_discard = 0;
    ++observation.visible_counts[0];
    auto mask = mask_for(PON_ACTION_ID);
    add_to_mask(mask, DAIMINKAN_ACTION_ID);
    add_to_mask(mask, PASS_ACTION_ID);
    const auto original_hand = observation.hand_counts;

    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation, mask
    );
    const ActionFeatures& pon = encoded[PON_ACTION_ID];
    assert(pon[feature_index(ActionFeatureIndex::PostStateKnown)] == 0.0F);
    assert(pon[feature_index(ActionFeatureIndex::RequiresFollowUpDiscard)] == 1.0F);
    assert(pon[feature_index(ActionFeatureIndex::MeldCountDelta)] == 1.0F);
    assert(pon[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] == 2.0F);
    assert(pon[feature_index(ActionFeatureIndex::TakesReplacementDraw)] == 0.0F);

    const ActionFeatures& kan = encoded[DAIMINKAN_ACTION_ID];
    assert(kan[feature_index(ActionFeatureIndex::PostStateKnown)] == 0.0F);
    assert(kan[feature_index(ActionFeatureIndex::RequiresFollowUpDiscard)] == 0.0F);
    assert(kan[feature_index(ActionFeatureIndex::MeldCountDelta)] == 1.0F);
    assert(kan[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] == 3.0F);
    assert(kan[feature_index(ActionFeatureIndex::TakesReplacementDraw)] == 1.0F);

    for (size_t action_id = 0; action_id < GYMJ::RL::ACTION_SPACE_SIZE; ++action_id) {
        if (action_id != PON_ACTION_ID && action_id != DAIMINKAN_ACTION_ID) {
            assert(all_zero(encoded[action_id]));
        }
    }
    assert(observation.hand_counts == original_hand);
}

void test_concealed_and_added_kan_have_distinct_effects() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation concealed = observation_with_base_hand();
    add_concealed_tile(concealed, 0);
    const ActionFeatures ankan = ActionFeatureEncoder::encode(
        concealed, mask_for(ANKAN_FIRST)
    )[ANKAN_FIRST];
    assert(ankan[feature_index(ActionFeatureIndex::MeldCountDelta)] == 1.0F);
    assert(ankan[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] == 4.0F);
    assert(ankan[feature_index(ActionFeatureIndex::TakesReplacementDraw)] == 1.0F);

    CompactObservation added;
    for (uint8_t tile_id : {0, 0, 0, 1, 2, 3, 9, 10, 11, 18, 5}) {
        add_concealed_tile(added, tile_id);
    }
    added.meld_count = 1;
    added.melds[0] = {5, 0, 1, MeldType::Pon, 1};
    added.visible_counts[5] = 4;
    const ActionFeatures kakan = ActionFeatureEncoder::encode(
        added, mask_for(KAKAN_FIRST + 5)
    )[KAKAN_FIRST + 5];
    assert(kakan[feature_index(ActionFeatureIndex::MeldCountDelta)] == 0.0F);
    assert(kakan[feature_index(ActionFeatureIndex::ConcealedTilesConsumed)] == 1.0F);
    assert(kakan[feature_index(ActionFeatureIndex::TakesReplacementDraw)] == 1.0F);
}

void test_call_features_require_publicly_consistent_inputs() {
    using namespace GYMJ::RL;

    CompactObservation response = observation_with_base_hand();
    assert(encode_throws_invalid_argument(response, mask_for(PON_ACTION_ID)));
    response.current_discard = 4;
    assert(encode_throws_invalid_argument(response, mask_for(PON_ACTION_ID)));
    assert(encode_throws_invalid_argument(response, mask_for(DAIMINKAN_ACTION_ID)));
    response.current_discard = 0;
    assert(encode_throws_invalid_argument(response, mask_for(ANKAN_FIRST)));
    assert(encode_throws_invalid_argument(response, mask_for(KAKAN_FIRST + 4)));
}

void test_empty_legal_mask_produces_an_empty_matrix() {
    using namespace GYMJ::RL;

    CompactObservation observation = observation_with_base_hand();
    add_concealed_tile(observation, 3);
    const ActionFeatureMatrix encoded = ActionFeatureEncoder::encode(
        observation,
        {}
    );
    for (const ActionFeatures& action : encoded) {
        assert(all_zero(action));
    }
}

void test_legal_discard_must_exist_in_hand() {
    using namespace GYMJ::RL;

    CompactObservation observation;
    bool threw = false;
    try {
        ActionFeatureEncoder::encode(observation, mask_for(26));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_malformed_observations_are_rejected() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation excessive_hand;
    excessive_hand.hand_counts[0] = 5;
    excessive_hand.visible_counts[0] = 5;
    assert(encode_throws_invalid_argument(excessive_hand, mask_for(0)));

    CompactObservation too_few_visible = observation_with_base_hand();
    add_concealed_tile(too_few_visible, 3);
    --too_few_visible.visible_counts[0];
    assert(encode_throws_invalid_argument(too_few_visible, mask_for(3)));

    CompactObservation excessive_visible = observation_with_base_hand();
    add_concealed_tile(excessive_visible, 3);
    excessive_visible.visible_counts[26] = 5;
    assert(encode_throws_invalid_argument(excessive_visible, mask_for(3)));

    CompactObservation excessive_meld_count = observation_with_base_hand();
    add_concealed_tile(excessive_meld_count, 3);
    excessive_meld_count.meld_count = static_cast<uint8_t>(
        excessive_meld_count.melds.size() + 1
    );
    assert(encode_throws_invalid_argument(
        excessive_meld_count,
        mask_for(3)
    ));

    CompactObservation invalid_own_meld = observation_with_base_hand();
    add_concealed_tile(invalid_own_meld, 3);
    invalid_own_meld.meld_count = 1;
    invalid_own_meld.melds[0] = {
        NO_TILE, 0, 1, MeldType::Pon, 1,
    };
    assert(encode_throws_invalid_argument(invalid_own_meld, mask_for(3)));

    CompactObservation wrong_structural_size = observation_with_base_hand();
    assert(encode_throws_invalid_argument(
        wrong_structural_size,
        mask_for(3)
    ));
}

int main() {
    test_feature_layout_is_stable();
    test_all_27_discards_match_the_shanten_calculator();
    test_flush_features_follow_the_candidate_discard();
    test_multiple_legal_discards_are_encoded_independently();
    test_effective_draw_features_use_public_remaining_counts();
    test_fully_visible_draw_is_not_counted_as_effective();
    test_route_specific_effective_draw_features_do_not_leak();
    test_only_own_melds_disable_seven_pairs();
    test_fixed_meld_can_rule_out_an_otherwise_pure_flush();
    test_mixed_own_meld_suits_rule_out_every_flush();
    test_invalid_meld_slots_do_not_affect_features();
    test_illegal_and_unimplemented_actions_stay_empty();
    test_pon_and_open_kan_encode_only_known_effects();
    test_concealed_and_added_kan_have_distinct_effects();
    test_call_features_require_publicly_consistent_inputs();
    test_empty_legal_mask_produces_an_empty_matrix();
    test_legal_discard_must_exist_in_hand();
    test_malformed_observations_are_rejected();
    return 0;
}
