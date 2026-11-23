#include "pattern_feature_encoder.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace {

using GYMJ::RL::CompactObservation;

CompactObservation make_hand(std::initializer_list<uint8_t> tiles) {
    CompactObservation observation;
    for (uint8_t tile_id : tiles) {
        assert(tile_id < GYMJ::TILE_TYPE_COUNT);
        ++observation.hand_counts[tile_id];
        ++observation.visible_counts[tile_id];
    }
    return observation;
}

size_t route_index(GYMJ::RL::PatternRoute route) {
    return static_cast<size_t>(route);
}

bool throws_invalid(const CompactObservation& observation) {
    try {
        GYMJ::RL::PatternFeatureEncoder::encode(observation);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

void test_route_features_preserve_each_effective_tile() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2,
        6, 6, 7, 8,
        9, 10, 11,
        18, 19, 20,
    });
    const auto encoded = PatternFeatureEncoder::encode(observation);
    const auto& normal = encoded.hand.routes[
        route_index(PatternRoute::Normal)
    ];

    assert(encoded.hand.post_discard_state == 1);
    assert(normal.shanten == 0);
    assert(normal.effective_draws.kinds == 1);
    assert(normal.effective_draws.copies == 2);
    assert(normal.effective_draws.remaining_by_tile[6] == 2);
    assert(encoded.hand.shape_wait_mask == (uint32_t{1} << 6));
    assert(encoded.hand.live_wait_mask == (uint32_t{1} << 6));
    assert(encoded.hand.live_wait_kinds == 1);
    assert(encoded.hand.live_wait_copies == 2);
    assert(encoded.hand.wait_values[6].base_fan == 1);
    assert(encoded.hand.wait_values[6].base_hu_points == 5);
}

void test_wait_value_keeps_overlapping_big_hand_patterns() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    const CompactObservation observation = make_hand({
        0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6,
    });
    const auto encoded = PatternFeatureEncoder::encode_hand(observation);
    const WaitValueFeatures& wait = encoded.wait_values[6];

    assert((wait.pattern_mask & PatternStandard) != 0);
    assert((wait.pattern_mask & PatternSevenPairs) != 0);
    assert((wait.pattern_mask & PatternFlush) != 0);
    assert(wait.base_fan == 6);
    assert(wait.base_hu_points == 30);
    assert(wait.remaining_copies == 3);
}

void test_open_all_triplets_disables_seven_pairs() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        1, 1, 1,
        9, 9, 9,
        18, 18, 18,
        22,
    });
    observation.meld_count = 1;
    observation.melds[0] = {0, 0, 1, MeldType::Pon, 1};
    observation.visible_counts[0] = 3;

    const auto encoded = PatternFeatureEncoder::encode_hand(observation);
    assert(encoded.routes[
        route_index(PatternRoute::SevenPairs)
    ].shanten == 8);
    assert(encoded.wait_values[22].base_fan == 2);
    assert((encoded.wait_values[22].pattern_mask & PatternAllTriplets) != 0);
    assert((encoded.wait_values[22].pattern_mask & PatternSevenPairs) == 0);
}

void test_turn_hand_has_route_distance_but_no_draw_analysis() {
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26, 26,
    });
    const auto encoded = PatternFeatureEncoder::encode_hand(observation);

    assert(encoded.post_discard_state == 0);
    assert(encoded.routes[route_index(PatternRoute::Normal)].shanten == -1);
    for (const auto& route : encoded.routes) {
        assert(route.effective_draws.kinds == 0);
        assert(route.effective_draws.copies == 0);
    }
    assert(encoded.shape_wait_mask == 0);
    assert(encoded.live_wait_mask == 0);
}

void test_flush_features_include_distance_and_specific_draws() {
    using namespace GYMJ;
    using namespace GYMJ::Algo;
    using namespace GYMJ::RL;

    const CompactObservation observation = make_hand({
        1, 1, 2, 4, 5, 6, 8,
        18, 19, 20, 22, 23, 24,
    });
    const auto encoded = PatternFeatureEncoder::encode_hand(observation);
    const auto& wan = encoded.flushes[WAN_FLUSH_INDEX];
    const auto& tiao = encoded.flushes[TIAO_FLUSH_INDEX];

    assert(wan.concealed_tiles == 7);
    assert(wan.off_suit_tiles == 6);
    assert(wan.shanten == 5);
    assert(tiao.concealed_tiles == 6);
    assert(tiao.off_suit_tiles == 7);
    assert(tiao.shanten == 4);
    assert(encoded.best_flush_suit_mask == flush_suit_bit(TIAO_FLUSH_INDEX));

    uint8_t copy_sum = 0;
    for (uint8_t copies : tiao.effective_draws.remaining_by_tile) {
        copy_sum = static_cast<uint8_t>(copy_sum + copies);
    }
    assert(copy_sum == tiao.effective_draws.copies);
}

void test_public_player_summary_preserves_ownership_and_responsibility() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    observation.discard_count = 3;
    observation.discards[0] = {YAO_JI_TILE_ID, 1, 1};
    observation.discards[1] = {BA_TONG_TILE_ID, 1, 0};
    observation.discards[2] = {BA_TONG_TILE_ID, 2, 0};
    observation.meld_count = 2;
    observation.melds[0] = {
        YAO_JI_TILE_ID, 3, 1, MeldType::Pon, 1,
    };
    observation.melds[1] = {20, 2, 2, MeldType::AnKan, 1};
    observation.visible_counts[YAO_JI_TILE_ID] = 3;
    observation.visible_counts[BA_TONG_TILE_ID] = 2;
    observation.visible_counts[20] = 4;

    const auto encoded = PatternFeatureEncoder::encode(observation);
    const auto& p1 = encoded.players[1];
    const auto& p2 = encoded.players[2];
    const auto& p3 = encoded.players[3];

    assert(p1.called_discard_counts[YAO_JI_TILE_ID] == 1);
    assert(p1.river_counts[BA_TONG_TILE_ID] == 1);
    assert(p1.chongfeng_chicken_count == 1);
    assert(p1.responsibility_chicken_given_count == 1);
    assert(p1.fixed_chicken_count == 1);

    assert(p2.river_counts[BA_TONG_TILE_ID] == 1);
    assert(p2.chongfeng_chicken_count == 1);
    assert(p2.bean_type_counts[0] == 1);
    assert(p2.meld_tile_counts[20] == 4);

    assert(p3.meld_tile_counts[YAO_JI_TILE_ID] == 3);
    assert(p3.fixed_chicken_count == 3);
    assert(p3.responsibility_chicken_received_count == 1);
}

void test_public_chongfeng_uses_global_first_and_opening_follow_discards() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    const std::array<DiscardObservation, 7> discards{{
        {0, 1, 0},
        {1, 2, 0},
        {YAO_JI_TILE_ID, 1, 0},
        {BA_TONG_TILE_ID, 2, 0},
        {BA_TONG_TILE_ID, 3, 0},
        {2, 0, 0},
        {YAO_JI_TILE_ID, 0, 0},
    }};
    observation.discard_count = discards.size();
    for (size_t index = 0; index < discards.size(); ++index) {
        observation.discards[index] = discards[index];
        ++observation.visible_counts[discards[index].tile];
    }

    const auto encoded = PatternFeatureEncoder::encode(observation);
    assert(encoded.players[0].chongfeng_chicken_count == 0);
    assert(encoded.players[1].chongfeng_chicken_count == 1);
    assert(encoded.players[2].chongfeng_chicken_count == 1);
    assert(encoded.players[3].chongfeng_chicken_count == 1);
}

void test_public_summary_keeps_both_chongfeng_chicken_types() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    observation.discard_count = 2;
    observation.discards[0] = {YAO_JI_TILE_ID, 0, 0};
    observation.discards[1] = {BA_TONG_TILE_ID, 0, 0};
    observation.visible_counts[YAO_JI_TILE_ID] = 1;
    observation.visible_counts[BA_TONG_TILE_ID] = 1;

    const auto encoded = PatternFeatureEncoder::encode(observation);
    assert(encoded.players[0].chongfeng_chicken_count == 2);
}

void test_pending_kakan_adds_the_exposed_fourth_tile() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    });
    observation.meld_count = 1;
    observation.melds[0] = {14, 2, 1, MeldType::Pon, 1};
    observation.phase = RoundPhase::AwaitKakanResponses;
    observation.current_discard = 14;
    observation.current_discard_actor = 2;
    observation.visible_counts[14] = 4;

    const auto encoded = PatternFeatureEncoder::encode(observation);
    const auto& owner = encoded.players[2];

    assert(owner.meld_tile_counts[14] == 3);
    assert(owner.public_owned_tile_counts[14] == 4);
}

void test_public_chicken_value_uses_only_known_tenpai_information() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation observation = make_hand({
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        BA_TONG_TILE_ID,
        YAO_JI_TILE_ID, 19, 20,
    });
    observation.shanten = 0;
    observation.discard_count = 2;
    observation.discards[0] = {YAO_JI_TILE_ID, 1, 0};
    observation.discards[1] = {BA_TONG_TILE_ID, 2, 0};
    ++observation.visible_counts[YAO_JI_TILE_ID];
    ++observation.visible_counts[BA_TONG_TILE_ID];
    observation.ready_kinds[1] = static_cast<uint8_t>(ReadyKind::BaoTing);

    auto encoded = PatternFeatureEncoder::encode(observation);
    assert(encoded.players[0].fixed_chicken_count == 2);
    assert(encoded.players[0].known_tenpai_status == 1);
    assert(encoded.players[0].signed_fixed_chicken_count == 2);
    assert(encoded.players[1].fixed_chicken_count == 1);
    assert(encoded.players[1].known_tenpai_status == 1);
    assert(encoded.players[1].signed_fixed_chicken_count == 1);
    assert(encoded.players[2].fixed_chicken_count == 1);
    assert(encoded.players[2].known_tenpai_status == 0);
    assert(encoded.players[2].signed_fixed_chicken_count == 0);

    observation.shanten = 1;
    encoded = PatternFeatureEncoder::encode(observation);
    assert(encoded.players[0].known_tenpai_status == -1);
    assert(encoded.players[0].signed_fixed_chicken_count == -2);
}

void test_malformed_inputs_are_rejected() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    CompactObservation wrong_size = make_hand({0, 1, 2});
    assert(throws_invalid(wrong_size));

    CompactObservation impossible = make_hand({
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    impossible.visible_counts[0] = 0;
    assert(throws_invalid(impossible));

    CompactObservation called_without_meld = make_hand({
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    });
    called_without_meld.discard_count = 1;
    called_without_meld.discards[0] = {YAO_JI_TILE_ID, 1, 1};
    assert(throws_invalid(called_without_meld));
}

int main() {
    static_assert(GYMJ::RL::PATTERN_FEATURE_VERSION == 2);
    test_route_features_preserve_each_effective_tile();
    test_wait_value_keeps_overlapping_big_hand_patterns();
    test_open_all_triplets_disables_seven_pairs();
    test_turn_hand_has_route_distance_but_no_draw_analysis();
    test_flush_features_include_distance_and_specific_draws();
    test_public_player_summary_preserves_ownership_and_responsibility();
    test_public_chongfeng_uses_global_first_and_opening_follow_discards();
    test_public_summary_keeps_both_chongfeng_chicken_types();
    test_pending_kakan_adds_the_exposed_fourth_tile();
    test_public_chicken_value_uses_only_known_tenpai_information();
    test_malformed_inputs_are_rejected();
    return 0;
}
