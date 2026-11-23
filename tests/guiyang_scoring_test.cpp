#include "guiyang_scoring.h"

#include <cassert>
#include <stdexcept>

namespace {

GYMJ::HuScore score(
    GYMJ::HuPattern pattern = GYMJ::HuPattern::Standard,
    bool flush = false,
    bool single_wait = false,
    bool reported = false,
    GYMJ::HuTiming timing = GYMJ::HuTiming::Normal,
    GYMJ::SpecialHu special = GYMJ::SpecialHu::None,
    bool su_gua_dou = false
) {
    GYMJ::HuFeatures features;
    features.pattern = pattern;
    features.is_flush = flush;
    features.is_single_wait = single_wait;
    features.ready_kind = reported
        ? GYMJ::ReadyKind::BaoTing
        : GYMJ::ReadyKind::None;
    features.is_su_gua_dou = su_gua_dou;
    features.timing = timing;
    features.special = special;
    return GYMJ::GuiyangScoring::calculate(features);
}

void assert_score(const GYMJ::HuScore& result, int fan) {
    assert(result.total_fan == fan);
    assert(result.hu_points == fan * GYMJ::GUIYANG_HU_BASE_POINTS);
}

void test_complete_shape_table() {
    using namespace GYMJ;

    assert_score(score(), 1);
    assert_score(score(HuPattern::AllTriplets), 2);
    assert_score(score(HuPattern::SevenPairs), 3);
    assert_score(score(HuPattern::Standard, true), 3);
    assert_score(score(HuPattern::AllTriplets, true), 5);
    assert_score(score(HuPattern::DiLong), 6);
    assert_score(score(HuPattern::DragonSevenPairs), 6);
    assert_score(score(HuPattern::SevenPairs, true), 6);
    assert_score(score(HuPattern::Standard, true, true), 6);
    assert_score(score(HuPattern::DragonSevenPairs, true), 9);
    assert_score(score(HuPattern::DiLong, true), 9);
}

void test_ready_and_timing_stack_with_shapes() {
    using namespace GYMJ;

    assert_score(score(
        HuPattern::Standard, false, false, true
    ), 3);
    assert_score(score(
        HuPattern::AllTriplets, false, false, true
    ), 5);
    assert_score(score(
        HuPattern::SevenPairs, false, false, true
    ), 6);
    assert_score(score(
        HuPattern::DragonSevenPairs, false, false, true
    ), 9);

    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::DiHu
    ), 6);
    assert_score(score(
        HuPattern::AllTriplets,
        false,
        false,
        false,
        HuTiming::DiHu
    ), 8);
    assert_score(score(
        HuPattern::AllTriplets,
        true,
        false,
        false,
        HuTiming::TianHu
    ), 14);
    assert_score(score(
        HuPattern::DragonSevenPairs,
        true,
        false,
        false,
        HuTiming::TianHu
    ), 18);
    assert_score(score(
        HuPattern::AllTriplets,
        false,
        false,
        true,
        HuTiming::DiHu
    ), 11);
}

void test_special_hu_equivalents_and_deduplication() {
    using namespace GYMJ;

    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::GangKai
    ), 2);
    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::RePao
    ), 2);
    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::QiangGang
    ), 3);
    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::ShaBao
    ), 3);

    assert_score(score(
        HuPattern::Standard,
        true,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::GangKai
    ), 5);
    assert_score(score(
        HuPattern::AllTriplets,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::QiangGang
    ), 5);
    assert_score(score(
        HuPattern::AllTriplets,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::GangKai
    ), 2);
    assert_score(score(
        HuPattern::Standard,
        true,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::QiangGang
    ), 3);
}

void test_su_gua_dou_is_exclusive() {
    using namespace GYMJ;

    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::None,
        true
    ), 2);

    bool threw = false;
    try {
        score(
            HuPattern::Standard,
            true,
            false,
            false,
            HuTiming::Normal,
            SpecialHu::None,
            true
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        score(
            HuPattern::Standard,
            false,
            false,
            true,
            HuTiming::Normal,
            SpecialHu::None,
            true
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_hot_cannon_and_kill_ready_can_share_one_context() {
    using namespace GYMJ;
    assert_score(score(
        HuPattern::Standard,
        false,
        false,
        false,
        HuTiming::Normal,
        SpecialHu::RePao | SpecialHu::ShaBao
    ), 5);
}

void test_tianting_stacks_with_kill_ready() {
    using namespace GYMJ;

    HuFeatures features;
    features.ready_kind = ReadyKind::TianTing;
    features.special = SpecialHu::ShaBao;
    const auto result = GuiyangScoring::calculate(features);

    assert(result.ready_fan == 5);
    assert(result.special_fan == 3);
    assert_score(result, 8);
}

} // namespace

int main() {
    test_complete_shape_table();
    test_ready_and_timing_stack_with_shapes();
    test_special_hu_equivalents_and_deduplication();
    test_su_gua_dou_is_exclusive();
    test_hot_cannon_and_kill_ready_can_share_one_context();
    test_tianting_stacks_with_kill_ready();
    return 0;
}
