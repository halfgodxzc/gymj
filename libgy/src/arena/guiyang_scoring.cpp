#include "guiyang_scoring.h"

#include <stdexcept>

namespace GYMJ {
namespace {

void add_item(
    HuScore& score,
    FanType type,
    int fan,
    int& subtotal
) {
    score.items.push_back({type, fan});
    subtotal += fan;
}

bool has_dadui_component(const HuFeatures& features) {
    return features.pattern == HuPattern::AllTriplets;
}

bool has_qing_component(const HuFeatures& features) {
    return features.is_flush;
}

void validate(const HuFeatures& features) {
    if (features.is_su_gua_dou) {
        const bool incompatible =
            features.pattern != HuPattern::Standard ||
            features.is_flush ||
            features.ready_kind != ReadyKind::None ||
            features.timing != HuTiming::Normal ||
            features.special != SpecialHu::None;
        if (incompatible) {
            throw std::invalid_argument(
                "SuGuaDou cannot combine with another big hand"
            );
        }
    }

    if (features.is_single_wait &&
        features.pattern != HuPattern::Standard &&
        features.pattern != HuPattern::AllTriplets) {
        throw std::invalid_argument(
            "single-wait scoring only applies to standard hands"
        );
    }
}

void add_shape_fan(const HuFeatures& features, HuScore& score) {
    if (features.is_flush && features.is_single_wait) {
        add_item(
            score,
            FanType::QingDanDiao,
            6,
            score.shape_fan
        );
        return;
    }

    switch (features.pattern) {
        case HuPattern::Standard:
            break;
        case HuPattern::AllTriplets:
            add_item(score, FanType::DaDuiZi, 2, score.shape_fan);
            break;
        case HuPattern::SevenPairs:
            add_item(score, FanType::QiDui, 3, score.shape_fan);
            break;
        case HuPattern::DragonSevenPairs:
            add_item(score, FanType::LongQiDui, 6, score.shape_fan);
            break;
        case HuPattern::DiLong:
            add_item(score, FanType::DiLong, 6, score.shape_fan);
            break;
    }

    if (features.is_flush) {
        add_item(score, FanType::QingYiSe, 3, score.shape_fan);
    }
}

void add_special_fan(const HuFeatures& features, HuScore& score) {
    if (has_special(features.special, SpecialHu::GangKai) &&
        !has_dadui_component(features)) {
        add_item(score, FanType::GangKai, 2, score.special_fan);
    }
    if (has_special(features.special, SpecialHu::RePao) &&
        !has_dadui_component(features)) {
        add_item(score, FanType::RePao, 2, score.special_fan);
    }

    const bool has_qing_equivalent =
        has_special(features.special, SpecialHu::QiangGang) ||
        has_special(features.special, SpecialHu::ShaBao);
    if (has_qing_equivalent && !has_qing_component(features)) {
        const FanType type = has_special(
            features.special,
            SpecialHu::QiangGang
        ) ? FanType::QiangGang : FanType::ShaBao;
        add_item(score, type, 3, score.special_fan);
    }
}

} // namespace

HuScore GuiyangScoring::calculate(const HuFeatures& features) {
    validate(features);

    HuScore score;
    if (features.is_su_gua_dou) {
        add_item(score, FanType::SuGuaDou, 2, score.shape_fan);
        score.total_fan = 2;
        score.hu_points = score.total_fan * GUIYANG_HU_BASE_POINTS;
        return score;
    }

    add_shape_fan(features, score);
    add_special_fan(features, score);

    switch (features.ready_kind) {
        case ReadyKind::None:
            break;
        case ReadyKind::TianTing:
            add_item(score, FanType::TianTing, 5, score.ready_fan);
            break;
        case ReadyKind::BaoTing:
            add_item(score, FanType::BaoTing, 3, score.ready_fan);
            break;
    }

    switch (features.timing) {
        case HuTiming::Normal:
            break;
        case HuTiming::DiHu:
            add_item(score, FanType::DiHu, 6, score.timing_fan);
            break;
        case HuTiming::TianHu:
            add_item(score, FanType::TianHu, 9, score.timing_fan);
            break;
    }

    score.total_fan =
        score.shape_fan +
        score.ready_fan +
        score.timing_fan +
        score.special_fan;
    if (score.total_fan == 0) {
        add_item(score, FanType::PingHu, 1, score.shape_fan);
        score.total_fan = 1;
    }
    score.hu_points = score.total_fan * GUIYANG_HU_BASE_POINTS;
    return score;
}

} // namespace GYMJ
