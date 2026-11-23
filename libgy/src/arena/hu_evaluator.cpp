#include "hu_evaluator.h"

#include "agari.h"

#include <algorithm>
#include <array>
#include <utility>

namespace GYMJ {
namespace {

using HandCounts = std::array<uint8_t, TILE_TYPE_COUNT>;

bool build_hands(
    const PlayerState& player,
    const HuContext& context,
    HandCounts& completed_hand,
    HandCounts& hand_before_hu
) {
    if (context.hu_tile.id >= TILE_TYPE_COUNT) return false;

    std::copy(
        player.tehai,
        player.tehai + TILE_TYPE_COUNT,
        completed_hand.begin()
    );
    hand_before_hu = completed_hand;

    for (uint8_t count : completed_hand) {
        if (count > MAX_TILE_COPIES) return false;
    }

    const uint8_t tile_id = context.hu_tile.id;
    if (context.method == HuMethod::Tsumo) {
        if (hand_before_hu[tile_id] == 0) return false;
        hand_before_hu[tile_id]--;
        return true;
    }

    if (completed_hand[tile_id] >= MAX_TILE_COPIES) return false;
    completed_hand[tile_id]++;
    return true;
}

bool has_concealed_quad(const HandCounts& hand) {
    return std::any_of(hand.begin(), hand.end(), [](uint8_t count) {
        return count == MAX_TILE_COPIES;
    });
}

bool is_di_long(
    const HandCounts& hand_before_hu,
    const std::vector<Meld>& melds,
    Tile hu_tile
) {
    if (melds.size() != 1 ||
        melds.front().type != MeldType::Pon ||
        melds.front().tile != hu_tile) {
        return false;
    }

    int pair_count = 0;
    int tile_count = 0;
    for (uint8_t count : hand_before_hu) {
        if (count == 0) continue;
        if (count != 2) return false;
        pair_count++;
        tile_count += count;
    }
    return pair_count == 5 && tile_count == 10;
}

bool is_single_wait(
    const HandCounts& hand_before_hu,
    int meld_count,
    Tile hu_tile
) {
    const auto waits = Algo::get_waits(hand_before_hu.data(), meld_count);
    return waits.size() == 1 && waits.front() == hu_tile;
}

SpecialHu effective_special(const HuContext& context) {
    if (context.method == HuMethod::QiangGang) {
        return context.special | SpecialHu::QiangGang;
    }
    return context.special;
}

HuFeatures make_features(
    HuPattern pattern,
    bool is_flush,
    bool single_wait,
    const HuContext& context
) {
    HuFeatures features;
    features.pattern = pattern;
    features.is_flush = is_flush;
    features.is_single_wait = single_wait;
    features.ready_kind = context.hu_player_ready_kind;
    features.timing = context.timing;
    features.special = effective_special(context);
    return features;
}

bool is_better_candidate(
    const HuCandidate& candidate,
    const HuCandidate& current_best
) {
    if (candidate.score.hu_points != current_best.score.hu_points) {
        return candidate.score.hu_points > current_best.score.hu_points;
    }
    if (candidate.score.total_fan != current_best.score.total_fan) {
        return candidate.score.total_fan > current_best.score.total_fan;
    }
    if (candidate.score.shape_fan != current_best.score.shape_fan) {
        return candidate.score.shape_fan > current_best.score.shape_fan;
    }
    return static_cast<uint8_t>(candidate.features.pattern) >
           static_cast<uint8_t>(current_best.features.pattern);
}

void add_candidate(HuEvaluation& result, const HuFeatures& features) {
    HuCandidate candidate{features, GuiyangScoring::calculate(features)};
    result.candidates.push_back(std::move(candidate));
    const std::size_t candidate_index = result.candidates.size() - 1;

    if (result.best_candidate_index == HuEvaluation::NO_CANDIDATE ||
        is_better_candidate(
            result.candidates[candidate_index],
            result.candidates[result.best_candidate_index]
        )) {
        result.best_candidate_index = candidate_index;
    }
}

bool has_bean(const std::vector<Meld>& melds) {
    return std::any_of(melds.begin(), melds.end(), [](const Meld& meld) {
        return meld.type == MeldType::AnKan ||
               meld.type == MeldType::Kakan ||
               meld.type == MeldType::Daiminkan;
    });
}

bool has_big_hand_candidate(const HuEvaluation& result) {
    return std::any_of(
        result.candidates.begin(),
        result.candidates.end(),
        [](const HuCandidate& candidate) {
            return candidate.features.pattern != HuPattern::Standard ||
                   candidate.features.is_flush;
        }
    );
}

bool is_legal(
    const PlayerState& player,
    const HuContext& context,
    const HuEvaluation& result
) {
    if (!result.shape_complete) return false;
    if (context.method == HuMethod::Tsumo ||
        context.method == HuMethod::QiangGang) {
        return true;
    }
    if (context.bypass_ron_passport ||
        context.hu_player_ready_kind != ReadyKind::None ||
        context.source_ready_kind != ReadyKind::None ||
        has_bean(player.melds[0]) ||
        has_big_hand_candidate(result)) {
        return true;
    }
    return has_special(context.special, SpecialHu::RePao) ||
           has_special(context.special, SpecialHu::QiangGang) ||
           has_special(context.special, SpecialHu::ShaBao);
}

} // namespace

const HuCandidate* HuEvaluation::best_candidate() const noexcept {
    if (best_candidate_index == NO_CANDIDATE ||
        best_candidate_index >= candidates.size()) {
        return nullptr;
    }
    return &candidates[best_candidate_index];
}

HuEvaluation HuEvaluator::evaluate(
    const PlayerState& player,
    const HuContext& context
) {
    HuEvaluation result;
    HandCounts completed_hand{};
    HandCounts hand_before_hu{};
    if (!build_hands(
            player,
            context,
            completed_hand,
            hand_before_hu
        )) {
        return result;
    }

    const auto& melds = player.melds[0];
    const int meld_count = static_cast<int>(melds.size());
    const bool flush = Algo::is_flush(completed_hand.data(), melds);
    const bool standard =
        Algo::is_standard_hu(completed_hand.data(), meld_count);
    const bool seven_pairs =
        melds.empty() && Algo::is_seven_pairs(completed_hand.data());
    const bool single_wait = standard && is_single_wait(
        hand_before_hu,
        meld_count,
        context.hu_tile
    );

    if (standard) {
        add_candidate(
            result,
            make_features(
                HuPattern::Standard,
                flush,
                single_wait,
                context
            )
        );
    }
    if (Algo::is_all_triplets(completed_hand.data(), melds)) {
        add_candidate(
            result,
            make_features(
                HuPattern::AllTriplets,
                flush,
                single_wait,
                context
            )
        );
    }
    if (seven_pairs) {
        add_candidate(
            result,
            make_features(HuPattern::SevenPairs, flush, false, context)
        );
        if (has_concealed_quad(completed_hand)) {
            add_candidate(
                result,
                make_features(
                    HuPattern::DragonSevenPairs,
                    flush,
                    false,
                    context
                )
            );
        }
    }
    if (is_di_long(hand_before_hu, melds, context.hu_tile)) {
        add_candidate(
            result,
            make_features(HuPattern::DiLong, flush, false, context)
        );
    }

    result.shape_complete = !result.candidates.empty();
    result.legal = is_legal(player, context, result);
    return result;
}

} // namespace GYMJ
