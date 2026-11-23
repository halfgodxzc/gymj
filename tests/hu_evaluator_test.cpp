#include "hu_evaluator.h"

#include <algorithm>
#include <cassert>
#include <initializer_list>

namespace {

void add_tiles(
    uint8_t hand[GYMJ::TILE_TYPE_COUNT],
    std::initializer_list<int> tiles
) {
    for (int tile : tiles) {
        hand[tile]++;
    }
}

GYMJ::HuContext context_for(
    GYMJ::HuMethod method,
    uint8_t hu_tile
) {
    GYMJ::HuContext context;
    context.method = method;
    context.hu_tile = GYMJ::Tile{hu_tile};
    return context;
}

bool has_candidate(
    const GYMJ::HuEvaluation& evaluation,
    GYMJ::HuPattern pattern
) {
    return std::any_of(
        evaluation.candidates.begin(),
        evaluation.candidates.end(),
        [&](const GYMJ::HuCandidate& candidate) {
            return candidate.features.pattern == pattern;
        }
    );
}

void test_all_triplets_is_preserved_beside_standard() {
    using namespace GYMJ;

    PlayerState player(0);
    add_tiles(player.tehai, {
        0, 0, 0,
        1, 1, 1,
        9, 9, 9,
        18, 18, 18,
        22, 22,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Tsumo, 22)
    );
    assert(evaluation.shape_complete);
    assert(evaluation.legal);
    assert(has_candidate(evaluation, HuPattern::Standard));
    assert(has_candidate(evaluation, HuPattern::AllTriplets));
    assert(evaluation.best_candidate() != nullptr);
    assert(
        evaluation.best_candidate()->features.pattern ==
        HuPattern::AllTriplets
    );
    assert(evaluation.best_candidate()->score.total_fan == 2);
}

void test_open_all_triplets_uses_melds_and_concealed_counts() {
    using namespace GYMJ;

    PlayerState player(0);
    player.melds[0].push_back({MeldType::Pon, Tile{0}, 1});
    add_tiles(player.tehai, {
        1, 1, 1,
        9, 9, 9,
        18, 18, 18,
        22, 22,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Tsumo, 22)
    );
    assert(evaluation.legal);
    assert(has_candidate(evaluation, HuPattern::AllTriplets));
    assert(evaluation.best_candidate() != nullptr);
    assert(
        evaluation.best_candidate()->features.pattern ==
        HuPattern::AllTriplets
    );
}

void test_seven_pairs_and_standard_overlap_selects_higher_score() {
    using namespace GYMJ;

    PlayerState player(0);
    add_tiles(player.tehai, {
        0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Tsumo, 6)
    );
    assert(has_candidate(evaluation, HuPattern::Standard));
    assert(has_candidate(evaluation, HuPattern::SevenPairs));
    assert(evaluation.best_candidate() != nullptr);
    assert(
        evaluation.best_candidate()->features.pattern ==
        HuPattern::SevenPairs
    );
    assert(evaluation.best_candidate()->score.total_fan == 6);
}

void test_dragon_seven_pairs_detects_concealed_quad() {
    using namespace GYMJ;

    PlayerState player(0);
    add_tiles(player.tehai, {
        0, 0, 0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Tsumo, 5)
    );
    assert(has_candidate(evaluation, HuPattern::SevenPairs));
    assert(has_candidate(evaluation, HuPattern::DragonSevenPairs));
    assert(evaluation.best_candidate() != nullptr);
    assert(
        evaluation.best_candidate()->features.pattern ==
        HuPattern::DragonSevenPairs
    );
    assert(evaluation.best_candidate()->score.total_fan == 9);
}

void test_di_long_uses_pon_tile_and_five_pairs() {
    using namespace GYMJ;

    PlayerState player(0);
    player.melds[0].push_back({MeldType::Pon, Tile{0}, 1});
    add_tiles(player.tehai, {
        1, 1,
        2, 2,
        9, 9,
        10, 10,
        18, 18,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Ron, 0)
    );
    assert(evaluation.shape_complete);
    assert(evaluation.legal);
    assert(evaluation.candidates.size() == 1);
    assert(has_candidate(evaluation, HuPattern::DiLong));
    assert(evaluation.best_candidate()->score.total_fan == 6);
}

void test_ordinary_ron_needs_passport_but_qianggang_does_not() {
    using namespace GYMJ;

    PlayerState player(0);
    add_tiles(player.tehai, {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    });

    auto ron_context = context_for(HuMethod::Ron, 26);
    auto evaluation = HuEvaluator::evaluate(player, ron_context);
    assert(evaluation.shape_complete);
    assert(!evaluation.legal);

    ron_context.source_ready_kind = ReadyKind::BaoTing;
    evaluation = HuEvaluator::evaluate(player, ron_context);
    assert(evaluation.legal);

    const auto qianggang = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::QiangGang, 26)
    );
    assert(qianggang.legal);
    assert(qianggang.best_candidate() != nullptr);
    assert(qianggang.best_candidate()->score.total_fan == 3);
}

void test_bean_allows_ordinary_ron() {
    using namespace GYMJ;

    PlayerState player(0);
    player.melds[0].push_back({MeldType::AnKan, Tile{0}, 0});
    add_tiles(player.tehai, {
        1, 2, 3,
        9, 10, 11,
        18, 19, 20,
        26,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Ron, 26)
    );
    assert(evaluation.shape_complete);
    assert(evaluation.legal);
}

void test_tsumo_uses_the_already_drawn_tile() {
    using namespace GYMJ;

    PlayerState player(0);
    add_tiles(player.tehai, {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26, 26,
    });

    const auto evaluation = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Tsumo, 26)
    );
    assert(evaluation.shape_complete);
    assert(evaluation.legal);

    const auto invalid = HuEvaluator::evaluate(
        player,
        context_for(HuMethod::Tsumo, 25)
    );
    assert(!invalid.shape_complete);
    assert(!invalid.legal);
}

} // namespace

int main() {
    test_all_triplets_is_preserved_beside_standard();
    test_open_all_triplets_uses_melds_and_concealed_counts();
    test_seven_pairs_and_standard_overlap_selects_higher_score();
    test_dragon_seven_pairs_detects_concealed_quad();
    test_di_long_uses_pon_tile_and_five_pairs();
    test_ordinary_ron_needs_passport_but_qianggang_does_not();
    test_bean_allows_ordinary_ron();
    test_tsumo_uses_the_already_drawn_tile();
    return 0;
}
