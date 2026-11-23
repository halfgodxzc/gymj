#include "guiyang_rules.h"

#include "hu_evaluator.h"

namespace GYMJ {

bool GuiyangRules::can_ron(
    const PlayerState& player,
    Tile hu_tile,
    bool bypass_passport
) {
    HuContext context;
    context.method = HuMethod::Ron;
    context.hu_tile = hu_tile;
    context.bypass_ron_passport = bypass_passport;
    return HuEvaluator::evaluate(player, context).legal;
}

int GuiyangRules::response_priority(ActionType type) {
    switch (type) {
        case ActionType::Ron:
            return 3;
        case ActionType::Daiminkan:
            return 2;
        case ActionType::Pon:
            return 1;
        default:
            return 0;
    }
}

} // namespace GYMJ
