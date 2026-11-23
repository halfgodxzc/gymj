#include "bean_scoring.h"

#include <stdexcept>

namespace GYMJ {
namespace {

ScoreCategory category_for(BeanKind kind) {
    switch (kind) {
        case BeanKind::AnKan:
            return ScoreCategory::AnKanBean;
        case BeanKind::Kakan:
            return ScoreCategory::KakanBean;
        case BeanKind::Daiminkan:
            return ScoreCategory::DaiminkanBean;
    }
    throw std::logic_error("unknown bean kind");
}

BeanKind kind_for(MeldType type) {
    switch (type) {
        case MeldType::AnKan:
            return BeanKind::AnKan;
        case MeldType::Kakan:
            return BeanKind::Kakan;
        case MeldType::Daiminkan:
            return BeanKind::Daiminkan;
        case MeldType::Pon:
            break;
    }
    throw std::logic_error("pon is not a bean");
}

void settle_self_drawn_bean(BeanResult& result, const BeanItem& item) {
    const ScoreCategory category = category_for(item.kind);
    for (uint8_t other = 0; other < PLAYER_COUNT; ++other) {
        if (other == item.owner) continue;
        if (item.owner_is_jiao) {
            result.ledger.transfer(
                category,
                other,
                item.owner,
                item.points_per_payer,
                item.tile
            );
        } else {
            result.ledger.transfer(
                category,
                item.owner,
                other,
                item.points_per_payer,
                item.tile
            );
        }
    }
}

void settle_given_bean(BeanResult& result, const BeanItem& item) {
    if (item.source >= PLAYER_COUNT || item.source == item.owner) {
        throw std::invalid_argument("daiminkan source is invalid");
    }
    if (item.owner_is_jiao) {
        result.ledger.transfer(
            ScoreCategory::DaiminkanBean,
            item.source,
            item.owner,
            item.points_per_payer,
            item.tile
        );
    } else {
        result.ledger.transfer(
            ScoreCategory::DaiminkanBean,
            item.owner,
            item.source,
            item.points_per_payer,
            item.tile
        );
    }
}

} // namespace

BeanResult BeanScoring::calculate(
    const std::array<PlayerState, PLAYER_COUNT>& players,
    const std::array<bool, PLAYER_COUNT>& is_jiao
) {
    BeanResult result;
    for (uint8_t owner = 0; owner < PLAYER_COUNT; ++owner) {
        if (players[owner].player_id != owner) {
            throw std::invalid_argument(
                "player state order does not match player id"
            );
        }

        for (const auto& meld : players[owner].melds[0]) {
            if (meld.type == MeldType::Pon) continue;

            const BeanKind kind = kind_for(meld.type);
            const BeanItem item{
                kind,
                owner,
                meld.source,
                meld.tile,
                is_jiao[owner],
                BEAN_POINTS,
            };
            result.items.push_back(item);

            if (kind == BeanKind::Daiminkan) {
                settle_given_bean(result, item);
            } else {
                settle_self_drawn_bean(result, item);
            }
        }
    }
    return result;
}

} // namespace GYMJ
