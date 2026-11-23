#include "round_settlement.h"

#include "agari.h"
#include "shanten.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace GYMJ {
namespace {

int concealed_tile_count(const PlayerState& player) {
    int count = 0;
    for (uint8_t tile_count : player.tehai) {
        count += tile_count;
    }
    return count;
}

void validate_hu_settlements(
    const RoundResult& round_result,
    const std::vector<HuSettlement>& hu_settlements,
    const std::vector<ChaJiaoSettlement>& cha_jiao_settlements
) {
    if (round_result.reason == RoundEndReason::ExhaustiveDraw) {
        if (!round_result.hu_players.empty() || !hu_settlements.empty()) {
            throw std::invalid_argument(
                "exhaustive draw cannot have a hu settlement"
            );
        }
        std::array<bool, PLAYER_COUNT> seen{};
        for (const auto& cha_jiao : cha_jiao_settlements) {
            if (cha_jiao.player_id >= PLAYER_COUNT ||
                cha_jiao.best_wait.id >= TILE_TYPE_COUNT ||
                cha_jiao.score.hu_points <= 0 ||
                seen[cha_jiao.player_id]) {
                throw std::invalid_argument(
                    "cha jiao settlement is invalid"
                );
            }
            seen[cha_jiao.player_id] = true;
        }
        return;
    }
    if (!cha_jiao_settlements.empty()) {
        throw std::invalid_argument(
            "completed hand cannot have a cha jiao settlement"
        );
    }
    if (round_result.reason == RoundEndReason::None) {
        throw std::invalid_argument("unfinished round cannot be settled");
    }
    if (round_result.hu_players.size() != hu_settlements.size() ||
        hu_settlements.empty()) {
        throw std::invalid_argument("hu settlement count does not match");
    }

    for (std::size_t index = 0; index < hu_settlements.size(); ++index) {
        const auto& hu = hu_settlements[index];
        if (hu.player_id >= PLAYER_COUNT ||
            hu.player_id != round_result.hu_players[index] ||
            hu.score.hu_points <= 0) {
            throw std::invalid_argument("hu settlement is invalid");
        }
    }
}

void settle_cha_jiao(
    ScoreLedger& ledger,
    const std::vector<ChaJiaoSettlement>& cha_jiao_settlements
) {
    if (cha_jiao_settlements.empty() ||
        cha_jiao_settlements.size() == PLAYER_COUNT) {
        return;
    }

    std::array<bool, PLAYER_COUNT> is_jiao{};
    for (const auto& cha_jiao : cha_jiao_settlements) {
        is_jiao[cha_jiao.player_id] = true;
    }

    for (const auto& cha_jiao : cha_jiao_settlements) {
        for (uint8_t payer = 0; payer < PLAYER_COUNT; ++payer) {
            if (is_jiao[payer]) continue;
            ledger.transfer(
                ScoreCategory::ChaJiao,
                payer,
                cha_jiao.player_id,
                cha_jiao.score.hu_points,
                cha_jiao.best_wait
            );
        }
    }
}

void settle_hu(
    ScoreLedger& ledger,
    const RoundResult& round_result,
    const std::vector<HuSettlement>& hu_settlements
) {
    if (round_result.reason == RoundEndReason::ExhaustiveDraw) return;

    if (round_result.reason == RoundEndReason::Tsumo) {
        if (hu_settlements.size() != 1) {
            throw std::invalid_argument(
                "self-draw must have exactly one hu player"
            );
        }
        const auto& hu = hu_settlements.front();
        for (uint8_t payer = 0; payer < PLAYER_COUNT; ++payer) {
            if (payer == hu.player_id) continue;
            ledger.transfer(
                ScoreCategory::Hu,
                payer,
                hu.player_id,
                hu.score.hu_points,
                round_result.hu_tile
            );
        }
        return;
    }

    if (round_result.fangpao_player >= PLAYER_COUNT) {
        throw std::invalid_argument(
            "ron settlement has no valid fangpao player"
        );
    }
    for (const auto& hu : hu_settlements) {
        if (hu.player_id == round_result.fangpao_player) {
            throw std::invalid_argument(
                "fangpao player cannot also be a hu player"
            );
        }
        ledger.transfer(
            ScoreCategory::Hu,
            round_result.fangpao_player,
            hu.player_id,
            hu.score.hu_points,
            round_result.hu_tile
        );
    }
}

void settle_chickens(
    ScoreLedger& ledger,
    const ChickenResult& chicken
) {
    for (uint8_t owner = 0; owner < PLAYER_COUNT; ++owner) {
        const auto& owner_result = chicken.player(owner);
        if (owner_result.total_points > 0) {
            for (uint8_t payer = 0; payer < PLAYER_COUNT; ++payer) {
                if (payer == owner) continue;
                ledger.transfer(
                    ScoreCategory::Chicken,
                    payer,
                    owner,
                    owner_result.total_points,
                    Tile{255},
                    owner
                );
            }
        } else if (owner_result.total_points < 0) {
            const int points = -owner_result.total_points;
            for (uint8_t recipient = 0;
                 recipient < PLAYER_COUNT;
                 ++recipient) {
                if (recipient == owner ||
                    !chicken.player(recipient).is_jiao) {
                    continue;
                }
                ledger.transfer(
                    ScoreCategory::Chicken,
                    owner,
                    recipient,
                    points,
                    Tile{255},
                    owner
                );
            }
        }
    }

    for (const auto& responsibility : chicken.responsibility_chickens) {
        const auto& source = chicken.player(responsibility.source_player);
        const auto& owner = chicken.player(responsibility.owner_player);
        if (owner.is_jiao) {
            ledger.transfer(
                ScoreCategory::ResponsibilityChicken,
                responsibility.source_player,
                responsibility.owner_player,
                responsibility.points,
                responsibility.tile
            );
        } else if (source.is_jiao) {
            ledger.transfer(
                ScoreCategory::ResponsibilityChicken,
                responsibility.owner_player,
                responsibility.source_player,
                responsibility.points,
                responsibility.tile
            );
        }
    }
}

} // namespace

std::vector<ChaJiaoSettlement>
RoundSettlementCalculator::evaluate_cha_jiao(
    const std::array<PlayerState, PLAYER_COUNT>& players,
    const std::array<ReadyState, PLAYER_COUNT>& ready_states
) {
    std::vector<ChaJiaoSettlement> settlements;
    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        const PlayerState& player = players[player_id];
        const int meld_count = static_cast<int>(player.melds[0].size());
        const int expected_concealed_count =
            COMPLETE_HAND_TILE_COUNT - 1 - TILES_PER_MELD * meld_count;
        if (concealed_tile_count(player) != expected_concealed_count) {
            throw std::logic_error(
                "exhaustive draw player has invalid concealed tile count"
            );
        }
        if (Algo::calculate_shanten(player.tehai, meld_count).best != 0) {
            continue;
        }

        const auto waits = Algo::get_waits(player.tehai, meld_count);
        ChaJiaoSettlement best;
        bool found = false;
        for (Tile wait : waits) {
            HuContext context;
            context.method = HuMethod::Ron;
            context.hu_tile = wait;
            context.hu_player_ready_kind = ready_states[player_id].kind;
            context.bypass_ron_passport = true;

            const HuEvaluation evaluation = HuEvaluator::evaluate(
                player,
                context
            );
            const HuCandidate* candidate = evaluation.best_candidate();
            if (!evaluation.legal || candidate == nullptr) continue;
            if (found &&
                candidate->score.hu_points <= best.score.hu_points) {
                continue;
            }

            best.player_id = player_id;
            best.best_wait = wait;
            best.features = candidate->features;
            best.score = candidate->score;
            found = true;
        }
        if (!found) {
            throw std::logic_error(
                "tenpai hand has no valid cha jiao wait"
            );
        }
        settlements.push_back(std::move(best));
    }
    return settlements;
}

RoundSettlementResult RoundSettlementCalculator::calculate(
    const RoundResult& round_result,
    const std::vector<HuSettlement>& hu_settlements,
    const std::vector<ChaJiaoSettlement>& cha_jiao_settlements,
    const std::optional<ChickenResult>& chicken,
    const BeanResult& beans
) {
    validate_hu_settlements(
        round_result,
        hu_settlements,
        cha_jiao_settlements
    );
    if (round_result.reason == RoundEndReason::ExhaustiveDraw &&
        chicken.has_value()) {
        throw std::invalid_argument(
            "exhaustive draw does not reveal a chicken indicator"
        );
    }
    if (round_result.reason == RoundEndReason::ExhaustiveDraw &&
        (!beans.items.empty() || !beans.ledger.transfers().empty())) {
        throw std::invalid_argument(
            "exhaustive draw does not settle beans"
        );
    }

    RoundSettlementResult result;
    result.hu_settlements = hu_settlements;
    result.cha_jiao_settlements = cha_jiao_settlements;
    result.chicken = chicken;
    result.beans = beans;

    settle_hu(result.ledger, round_result, hu_settlements);
    if (round_result.reason == RoundEndReason::ExhaustiveDraw) {
        settle_cha_jiao(result.ledger, cha_jiao_settlements);
    }
    if (chicken.has_value()) {
        settle_chickens(result.ledger, *chicken);
    }
    result.ledger.append(beans.ledger);

    if (result.ledger.net_total() != 0) {
        throw std::logic_error("round settlement is not zero-sum");
    }
    return result;
}

} // namespace GYMJ
