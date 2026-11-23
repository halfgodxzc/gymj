#pragma once

#include "arena_types.h"
#include "bean_scoring.h"
#include "chicken.h"
#include "hu_evaluator.h"

#include <array>
#include <optional>
#include <vector>

namespace GYMJ {

struct HuSettlement {
    uint8_t player_id = INVALID_PLAYER_ID;
    HuFeatures features;
    HuScore score;
};

struct ChaJiaoSettlement {
    uint8_t player_id = INVALID_PLAYER_ID;
    Tile best_wait{255};
    HuFeatures features;
    HuScore score;
};

enum class ChickenAttributionKind : uint8_t {
    Chongfeng,
    Pon,
    Responsibility
};

// One signed portion of an actual ledger transfer, attached to an applied event.
struct ChickenAttribution {
    uint8_t player_id = INVALID_PLAYER_ID;
    uint32_t event_sequence = 0;
    std::size_t transfer_index = 0;
    ChickenAttributionKind kind = ChickenAttributionKind::Chongfeng;
    Tile tile{255};
    int points = 0;
};

struct RoundSettlementResult {
    std::vector<HuSettlement> hu_settlements;
    std::vector<ChaJiaoSettlement> cha_jiao_settlements;
    std::optional<ChickenResult> chicken;
    BeanResult beans;
    ScoreLedger ledger;
    std::vector<ChickenAttribution> chicken_attributions;
    std::array<int, PLAYER_COUNT> chicken_unattributed_deltas{};
};

class RoundSettlementCalculator {
public:
    static std::vector<ChaJiaoSettlement> evaluate_cha_jiao(
        const std::array<PlayerState, PLAYER_COUNT>& players,
        const std::array<ReadyState, PLAYER_COUNT>& ready_states
    );

    static RoundSettlementResult calculate(
        const RoundResult& round_result,
        const std::vector<HuSettlement>& hu_settlements,
        const std::vector<ChaJiaoSettlement>& cha_jiao_settlements,
        const std::optional<ChickenResult>& chicken,
        const BeanResult& beans
    );
};

} // namespace GYMJ
