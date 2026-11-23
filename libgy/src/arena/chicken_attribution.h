#pragma once

#include "round_settlement.h"

namespace GYMJ {

// Does not modify ledger amounts. Rebuilding replaces, rather than appends, data.
void attribute_chicken_settlement(
    RoundSettlementResult& settlement,
    const std::vector<GameEvent>& history,
    const std::vector<DiscardEvent>& discards
);

} // namespace GYMJ
