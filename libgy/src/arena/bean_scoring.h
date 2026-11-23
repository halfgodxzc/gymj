#pragma once

#include "score_ledger.h"

#include <array>
#include <cstdint>
#include <vector>

namespace GYMJ {

inline constexpr int BEAN_POINTS = 5;

enum class BeanKind : uint8_t {
    AnKan,
    Kakan,
    Daiminkan
};

struct BeanItem {
    BeanKind kind = BeanKind::AnKan;
    uint8_t owner = INVALID_PLAYER_ID;
    uint8_t source = INVALID_PLAYER_ID;
    Tile tile{255};
    bool owner_is_jiao = false;
    int points_per_payer = BEAN_POINTS;
};

struct BeanResult {
    std::vector<BeanItem> items;
    ScoreLedger ledger;
};

class BeanScoring {
public:
    static BeanResult calculate(
        const std::array<PlayerState, PLAYER_COUNT>& players,
        const std::array<bool, PLAYER_COUNT>& is_jiao
    );
};

} // namespace GYMJ
