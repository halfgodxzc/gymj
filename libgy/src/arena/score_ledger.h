#pragma once

#include "settlement_snapshot.h"

#include <array>
#include <cstdint>
#include <vector>

namespace GYMJ {

enum class ScoreCategory : uint8_t {
    Hu,
    ChaJiao,
    Chicken,
    ResponsibilityChicken,
    AnKanBean,
    KakanBean,
    DaiminkanBean
};

struct ScoreTransfer {
    ScoreCategory category = ScoreCategory::Hu;
    uint8_t from = INVALID_PLAYER_ID;
    uint8_t to = INVALID_PLAYER_ID;
    int points = 0;
    Tile tile{255};
    uint8_t chicken_owner = INVALID_PLAYER_ID;
};

class ScoreLedger {
public:
    void transfer(
        ScoreCategory category,
        uint8_t from,
        uint8_t to,
        int points,
        Tile tile = Tile{255},
        uint8_t chicken_owner = INVALID_PLAYER_ID
    );
    void append(const ScoreLedger& other);

    const std::vector<ScoreTransfer>& transfers() const noexcept;
    const std::array<int, PLAYER_COUNT>& score_deltas() const noexcept;
    int net_total() const noexcept;

private:
    std::vector<ScoreTransfer> transfers_;
    std::array<int, PLAYER_COUNT> score_deltas_{};
};

} // namespace GYMJ
