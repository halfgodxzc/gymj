#include "score_ledger.h"

#include <numeric>
#include <stdexcept>

namespace GYMJ {

void ScoreLedger::transfer(
    ScoreCategory category,
    uint8_t from,
    uint8_t to,
    int points,
    Tile tile,
    uint8_t chicken_owner
) {
    if (from >= PLAYER_COUNT || to >= PLAYER_COUNT) {
        throw std::invalid_argument("score transfer player id is out of range");
    }
    if (from == to) {
        throw std::invalid_argument("score transfer cannot pay the same player");
    }
    if (points <= 0) {
        throw std::invalid_argument("score transfer points must be positive");
    }
    if (tile.id != 255 && tile.id >= TILE_TYPE_COUNT) {
        throw std::invalid_argument("score transfer tile id is out of range");
    }

    if (chicken_owner != INVALID_PLAYER_ID &&
        (chicken_owner >= PLAYER_COUNT || category != ScoreCategory::Chicken ||
         (chicken_owner != from && chicken_owner != to))) {
        throw std::invalid_argument("invalid chicken transfer owner");
    }
    transfers_.push_back({category, from, to, points, tile, chicken_owner});
    score_deltas_[from] -= points;
    score_deltas_[to] += points;
}

void ScoreLedger::append(const ScoreLedger& other) {
    for (const auto& transfer_item : other.transfers_) {
        transfer(
            transfer_item.category,
            transfer_item.from,
            transfer_item.to,
            transfer_item.points,
            transfer_item.tile,
            transfer_item.chicken_owner
        );
    }
}

const std::vector<ScoreTransfer>& ScoreLedger::transfers() const noexcept {
    return transfers_;
}

const std::array<int, PLAYER_COUNT>& ScoreLedger::score_deltas() const noexcept {
    return score_deltas_;
}

int ScoreLedger::net_total() const noexcept {
    return std::accumulate(
        score_deltas_.begin(),
        score_deltas_.end(),
        0
    );
}

} // namespace GYMJ
