#pragma once

#include "guiyang_mahjong.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace GYMJ {

inline constexpr std::size_t PLAYER_COUNT = 4;
inline constexpr uint8_t INVALID_PLAYER_ID = 255;

enum class HuTileOrigin : uint8_t {
    SelfDraw,
    Discard,
    Kakan
};

struct HuTileClaim {
    HuTileOrigin origin = HuTileOrigin::SelfDraw;
    Tile tile{255};
    uint8_t source_player = INVALID_PLAYER_ID;
    std::vector<uint8_t> hu_players;
};

struct PlayerSettlementState {
    uint8_t player_id = INVALID_PLAYER_ID;
    std::array<uint8_t, TILE_TYPE_COUNT> hand_counts{};
    std::vector<Tile> discards;
    std::vector<Meld> melds;

    uint8_t hand_count(Tile tile) const;
    std::size_t discard_count(Tile tile) const;
    uint8_t meld_tile_count(Tile tile) const;
    int settlement_tile_count(Tile tile) const;
};

struct SettlementSnapshot {
    std::array<PlayerSettlementState, PLAYER_COUNT> players;
    HuTileClaim hu_tile_claim;

    const PlayerSettlementState& player(uint8_t player_id) const;
};

class SettlementSnapshotBuilder {
public:
    static SettlementSnapshot build(
        const std::array<PlayerState, PLAYER_COUNT>& players,
        const HuTileClaim& hu_tile_claim
    );
};

} // namespace GYMJ
