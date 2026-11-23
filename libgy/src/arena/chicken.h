#pragma once

#include "settlement_snapshot.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace GYMJ {

inline constexpr int NORMAL_CHICKEN_POINTS = 1;
inline constexpr int FIXED_CHICKEN_POINTS = 2;
inline constexpr int CHONGFENG_CHICKEN_POINTS = 5;
inline constexpr int GOLD_CHICKEN_MULTIPLIER = 2;
inline constexpr std::size_t FIXED_CHICKEN_TYPE_COUNT = 2;
inline constexpr uint8_t CHONGFENG_YAO_JI_MASK = uint8_t{1} << 0;
inline constexpr uint8_t CHONGFENG_BA_TONG_MASK = uint8_t{1} << 1;
inline constexpr uint8_t CHONGFENG_ALL_FIXED_MASK =
    CHONGFENG_YAO_JI_MASK | CHONGFENG_BA_TONG_MASK;

enum class ChickenKind : uint8_t {
    Flipped,
    YaoJi,
    BaTong,
    ChongfengYaoJi,
    ChongfengBaTong
};

struct ChickenTileCounts {
    int hand = 0;
    int discards = 0;
    int melds = 0;

    int total() const;
};

struct ChickenItem {
    ChickenKind kind = ChickenKind::Flipped;
    Tile tile{255};
    ChickenTileCounts counts;
    bool is_gold = false;
    int points_per_tile = 0;

    int tile_count() const;
    int total_points() const;
};

struct PlayerChickenResult {
    uint8_t player_id = INVALID_PLAYER_ID;
    bool is_jiao = false;
    bool is_burned = false;
    std::vector<ChickenItem> items;
    int total_tile_count = 0;
    int raw_total_points = 0;
    int total_points = 0;

    bool has_any_chicken() const;
};

struct ResponsibilityChicken {
    uint8_t source_player = INVALID_PLAYER_ID;
    uint8_t owner_player = INVALID_PLAYER_ID;
    Tile tile{255};
};

struct ResponsibilityChickenResult {
    uint8_t source_player = INVALID_PLAYER_ID;
    uint8_t owner_player = INVALID_PLAYER_ID;
    Tile tile{255};
    bool is_gold = false;
    int points = 0;
};

struct ChickenContext {
    std::optional<Tile> indicator;
    std::array<uint8_t, PLAYER_COUNT> chongfeng_tile_masks{};
    std::array<bool, PLAYER_COUNT> is_jiao{};
    std::array<bool, PLAYER_COUNT> burned_players{};
    std::vector<ResponsibilityChicken> responsibility_chickens;
};

struct ChickenResult {
    std::optional<Tile> indicator;
    std::optional<Tile> flipped_chicken;
    std::array<PlayerChickenResult, PLAYER_COUNT> players;
    std::vector<ResponsibilityChickenResult> responsibility_chickens;

    const PlayerChickenResult& player(uint8_t player_id) const;
};

// Tracks each fixed chicken's global first discard and opening follows.
class ChongfengTracker {
public:
    bool observe_discard(uint8_t player_id, Tile tile);

private:
    std::array<bool, FIXED_CHICKEN_TYPE_COUNT> fixed_chicken_seen_{};
    std::array<bool, PLAYER_COUNT> player_has_discarded_{};
};

class ChickenResolver {
public:
    static Tile flipped_chicken_for(Tile indicator);
    static bool is_fixed_chicken(Tile tile);
    static ChickenResult resolve(
        const SettlementSnapshot& snapshot,
        const ChickenContext& context
    );
};

} // namespace GYMJ
