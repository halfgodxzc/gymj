#include "chicken.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GYMJ {
namespace {

void validate_tile(Tile tile, const char* field_name) {
    if (tile.id >= TILE_TYPE_COUNT) {
        throw std::invalid_argument(
            std::string(field_name) + " tile id is out of range"
        );
    }
}

ChickenTileCounts count_tile(
    const PlayerSettlementState& player,
    Tile tile,
    bool include_hand
) {
    return {
        include_hand ? player.hand_count(tile) : 0,
        static_cast<int>(player.discard_count(tile)),
        player.meld_tile_count(tile),
    };
}

ChickenKind fixed_kind(Tile tile) {
    return tile.id == YAO_JI_TILE_ID
        ? ChickenKind::YaoJi
        : ChickenKind::BaTong;
}

ChickenKind chongfeng_kind(Tile tile) {
    return tile.id == YAO_JI_TILE_ID
        ? ChickenKind::ChongfengYaoJi
        : ChickenKind::ChongfengBaTong;
}

size_t fixed_chicken_index(Tile tile) {
    if (tile.id == YAO_JI_TILE_ID) return 0;
    if (tile.id == BA_TONG_TILE_ID) return 1;
    throw std::invalid_argument("tile is not a fixed chicken");
}

uint8_t chongfeng_mask(Tile tile) {
    return tile.id == YAO_JI_TILE_ID
        ? CHONGFENG_YAO_JI_MASK
        : CHONGFENG_BA_TONG_MASK;
}

void add_item(
    PlayerChickenResult& result,
    ChickenKind kind,
    Tile tile,
    const ChickenTileCounts& counts,
    bool is_gold,
    int base_points,
    int score_sign
) {
    if (counts.total() == 0) return;

    ChickenItem item;
    item.kind = kind;
    item.tile = tile;
    item.counts = counts;
    item.is_gold = is_gold;
    item.points_per_tile = score_sign * base_points *
                           (is_gold ? GOLD_CHICKEN_MULTIPLIER : 1);

    result.total_tile_count += item.tile_count();
    result.raw_total_points += item.total_points();
    result.total_points += item.total_points();
    result.items.push_back(item);
}

void add_fixed_chicken(
    PlayerChickenResult& result,
    const PlayerSettlementState& player,
    Tile tile,
    const std::optional<Tile>& flipped_chicken,
    uint8_t chongfeng_tile_mask,
    bool is_jiao
) {
    ChickenTileCounts ordinary_counts = count_tile(
        player,
        tile,
        is_jiao
    );
    const int score_sign = is_jiao ? 1 : -1;
    const bool is_gold = flipped_chicken.has_value() &&
                         *flipped_chicken == tile;
    const bool is_chongfeng =
        (chongfeng_tile_mask & chongfeng_mask(tile)) != 0;

    if (is_chongfeng) {
        if (ordinary_counts.discards == 0) {
            throw std::invalid_argument(
                "chongfeng chicken is absent from the player's discards"
            );
        }
        ordinary_counts.discards--;
    }

    add_item(
        result,
        fixed_kind(tile),
        tile,
        ordinary_counts,
        is_gold,
        FIXED_CHICKEN_POINTS,
        score_sign
    );

    if (is_chongfeng) {
        add_item(
            result,
            chongfeng_kind(tile),
            tile,
            {0, 1, 0},
            is_gold,
            CHONGFENG_CHICKEN_POINTS,
            score_sign
        );
    }
}

} // namespace

bool ChongfengTracker::observe_discard(uint8_t player_id, Tile tile) {
    if (player_id >= PLAYER_COUNT) {
        throw std::out_of_range("chongfeng player id is out of range");
    }
    validate_tile(tile, "chongfeng");

    const bool is_opening_discard = !player_has_discarded_[player_id];
    player_has_discarded_[player_id] = true;
    if (!ChickenResolver::is_fixed_chicken(tile)) return false;

    const size_t chicken_index = fixed_chicken_index(tile);
    const bool is_global_first = !fixed_chicken_seen_[chicken_index];
    fixed_chicken_seen_[chicken_index] = true;
    return is_global_first || is_opening_discard;
}

int ChickenTileCounts::total() const {
    return hand + discards + melds;
}

int ChickenItem::tile_count() const {
    return counts.total();
}

int ChickenItem::total_points() const {
    return tile_count() * points_per_tile;
}

bool PlayerChickenResult::has_any_chicken() const {
    return total_tile_count > 0;
}

const PlayerChickenResult& ChickenResult::player(uint8_t player_id) const {
    if (player_id >= PLAYER_COUNT) {
        throw std::out_of_range("chicken player id is out of range");
    }
    return players[player_id];
}

Tile ChickenResolver::flipped_chicken_for(Tile indicator) {
    validate_tile(indicator, "indicator");
    const uint8_t suit_start = static_cast<uint8_t>(
        indicator.suit() * TILES_PER_SUIT
    );
    const uint8_t next_offset = static_cast<uint8_t>(
        (indicator.id % TILES_PER_SUIT + 1) % TILES_PER_SUIT
    );
    return Tile{static_cast<uint8_t>(suit_start + next_offset)};
}

bool ChickenResolver::is_fixed_chicken(Tile tile) {
    validate_tile(tile, "fixed chicken");
    return tile.id == YAO_JI_TILE_ID || tile.id == BA_TONG_TILE_ID;
}

ChickenResult ChickenResolver::resolve(
    const SettlementSnapshot& snapshot,
    const ChickenContext& context
) {
    ChickenResult result;
    result.indicator = context.indicator;
    if (context.indicator.has_value()) {
        result.flipped_chicken = flipped_chicken_for(*context.indicator);
    }

    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        const auto& player = snapshot.player(player_id);
        auto& player_result = result.players[player_id];
        player_result.player_id = player_id;
        player_result.is_jiao = context.is_jiao[player_id];
        player_result.is_burned = context.burned_players[player_id];

        const uint8_t chongfeng_mask =
            context.chongfeng_tile_masks[player_id];
        if ((chongfeng_mask & static_cast<uint8_t>(
                ~CHONGFENG_ALL_FIXED_MASK
            )) != 0) {
            throw std::invalid_argument(
                "chongfeng marker contains an unknown fixed chicken"
            );
        }

        add_fixed_chicken(
            player_result,
            player,
            Tile{YAO_JI_TILE_ID},
            result.flipped_chicken,
            chongfeng_mask,
            player_result.is_jiao
        );
        add_fixed_chicken(
            player_result,
            player,
            Tile{BA_TONG_TILE_ID},
            result.flipped_chicken,
            chongfeng_mask,
            player_result.is_jiao
        );

        if (result.flipped_chicken.has_value() &&
            !is_fixed_chicken(*result.flipped_chicken)) {
            add_item(
                player_result,
                ChickenKind::Flipped,
                *result.flipped_chicken,
                count_tile(
                    player,
                    *result.flipped_chicken,
                    player_result.is_jiao
                ),
                false,
                NORMAL_CHICKEN_POINTS,
                player_result.is_jiao ? 1 : -1
            );
        }

        if (player_result.is_burned) {
            player_result.total_points = 0;
        }
    }

    for (const auto& responsibility : context.responsibility_chickens) {
        if (responsibility.source_player >= PLAYER_COUNT ||
            responsibility.owner_player >= PLAYER_COUNT ||
            responsibility.source_player == responsibility.owner_player) {
            throw std::invalid_argument(
                "responsibility chicken players are invalid"
            );
        }
        if (!is_fixed_chicken(responsibility.tile)) {
            throw std::invalid_argument(
                "responsibility chicken must be YaoJi or BaTong"
            );
        }

        const auto& owner = snapshot.player(responsibility.owner_player);
        const bool matching_called_meld = std::any_of(
            owner.melds.begin(),
            owner.melds.end(),
            [&](const Meld& meld) {
                return meld.tile == responsibility.tile &&
                       meld.source == responsibility.source_player &&
                       meld.type != MeldType::AnKan;
            }
        );
        if (!matching_called_meld) {
            throw std::invalid_argument(
                "responsibility chicken has no matching called meld"
            );
        }

        const bool is_gold = result.flipped_chicken.has_value() &&
                             *result.flipped_chicken == responsibility.tile;
        result.responsibility_chickens.push_back({
            responsibility.source_player,
            responsibility.owner_player,
            responsibility.tile,
            is_gold,
            CHONGFENG_CHICKEN_POINTS *
                (is_gold ? GOLD_CHICKEN_MULTIPLIER : 1),
        });
    }
    return result;
}

} // namespace GYMJ
