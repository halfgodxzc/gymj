#include "settlement_snapshot.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GYMJ {
namespace {

void validate_tile(Tile tile) {
    if (tile.id >= TILE_TYPE_COUNT) {
        throw std::invalid_argument("hu tile id is out of range");
    }
}

void validate_player_id(uint8_t player_id, const char* field_name) {
    if (player_id >= PLAYER_COUNT) {
        throw std::invalid_argument(
            std::string(field_name) + " is out of range"
        );
    }
}

void validate_claim(const HuTileClaim& claim) {
    validate_tile(claim.tile);
    validate_player_id(claim.source_player, "hu tile source player");
    if (claim.hu_players.empty()) {
        throw std::invalid_argument("hu tile claim has no hu player");
    }

    std::array<bool, PLAYER_COUNT> seen{};
    for (uint8_t hu_player : claim.hu_players) {
        validate_player_id(hu_player, "hu player");
        if (seen[hu_player]) {
            throw std::invalid_argument("hu tile claim has duplicate hu player");
        }
        seen[hu_player] = true;
    }

    if (claim.origin == HuTileOrigin::SelfDraw) {
        if (claim.hu_players.size() != 1 ||
            claim.hu_players.front() != claim.source_player) {
            throw std::invalid_argument(
                "self-draw claim must have the source as its only hu player"
            );
        }
        return;
    }

    if (seen[claim.source_player]) {
        throw std::invalid_argument(
            "discard or kakan source cannot also be a hu player"
        );
    }
}

PlayerSettlementState make_player_state(const PlayerState& player) {
    PlayerSettlementState state;
    state.player_id = player.player_id;
    std::copy(
        player.tehai,
        player.tehai + TILE_TYPE_COUNT,
        state.hand_counts.begin()
    );
    state.discards = player.kawa[0];
    state.melds = player.melds[0];
    return state;
}

bool has_matching_pon(
    const PlayerSettlementState& state,
    Tile tile
) {
    return std::any_of(
        state.melds.begin(),
        state.melds.end(),
        [&](const Meld& meld) {
            return meld.type == MeldType::Pon && meld.tile == tile;
        }
    );
}

void add_hu_tile_to_hu_players(SettlementSnapshot& snapshot) {
    const Tile tile = snapshot.hu_tile_claim.tile;
    for (uint8_t hu_player : snapshot.hu_tile_claim.hu_players) {
        auto& state = snapshot.players[hu_player];
        if (state.hand_count(tile) >= MAX_TILE_COPIES) {
            throw std::invalid_argument(
                "hu player already has four copies of hu tile in hand"
            );
        }
        state.hand_counts[tile.id]++;
    }
}

void apply_discard_claim(SettlementSnapshot& snapshot) {
    const auto& claim = snapshot.hu_tile_claim;
    auto& source = snapshot.players[claim.source_player];
    if (source.discards.empty() ||
        source.discards.back() != claim.tile) {
        throw std::invalid_argument(
            "hu discard is not the source player's latest discard"
        );
    }
    source.discards.pop_back();
    add_hu_tile_to_hu_players(snapshot);
}

void apply_kakan_claim(SettlementSnapshot& snapshot) {
    const auto& claim = snapshot.hu_tile_claim;
    auto& source = snapshot.players[claim.source_player];
    if (!has_matching_pon(source, claim.tile)) {
        throw std::invalid_argument(
            "kakan source has no matching pon"
        );
    }
    if (source.hand_count(claim.tile) == 0) {
        throw std::invalid_argument(
            "kakan source has no fourth concealed tile"
        );
    }
    source.hand_counts[claim.tile.id]--;
    add_hu_tile_to_hu_players(snapshot);
}

} // namespace

uint8_t PlayerSettlementState::hand_count(Tile tile) const {
    validate_tile(tile);
    return hand_counts[tile.id];
}

std::size_t PlayerSettlementState::discard_count(Tile tile) const {
    validate_tile(tile);
    return static_cast<std::size_t>(std::count(
        discards.begin(),
        discards.end(),
        tile
    ));
}

uint8_t PlayerSettlementState::meld_tile_count(Tile tile) const {
    validate_tile(tile);
    int count = 0;
    for (const auto& meld : melds) {
        if (meld.tile != tile) continue;
        count += meld.type == MeldType::Pon ? 3 : MAX_TILE_COPIES;
    }
    return static_cast<uint8_t>(count);
}

int PlayerSettlementState::settlement_tile_count(Tile tile) const {
    return hand_count(tile) +
           static_cast<int>(discard_count(tile)) +
           meld_tile_count(tile);
}

const PlayerSettlementState& SettlementSnapshot::player(
    uint8_t player_id
) const {
    if (player_id >= PLAYER_COUNT) {
        throw std::out_of_range("settlement player id is out of range");
    }
    return players[player_id];
}

SettlementSnapshot SettlementSnapshotBuilder::build(
    const std::array<PlayerState, PLAYER_COUNT>& players,
    const HuTileClaim& hu_tile_claim
) {
    validate_claim(hu_tile_claim);

    SettlementSnapshot snapshot;
    snapshot.hu_tile_claim = hu_tile_claim;
    for (std::size_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        if (players[player_id].player_id != player_id) {
            throw std::invalid_argument(
                "player state order does not match player id"
            );
        }
        snapshot.players[player_id] = make_player_state(players[player_id]);
    }

    switch (hu_tile_claim.origin) {
        case HuTileOrigin::SelfDraw: {
            const auto& hu_player =
                snapshot.players[hu_tile_claim.source_player];
            if (hu_player.hand_count(hu_tile_claim.tile) == 0) {
                throw std::invalid_argument(
                    "self-drawn hu tile is not in the hu player's hand"
                );
            }
            break;
        }
        case HuTileOrigin::Discard:
            apply_discard_claim(snapshot);
            break;
        case HuTileOrigin::Kakan:
            apply_kakan_claim(snapshot);
            break;
    }
    return snapshot;
}

} // namespace GYMJ
