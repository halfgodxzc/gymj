#include "privileged_pattern_encoder.h"

#include "chicken.h"

#include <algorithm>
#include <stdexcept>

namespace GYMJ {
namespace RL {
namespace {

uint8_t relative_player(uint8_t absolute, uint8_t observer) {
    if (absolute >= PLAYER_COUNT) return NO_PLAYER;
    return static_cast<uint8_t>(
        (absolute - observer + PLAYER_COUNT) % PLAYER_COUNT
    );
}

size_t bean_index(MeldType type) {
    switch (type) {
        case MeldType::AnKan:
            return 0;
        case MeldType::Kakan:
            return 1;
        case MeldType::Daiminkan:
            return 2;
        case MeldType::Pon:
            break;
    }
    throw std::invalid_argument("pon is not a bean type");
}

CompactObservation make_true_hand_observation(
    const PlayerState& player,
    uint8_t absolute_player,
    const std::array<uint8_t, TILE_TYPE_COUNT>& wall_counts
) {
    CompactObservation observation;
    std::copy(
        player.tehai,
        player.tehai + TILE_TYPE_COUNT,
        observation.hand_counts.begin()
    );
    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        observation.visible_counts[tile_id] = static_cast<uint8_t>(
            MAX_TILE_COPIES - wall_counts[tile_id]
        );
    }
    for (const Meld& meld : player.melds[0]) {
        if (observation.meld_count >= observation.melds.size()) {
            throw std::logic_error("own melds exceed observation capacity");
        }
        observation.melds[observation.meld_count++] = {
            meld.tile.id,
            0,
            relative_player(meld.source, absolute_player),
            meld.type,
            1,
        };
    }
    return observation;
}

PrivilegedEconomyFeatures encode_economy(
    const std::array<PlayerState, PLAYER_COUNT>& players,
    const std::vector<DiscardEvent>& discard_history,
    uint8_t actor_id
) {
    PrivilegedEconomyFeatures result;
    for (uint8_t relative = 0; relative < PLAYER_COUNT; ++relative) {
        const uint8_t absolute = static_cast<uint8_t>(
            (actor_id + relative) % PLAYER_COUNT
        );
        const PlayerState& player = players[absolute];
        auto& counts = result.settlement_tile_counts[relative];
        for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
            counts[tile_id] = player.tehai[tile_id];
        }
        for (Tile tile : player.kawa[0]) {
            ++counts[tile.id];
        }
        for (const Meld& meld : player.melds[0]) {
            counts[meld.tile.id] = static_cast<uint8_t>(
                counts[meld.tile.id] +
                (meld.type == MeldType::Pon ? 3 : 4)
            );
            if (meld.type == MeldType::Pon) continue;
            const uint8_t source = meld.type == MeldType::Daiminkan
                ? relative_player(meld.source, actor_id)
                : relative;
            ++result.bean_source_counts[relative][source][
                bean_index(meld.type)
            ];
        }
        result.player_tenpai[relative] = static_cast<uint8_t>(
            player.shanten <= 0
        );
        const int fixed_chicken_count =
            counts[YAO_JI_TILE_ID] + counts[BA_TONG_TILE_ID];
        result.signed_fixed_chicken_counts[relative] = static_cast<int8_t>(
            result.player_tenpai[relative]
                ? fixed_chicken_count
                : -fixed_chicken_count
        );
    }

    ChongfengTracker chongfeng_tracker;
    for (const DiscardEvent& event : discard_history) {
        if (!chongfeng_tracker.observe_discard(event.actor, event.tile)) {
            continue;
        }
        const uint8_t source = relative_player(event.actor, actor_id);
        if (!event.was_called) {
            result.chongfeng_tile_masks[source] |=
                event.tile.id == YAO_JI_TILE_ID
                    ? CHONGFENG_YAO_JI_MASK
                    : CHONGFENG_BA_TONG_MASK;
            if (result.chongfeng_tiles[source] == NO_TILE) {
                result.chongfeng_tiles[source] = event.tile.id;
            }
            continue;
        }
        if (event.called_by >= PLAYER_COUNT) {
            throw std::logic_error(
                "called fixed chicken has no absolute owner"
            );
        }
        const uint8_t owner = relative_player(event.called_by, actor_id);
        ++result.responsibility_chickens[source][owner];
    }
    return result;
}

} // namespace

PrivilegedPatternFeatures PrivilegedPatternEncoder::encode(
    const Arena& arena,
    uint8_t actor_id
) {
    if (actor_id >= PLAYER_COUNT) {
        throw std::out_of_range(
            "privileged pattern actor must be in range 0..3"
        );
    }
    if (arena.current_decision().player != actor_id) {
        throw std::invalid_argument("actor is not the current decision maker");
    }

    PrivilegedPatternFeatures result;
    for (Tile tile : arena.wall_.remaining_tiles()) {
        ++result.remaining_wall_counts[tile.id];
    }
    for (uint8_t relative = 0; relative < PLAYER_COUNT; ++relative) {
        const uint8_t absolute = static_cast<uint8_t>(
            (actor_id + relative) % PLAYER_COUNT
        );
        const CompactObservation observation = make_true_hand_observation(
            arena.players_[absolute],
            absolute,
            result.remaining_wall_counts
        );
        result.player_patterns[relative] =
            PatternFeatureEncoder::encode_hand(observation);
        result.true_shape_wait_masks[relative] =
            result.player_patterns[relative].shape_wait_mask;
    }
    result.economy = encode_economy(
        arena.players_,
        arena.discard_history_,
        actor_id
    );
    result.progress.decision_sequence = arena.decision_sequence_;
    for (uint8_t relative = 0; relative < PLAYER_COUNT; ++relative) {
        const uint8_t absolute = static_cast<uint8_t>(
            (actor_id + relative) % PLAYER_COUNT
        );
        const auto& progress = arena.progress_[absolute];
        result.progress.has_normal_drawn[relative] =
            progress.has_normal_drawn;
        result.progress.has_discarded[relative] = progress.has_discarded;
        result.progress.opening_hand_changed[relative] =
            progress.opening_hand_changed;
        result.progress.ready_window_closed[relative] =
            progress.ready_window_closed;
    }
    switch (arena.last_draw_kind_) {
        case Arena::DrawKind::Opening:
            result.progress.last_draw_kind = DecisionDrawKind::Opening;
            break;
        case Arena::DrawKind::Normal:
            result.progress.last_draw_kind = DecisionDrawKind::Normal;
            break;
        case Arena::DrawKind::Replacement:
            result.progress.last_draw_kind = DecisionDrawKind::Replacement;
            break;
    }
    result.progress.must_discard_after_call = arena.must_discard_;
    if (arena.hot_discard_player_.has_value()) {
        result.progress.hot_discard_player = relative_player(
            *arena.hot_discard_player_,
            actor_id
        );
    }
    result.progress.pending_discard_is_hot = arena.pending_discard_is_hot_;
    result.progress.pending_kakan = arena.pending_kakan_.has_value();
    return result;
}

} // namespace RL
} // namespace GYMJ
