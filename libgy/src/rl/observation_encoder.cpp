#include "observation_encoder.h"

#include <algorithm>
#include <stdexcept>

namespace GYMJ {
namespace RL {
namespace {

uint8_t relative_player(uint8_t absolute_player, uint8_t observer_id) {
    if (absolute_player >= 4) {
        throw std::logic_error("absolute player id must be in range 0..3");
    }
    return static_cast<uint8_t>(
        (absolute_player - observer_id + 4) % 4
    );
}

void add_visible_tiles(
    std::array<uint8_t, 27>& visible_counts,
    uint8_t tile_id,
    uint8_t count
) {
    if (tile_id >= 27 || visible_counts[tile_id] + count > 4) {
        throw std::logic_error("visible tile count exceeds physical copies");
    }
    visible_counts[tile_id] += count;
}

} // namespace

CompactObservation ObservationEncoder::encode(
    const Arena& arena,
    uint8_t observer_id
) {
    if (observer_id >= 4) {
        throw std::out_of_range("observer id must be in range 0..3");
    }

    CompactObservation observation;
    const PlayerState& player = arena.player(observer_id);

    std::copy(
        player.tehai,
        player.tehai + 27,
        observation.hand_counts.begin()
    );
    observation.visible_counts = observation.hand_counts;
    observation.shanten = player.shanten;
    for (uint8_t tile_id = 0; tile_id < 27; ++tile_id) {
        if (player.waits[tile_id]) {
            observation.waits |= uint32_t{1} << tile_id;
        }
    }

    const auto discard_history = arena.discard_history_for(observer_id);
    if (discard_history.size() > MAX_DISCARD_EVENTS) {
        throw std::logic_error("discard history exceeds Observation V1 capacity");
    }
    observation.discard_count =
        static_cast<uint8_t>(discard_history.size());
    for (size_t i = 0; i < discard_history.size(); ++i) {
        const auto& event = discard_history[i];
        observation.discards[i] = {
            event.tile.id,
            event.relative_actor,
            event.was_called ? uint8_t{1} : uint8_t{0},
        };
        if (!event.was_called) {
            add_visible_tiles(
                observation.visible_counts,
                event.tile.id,
                1
            );
        }
    }

    size_t meld_index = 0;
    for (uint8_t relative_owner = 0; relative_owner < 4; ++relative_owner) {
        for (const auto& meld : player.melds[relative_owner]) {
            if (meld_index >= MAX_MELDS) {
                throw std::logic_error("meld count exceeds Observation V1 capacity");
            }
            observation.melds[meld_index++] = {
                meld.tile.id,
                relative_owner,
                relative_player(meld.source, observer_id),
                meld.type,
                1,
            };
            add_visible_tiles(
                observation.visible_counts,
                meld.tile.id,
                meld.type == MeldType::Pon ? 3 : 4
            );
        }
    }
    observation.meld_count = static_cast<uint8_t>(meld_index);

    observation.phase = arena.phase();
    observation.dealer = relative_player(arena.dealer(), observer_id);
    observation.wall_remaining =
        static_cast<uint8_t>(arena.wall_remaining());
    for (uint8_t absolute_player = 0; absolute_player < 4; ++absolute_player) {
        const uint8_t relative_owner =
            relative_player(absolute_player, observer_id);
        observation.ready_kinds[relative_owner] = static_cast<uint8_t>(
            arena.ready_state(absolute_player).kind
        );
    }
    observation.locked_ready_waits =
        arena.ready_state(observer_id).locked_waits;

    if (arena.phase() == RoundPhase::AwaitTurnAction ||
        arena.phase() == RoundPhase::AwaitReadyDeclaration ||
        arena.phase() == RoundPhase::AwaitDiscardResponses ||
        arena.phase() == RoundPhase::AwaitKakanResponses) {
        const DecisionContext decision = arena.current_decision();
        observation.current_actor =
            relative_player(decision.player, observer_id);

        if (decision.player == observer_id) {
            const EncodedDecision encoded =
                ActionEncoder::encode_legal_actions(decision.legal_actions);
            observation.legal_mask = encoded.mask_words();
        }

        if (arena.phase() == RoundPhase::AwaitDiscardResponses ||
            arena.phase() == RoundPhase::AwaitKakanResponses) {
            const Tile response_tile = arena.response_tile();
            observation.current_discard = response_tile.id;
            observation.current_discard_actor = relative_player(
                arena.response_actor(),
                observer_id
            );
            if (arena.phase() == RoundPhase::AwaitKakanResponses &&
                arena.response_actor() != observer_id) {
                add_visible_tiles(
                    observation.visible_counts,
                    response_tile.id,
                    1
                );
            }
        }
    }

    return observation;
}

} // namespace RL
} // namespace GYMJ
