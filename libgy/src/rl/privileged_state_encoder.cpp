#include "privileged_state_encoder.h"

#include <algorithm>
#include <stdexcept>

namespace GYMJ {
namespace RL {

PrivilegedState PrivilegedStateEncoder::encode(
    const Arena& arena,
    uint8_t actor_id
) {
    if (actor_id >= PLAYER_COUNT) {
        throw std::out_of_range("actor id must be in range 0..3");
    }
    if (arena.current_decision().player != actor_id) {
        throw std::invalid_argument("actor is not the current decision maker");
    }

    PrivilegedState state;
    for (size_t relative_player = 0; relative_player < PLAYER_COUNT;
         ++relative_player) {
        const uint8_t absolute_player = static_cast<uint8_t>(
            (actor_id + relative_player) % PLAYER_COUNT
        );
        state.player_tenpai[relative_player] = static_cast<uint8_t>(
            arena.player(absolute_player).shanten <= 0
        );
    }
    for (size_t relative_player = 1; relative_player < PLAYER_COUNT;
         ++relative_player) {
        const uint8_t absolute_player = static_cast<uint8_t>(
            (actor_id + relative_player) % PLAYER_COUNT
        );
        const auto& hand = arena.player(absolute_player).tehai;
        std::copy(
            hand,
            hand + TILE_TYPE_COUNT,
            state.opponent_hands[relative_player - 1].begin()
        );
    }

    const auto remaining_tiles = arena.wall_.remaining_tiles();
    if (remaining_tiles.size() > state.wall_tiles.size()) {
        throw std::logic_error("remaining wall exceeds privileged state capacity");
    }
    state.wall_tiles.fill(PRIVILEGED_TILE_PAD);
    state.wall_count = static_cast<uint8_t>(remaining_tiles.size());
    for (size_t i = 0; i < remaining_tiles.size(); ++i) {
        state.wall_tiles[i] = remaining_tiles[i].id;
    }

    uint8_t next_draw_player = 0;
    switch (arena.phase_) {
        case RoundPhase::AwaitTurnAction:
            next_draw_player = static_cast<uint8_t>(
                (arena.current_player_ + 1) % PLAYER_COUNT
            );
            break;
        case RoundPhase::AwaitReadyDeclaration:
            if (!arena.pending_ready_.has_value()) {
                throw std::logic_error(
                    "ready phase has no pending ready decision"
                );
            }
            next_draw_player =
                arena.pending_ready_->origin ==
                    Arena::ReadyDecisionOrigin::BeforeOpeningDraw
                ? arena.dealer()
                : static_cast<uint8_t>(
                    (arena.pending_discard_.actor + 1) % PLAYER_COUNT
                );
            break;
        case RoundPhase::AwaitDiscardResponses:
            next_draw_player = static_cast<uint8_t>(
                (arena.pending_discard_.actor + 1) % PLAYER_COUNT
            );
            break;
        case RoundPhase::AwaitKakanResponses:
            if (!arena.pending_kakan_.has_value()) {
                throw std::logic_error(
                    "kakan response phase has no pending kakan"
                );
            }
            next_draw_player = static_cast<uint8_t>(
                (arena.pending_kakan_->actor + 1) % PLAYER_COUNT
            );
            break;
        case RoundPhase::NotStarted:
        case RoundPhase::ReplacementDraw:
        case RoundPhase::Finished:
            throw std::logic_error(
                "privileged draw forecast requires an active decision"
            );
    }

    for (auto& player_draws : state.future_draw_tiles) {
        player_draws.fill(PRIVILEGED_TILE_PAD);
    }
    std::array<size_t, PLAYER_COUNT> future_counts{};
    const size_t forecast_count = std::min(
        remaining_tiles.size(),
        PLAYER_COUNT * PRIVILEGED_FUTURE_DRAW_COUNT
    );
    for (size_t offset = 0; offset < forecast_count; ++offset) {
        const uint8_t absolute_player = static_cast<uint8_t>(
            (next_draw_player + offset) % PLAYER_COUNT
        );
        const uint8_t relative_player = static_cast<uint8_t>(
            (absolute_player - actor_id + PLAYER_COUNT) % PLAYER_COUNT
        );
        auto& player_draws = state.future_draw_tiles[relative_player];
        player_draws[future_counts[relative_player]++] =
            remaining_tiles[offset].id;
    }

    state.replacement_draw_tiles.fill(PRIVILEGED_TILE_PAD);
    const size_t replacement_count = std::min(
        remaining_tiles.size(),
        PRIVILEGED_FUTURE_DRAW_COUNT
    );
    for (size_t offset = 0; offset < replacement_count; ++offset) {
        state.replacement_draw_tiles[offset] =
            remaining_tiles[remaining_tiles.size() - 1 - offset].id;
    }
    return state;
}

} // namespace RL
} // namespace GYMJ
