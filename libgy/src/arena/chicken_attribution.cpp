#include "chicken_attribution.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace GYMJ {

void attribute_chicken_settlement(
    RoundSettlementResult& settlement,
    const std::vector<GameEvent>& history,
    const std::vector<DiscardEvent>& discards
) {
    std::vector<const GameEvent*> discard_events;
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& event = history[i];
        if (event.sequence != i) {
            throw std::logic_error("non-contiguous chicken attribution history");
        }
        if (event.applied && event.action.type == ActionType::Discard) {
            discard_events.push_back(&event);
        }
    }
    if (discard_events.size() != discards.size()) {
        throw std::logic_error("chicken attribution discard history mismatch");
    }
    const auto& transfers = settlement.ledger.transfers();
    std::vector<ChickenAttribution> entries;
    std::array<int, PLAYER_COUNT> residual{};
    std::vector<std::array<int, PLAYER_COUNT>> allocated(transfers.size());
    for (const auto& transfer : transfers) {
        if (transfer.category == ScoreCategory::Chicken ||
            transfer.category == ScoreCategory::ResponsibilityChicken) {
            residual[transfer.from] -= transfer.points;
            residual[transfer.to] += transfer.points;
        }
    }
    auto add = [&](uint8_t player, const GameEvent& event, std::size_t index,
                   ChickenAttributionKind kind, Tile tile, int points) {
        if (!points) return;
        const auto& transfer = transfers[index];
        if (event.action.actor != player || !event.applied ||
            (points > 0 ? transfer.to : transfer.from) != player) {
            throw std::logic_error("invalid chicken attribution owner or sign");
        }
        allocated[index][player] += std::abs(points);
        if (allocated[index][player] > transfer.points) {
            throw std::logic_error("chicken attribution exceeds ledger transfer");
        }
        entries.push_back({player, event.sequence, index, kind, tile, points});
        residual[player] -= points;
    };
    auto ordinary = [&](const GameEvent& event, ChickenAttributionKind kind,
                        int signed_points) {
        const auto owner = event.action.actor;
        for (std::size_t i = 0; i < transfers.size(); ++i) {
            const auto& transfer = transfers[i];
            if (transfer.category != ScoreCategory::Chicken ||
                transfer.chicken_owner != owner) continue;
            if ((signed_points > 0 && transfer.to == owner) ||
                (signed_points < 0 && transfer.from == owner)) {
                add(owner, event, i, kind, event.action.tile, signed_points);
            }
        }
    };
    ChongfengTracker tracker;
    for (std::size_t d = 0; d < discards.size(); ++d) {
        const auto& discard = discards[d];
        const auto& event = *discard_events[d];
        if (event.action.actor != discard.actor || event.action.tile != discard.tile) {
            throw std::logic_error("chicken attribution discard identity mismatch");
        }
        const bool qualified = tracker.observe_discard(discard.actor, discard.tile);
        if (!settlement.chicken || !qualified) continue;
        const auto& chicken = *settlement.chicken;
        if (!discard.was_called) {
            const auto& owner = chicken.player(discard.actor);
            if (owner.is_burned) continue;
            const auto expected = discard.tile.id == YAO_JI_TILE_ID
                ? ChickenKind::ChongfengYaoJi : ChickenKind::ChongfengBaTong;
            for (const auto& item : owner.items) {
                if (item.kind == expected && item.tile == discard.tile) {
                    ordinary(event, ChickenAttributionKind::Chongfeng,
                             item.points_per_tile);
                }
            }
            continue;
        }
        // Only the immediately pending discard may be called. Later identical
        // tiles must never steal the responsibility of this event.
        const std::size_t end = d + 1 < discards.size()
            ? discard_events[d + 1]->sequence : history.size();
        const GameEvent* call = nullptr;
        for (std::size_t i = event.sequence + 1; i < end; ++i) {
            const auto& candidate = history[i];
            const auto& action = candidate.action;
            if (candidate.applied &&
                (action.type == ActionType::Pon || action.type == ActionType::Daiminkan) &&
                action.actor == discard.called_by && action.target == discard.actor &&
                action.tile == discard.tile) {
                if (call) throw std::logic_error("multiple calls for one chicken discard");
                call = &candidate;
            }
        }
        if (!call) throw std::logic_error("missing called chicken event");
        for (std::size_t i = 0; i < transfers.size(); ++i) {
            const auto& transfer = transfers[i];
            if (transfer.category != ScoreCategory::ResponsibilityChicken ||
                transfer.tile != discard.tile) continue;
            const auto source = discard.actor;
            const auto owner = discard.called_by;
            if (!((transfer.from == source && transfer.to == owner) ||
                  (transfer.from == owner && transfer.to == source))) continue;
            const int source_points = transfer.to == source
                ? transfer.points : -transfer.points;
            add(source, event, i, ChickenAttributionKind::Responsibility,
                discard.tile, source_points);
            if (call->action.type == ActionType::Pon) {
                add(owner, *call, i, ChickenAttributionKind::Responsibility,
                    discard.tile, -source_points);
            }
        }
    }
    if (settlement.chicken) {
        for (const auto& event : history) {
            if (!event.applied || event.action.type != ActionType::Pon ||
                !ChickenResolver::is_fixed_chicken(event.action.tile)) continue;
            const auto& owner = settlement.chicken->player(event.action.actor);
            if (owner.is_burned) continue;
            for (const auto& item : owner.items) {
                if (item.tile != event.action.tile || item.counts.melds < TILES_PER_MELD ||
                    (item.kind != ChickenKind::YaoJi && item.kind != ChickenKind::BaTong)) {
                    continue;
                }
                // The fourth tile of a later kakan remains unattributed in v1.
                ordinary(event, ChickenAttributionKind::Pon,
                         TILES_PER_MELD * item.points_per_tile);
            }
        }
    }
    settlement.chicken_attributions = std::move(entries);
    settlement.chicken_unattributed_deltas = residual;
}

} // namespace GYMJ
