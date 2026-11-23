#include "pattern_feature_encoder.h"

#include "chicken.h"
#include "shanten.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace GYMJ {
namespace RL {
namespace {

constexpr size_t route_index(PatternRoute route) {
    return static_cast<size_t>(route);
}

int8_t known_tenpai_status(
    const CompactObservation& observation,
    size_t relative_player
) {
    if (relative_player == 0) {
        return observation.shanten <= 0 ? int8_t{1} : int8_t{-1};
    }
    return observation.ready_kinds[relative_player] !=
            static_cast<uint8_t>(ReadyKind::None)
        ? int8_t{1}
        : int8_t{0};
}

uint8_t meld_tile_count(MeldType type) {
    return type == MeldType::Pon ? 3 : 4;
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

struct OwnMelds {
    std::vector<Meld> melds;
    uint8_t suit_mask = 0;
};

void validate_counts(const CompactObservation& observation) {
    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        const uint8_t hand = observation.hand_counts[tile_id];
        const uint8_t visible = observation.visible_counts[tile_id];
        if (hand > MAX_TILE_COPIES || visible > MAX_TILE_COPIES) {
            throw std::invalid_argument(
                "pattern feature tile count exceeds physical copies"
            );
        }
        if (visible < hand) {
            throw std::invalid_argument(
                "pattern feature visible count is below concealed count"
            );
        }
    }
    if (observation.discard_count > observation.discards.size()) {
        throw std::invalid_argument(
            "pattern feature discard count exceeds capacity"
        );
    }
    if (observation.meld_count > observation.melds.size()) {
        throw std::invalid_argument(
            "pattern feature meld count exceeds capacity"
        );
    }
}

OwnMelds collect_own_melds(const CompactObservation& observation) {
    OwnMelds result;
    for (size_t index = 0; index < observation.meld_count; ++index) {
        const MeldObservation& observed = observation.melds[index];
        if (!observed.valid) continue;
        if (observed.tile >= TILE_TYPE_COUNT ||
            observed.relative_owner >= PLAYER_COUNT ||
            observed.relative_source >= PLAYER_COUNT ||
            static_cast<size_t>(observed.type) >= MELD_TYPE_COUNT) {
            throw std::invalid_argument(
                "pattern feature meld contains an invalid field"
            );
        }
        if (observed.relative_owner != 0) continue;

        result.melds.push_back({
            observed.type,
            Tile{observed.tile},
            observed.relative_source,
        });
        result.suit_mask |= Algo::flush_suit_bit(
            observed.tile / TILES_PER_SUIT
        );
    }
    if (result.melds.size() > COMPLETE_HAND_MELD_COUNT) {
        throw std::invalid_argument("player has more than four fixed melds");
    }
    return result;
}

PlayerState build_player(
    const CompactObservation& observation,
    const OwnMelds& own_melds
) {
    PlayerState player(0);
    std::copy(
        observation.hand_counts.begin(),
        observation.hand_counts.end(),
        player.tehai
    );
    player.melds[0] = own_melds.melds;
    return player;
}

void encode_wait_values(
    const CompactObservation& observation,
    const OwnMelds& own_melds,
    HandPatternFeatures& result
) {
    PlayerState player = build_player(observation, own_melds);
    for (uint8_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (observation.hand_counts[tile_id] >= MAX_TILE_COPIES) continue;

        HuContext context;
        context.method = HuMethod::Ron;
        context.hu_tile = Tile{tile_id};
        context.bypass_ron_passport = true;
        const HuEvaluation evaluation = HuEvaluator::evaluate(player, context);
        if (!evaluation.shape_complete) continue;

        WaitValueFeatures& wait = result.wait_values[tile_id];
        wait.pattern_mask = hu_pattern_feature_mask(evaluation);
        wait.remaining_copies = static_cast<uint8_t>(
            MAX_TILE_COPIES - observation.visible_counts[tile_id]
        );
        if (const HuCandidate* best = evaluation.best_candidate()) {
            wait.base_fan = static_cast<int8_t>(best->score.total_fan);
            wait.base_hu_points = static_cast<int16_t>(
                best->score.hu_points
            );
        }

        result.shape_wait_mask |= uint32_t{1} << tile_id;
        if (wait.remaining_copies == 0) continue;
        result.live_wait_mask |= uint32_t{1} << tile_id;
        ++result.live_wait_kinds;
        result.live_wait_copies = static_cast<uint8_t>(
            result.live_wait_copies + wait.remaining_copies
        );
    }
}

std::array<PublicPlayerFeatures, PLAYER_COUNT> encode_players(
    const CompactObservation& observation
) {
    std::array<PublicPlayerFeatures, PLAYER_COUNT> players{};
    ChongfengTracker chongfeng_tracker;
    std::vector<std::pair<uint8_t, uint8_t>> called_chongfeng_candidates;

    for (size_t index = 0; index < observation.discard_count; ++index) {
        const DiscardObservation& discard = observation.discards[index];
        if (discard.tile >= TILE_TYPE_COUNT ||
            discard.relative_actor >= PLAYER_COUNT ||
            discard.was_called > 1) {
            throw std::invalid_argument(
                "pattern feature discard contains an invalid field"
            );
        }

        PublicPlayerFeatures& player = players[discard.relative_actor];
        ++player.discard_count;
        if (discard.was_called) {
            ++player.called_discard_counts[discard.tile];
        } else {
            ++player.river_counts[discard.tile];
            ++player.public_owned_tile_counts[discard.tile];
        }

        if (chongfeng_tracker.observe_discard(
                discard.relative_actor,
                Tile{discard.tile}
            )) {
            if (discard.was_called) {
                called_chongfeng_candidates.push_back({
                    discard.relative_actor,
                    discard.tile,
                });
            } else {
                ++player.chongfeng_chicken_count;
            }
        }
    }

    for (size_t index = 0; index < observation.meld_count; ++index) {
        const MeldObservation& meld = observation.melds[index];
        if (!meld.valid) continue;
        if (meld.tile >= TILE_TYPE_COUNT ||
            meld.relative_owner >= PLAYER_COUNT ||
            meld.relative_source >= PLAYER_COUNT ||
            static_cast<size_t>(meld.type) >= MELD_TYPE_COUNT) {
            throw std::invalid_argument(
                "pattern feature meld contains an invalid field"
            );
        }

        PublicPlayerFeatures& owner = players[meld.relative_owner];
        ++owner.meld_count;
        ++owner.meld_type_counts[static_cast<size_t>(meld.type)];
        const uint8_t tile_count = meld_tile_count(meld.type);
        owner.meld_tile_counts[meld.tile] = static_cast<uint8_t>(
            owner.meld_tile_counts[meld.tile] + tile_count
        );
        owner.public_owned_tile_counts[meld.tile] = static_cast<uint8_t>(
            owner.public_owned_tile_counts[meld.tile] + tile_count
        );
        if (meld.type != MeldType::Pon) {
            ++owner.bean_type_counts[bean_index(meld.type)];
        }
    }

    // A pending kakan exposes its fourth tile before robbing responses finish.
    // The meld is still represented as the original Pon until resolution.
    if (observation.phase == RoundPhase::AwaitKakanResponses) {
        if (observation.current_discard >= TILE_TYPE_COUNT ||
            observation.current_discard_actor >= PLAYER_COUNT) {
            throw std::invalid_argument(
                "pending kakan contains an invalid public tile"
            );
        }
        PublicPlayerFeatures& owner =
            players[observation.current_discard_actor];
        owner.public_owned_tile_counts[observation.current_discard] =
            static_cast<uint8_t>(
                owner.public_owned_tile_counts[
                    observation.current_discard
                ] + 1
            );
    }

    for (const auto& [source, tile] : called_chongfeng_candidates) {
        const auto called = std::find_if(
            observation.melds.begin(),
            observation.melds.begin() + observation.meld_count,
            [&](const MeldObservation& meld) {
                return meld.valid &&
                       meld.tile == tile &&
                       meld.relative_source == source &&
                       meld.relative_owner != source;
            }
        );
        if (called == observation.melds.begin() + observation.meld_count) {
            throw std::invalid_argument(
                "called fixed chicken has no matching public meld"
            );
        }
        ++players[source].responsibility_chicken_given_count;
        ++players[called->relative_owner]
            .responsibility_chicken_received_count;
    }

    for (size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        players[0].public_owned_tile_counts[tile_id] =
            static_cast<uint8_t>(
                players[0].public_owned_tile_counts[tile_id] +
                observation.hand_counts[tile_id]
            );
    }
    for (PublicPlayerFeatures& player : players) {
        player.fixed_chicken_count = static_cast<uint8_t>(
            player.public_owned_tile_counts[YAO_JI_TILE_ID] +
            player.public_owned_tile_counts[BA_TONG_TILE_ID]
        );
    }
    for (size_t relative = 0; relative < PLAYER_COUNT; ++relative) {
        PublicPlayerFeatures& player = players[relative];
        player.known_tenpai_status = known_tenpai_status(
            observation,
            relative
        );
        if (player.known_tenpai_status == 0) continue;
        const int chicken_count = player.fixed_chicken_count;
        player.signed_fixed_chicken_count = static_cast<int8_t>(
            player.known_tenpai_status > 0
                ? chicken_count
                : -chicken_count
        );
    }
    return players;
}

} // namespace

uint16_t hu_pattern_feature_mask(const HuEvaluation& evaluation) {
    uint16_t mask = 0;
    for (const HuCandidate& candidate : evaluation.candidates) {
        switch (candidate.features.pattern) {
            case HuPattern::Standard:
                mask |= PatternStandard;
                break;
            case HuPattern::AllTriplets:
                mask |= PatternAllTriplets;
                break;
            case HuPattern::SevenPairs:
                mask |= PatternSevenPairs;
                break;
            case HuPattern::DragonSevenPairs:
                mask |= PatternDragonSevenPairs;
                break;
            case HuPattern::DiLong:
                mask |= PatternDiLong;
                break;
        }
        if (candidate.features.is_flush) mask |= PatternFlush;
        if (candidate.features.is_single_wait) mask |= PatternSingleWait;
    }
    return mask;
}

HandPatternFeatures PatternFeatureEncoder::encode_hand(
    const CompactObservation& observation
) {
    validate_counts(observation);
    const OwnMelds own_melds = collect_own_melds(observation);
    const int meld_count = static_cast<int>(own_melds.melds.size());
    const int concealed_tiles = std::accumulate(
        observation.hand_counts.begin(),
        observation.hand_counts.end(),
        0
    );
    const int post_discard_tiles =
        COMPLETE_HAND_TILE_COUNT - 1 - meld_count * TILES_PER_MELD;
    const int turn_tiles =
        COMPLETE_HAND_TILE_COUNT - meld_count * TILES_PER_MELD;
    if (concealed_tiles != post_discard_tiles &&
        concealed_tiles != turn_tiles) {
        throw std::invalid_argument(
            "pattern feature hand has an invalid structural tile count"
        );
    }

    HandPatternFeatures result;
    const Algo::ShantenResult shanten = Algo::calculate_shanten(
        observation.hand_counts.data(),
        meld_count
    );
    result.routes[route_index(PatternRoute::Normal)].shanten = shanten.normal;
    result.routes[route_index(PatternRoute::SevenPairs)].shanten =
        shanten.seven_pairs;
    result.routes[route_index(PatternRoute::AllTriplets)].shanten =
        shanten.all_triplets;

    const Algo::FlushProgress progress = Algo::analyze_flush_progress(
        observation.hand_counts.data(),
        own_melds.suit_mask
    );
    for (size_t suit = 0; suit < Algo::FLUSH_SUIT_COUNT; ++suit) {
        FlushPatternFeatures& flush = result.flushes[suit];
        flush.off_suit_tiles = progress.off_suit_tiles[suit];
        flush.meld_compatible = progress.meld_compatible[suit];
        const size_t first = suit * TILES_PER_SUIT;
        for (size_t tile_id = first; tile_id < first + TILES_PER_SUIT;
             ++tile_id) {
            flush.concealed_tiles = static_cast<uint8_t>(
                flush.concealed_tiles + observation.hand_counts[tile_id]
            );
        }
    }

    if (concealed_tiles != post_discard_tiles) return result;
    result.post_discard_state = 1;

    const Algo::EffectiveDrawAnalysis effective =
        Algo::analyze_effective_draws(
            observation.hand_counts.data(),
            observation.visible_counts.data(),
            meld_count
        );
    result.routes[route_index(PatternRoute::Normal)].effective_draws =
        effective.normal;
    result.routes[route_index(PatternRoute::SevenPairs)].effective_draws =
        effective.seven_pairs;
    result.routes[route_index(PatternRoute::AllTriplets)].effective_draws =
        effective.all_triplets;

    const Algo::FlushTargetAnalysis flushes = Algo::analyze_flush_targets(
        observation.hand_counts.data(),
        observation.visible_counts.data(),
        meld_count,
        own_melds.suit_mask
    );
    result.best_flush_suit_mask = flushes.best_suit_mask;
    for (size_t suit = 0; suit < Algo::FLUSH_SUIT_COUNT; ++suit) {
        const Algo::FlushSuitCandidate& source = flushes.suits[suit];
        FlushPatternFeatures& target = result.flushes[suit];
        target.shanten = source.shanten;
        target.concealed_tiles = source.concealed_tiles;
        target.off_suit_tiles = source.off_suit_tiles;
        target.meld_compatible = source.meld_compatible;
        target.effective_draws = source.effective_draws;
    }

    encode_wait_values(observation, own_melds, result);
    return result;
}

PublicPatternFeatures PatternFeatureEncoder::encode(
    const CompactObservation& observation
) {
    PublicPatternFeatures result;
    result.hand = encode_hand(observation);
    result.players = encode_players(observation);
    return result;
}

} // namespace RL
} // namespace GYMJ
