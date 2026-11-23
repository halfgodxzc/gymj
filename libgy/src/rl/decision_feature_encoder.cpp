#include "decision_feature_encoder.h"

#include <algorithm>
#include <stdexcept>

namespace GYMJ {
namespace RL {
namespace {

constexpr size_t PON_ACTION_ID = 3 * TILE_TYPE_COUNT;

uint8_t relative_player(uint8_t absolute, uint8_t observer) {
    if (absolute >= PLAYER_COUNT) return NO_PLAYER;
    return static_cast<uint8_t>(
        (absolute - observer + PLAYER_COUNT) % PLAYER_COUNT
    );
}

ActionFamily action_family(ActionType type, RoundPhase phase) {
    switch (type) {
        case ActionType::Discard:
        case ActionType::Tsumo:
            return ActionFamily::Turn;
        case ActionType::Pon:
        case ActionType::Ron:
            return ActionFamily::Response;
        case ActionType::DeclareReady:
            return ActionFamily::Ready;
        case ActionType::Pass:
            return phase == RoundPhase::AwaitReadyDeclaration
                ? ActionFamily::Ready
                : ActionFamily::Response;
        case ActionType::AnKan:
        case ActionType::Kakan:
        case ActionType::Daiminkan:
        case ActionType::Draw:
            return ActionFamily::Unsupported;
    }
    throw std::invalid_argument("unknown action type");
}

ActionSemantics make_semantics(
    const Action& action,
    RoundPhase phase,
    uint8_t actor_id
) {
    if (action.actor != actor_id) {
        throw std::logic_error(
            "legal action actor does not match the decision maker"
        );
    }
    ActionSemantics result;
    result.type = action.type;
    result.tile = action.tile.id < TILE_TYPE_COUNT
        ? action.tile.id
        : NO_TILE;
    result.relative_source = relative_player(action.target, actor_id);
    result.relative_target = relative_player(action.actor, actor_id);
    result.family = action_family(action.type, phase);
    return result;
}

void populate_hu_features(
    CandidateActionFeatures& target,
    const HuEvaluation& evaluation,
    HuMethod method
) {
    if (!evaluation.legal || evaluation.best_candidate() == nullptr) {
        throw std::logic_error("legal hu action has no legal evaluation");
    }
    const HuCandidate& best = *evaluation.best_candidate();
    target.hu.available = 1;
    target.hu.is_tsumo = static_cast<uint8_t>(method == HuMethod::Tsumo);
    target.hu.is_qianggang = static_cast<uint8_t>(
        method == HuMethod::QiangGang
    );
    target.hu.pattern_mask = hu_pattern_feature_mask(evaluation);
    target.hu.shape_fan = static_cast<int8_t>(best.score.shape_fan);
    target.hu.ready_fan = static_cast<int8_t>(best.score.ready_fan);
    target.hu.timing_fan = static_cast<int8_t>(best.score.timing_fan);
    target.hu.special_fan = static_cast<int8_t>(best.score.special_fan);
    target.hu.total_fan = static_cast<int8_t>(best.score.total_fan);
    target.hu.hu_points = static_cast<int16_t>(best.score.hu_points);
}

void populate_ready_features(
    CandidateActionFeatures& target,
    ReadyKind kind,
    uint32_t candidate_waits,
    const CompactObservation& observation,
    const PublicPatternFeatures& public_state
) {
    ReadyActionFeatures& ready = target.ready;
    ready.available = 1;
    ready.kind = kind;
    ready.wait_mask = candidate_waits;
    for (uint8_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if ((ready.wait_mask & (uint32_t{1} << tile_id)) == 0) continue;
        ++ready.wait_kinds;
        ready.wait_copies = static_cast<uint8_t>(
            ready.wait_copies + MAX_TILE_COPIES -
            observation.visible_counts[tile_id]
        );
        ready.max_base_hu_points = std::max(
            ready.max_base_hu_points,
            public_state.hand.wait_values[tile_id].base_hu_points
        );
    }
}

void populate_pass_features(
    CandidateActionFeatures& target,
    const DecisionContext& context
) {
    target.pass.available = 1;
    for (const Action& alternative : context.legal_actions) {
        switch (alternative.type) {
            case ActionType::Pon:
                target.pass.declines_pon = 1;
                break;
            case ActionType::Daiminkan:
                target.pass.declines_daiminkan = 1;
                break;
            case ActionType::Ron:
                target.pass.declines_ron = 1;
                break;
            case ActionType::DeclareReady:
                target.pass.declines_ready = 1;
                break;
            default:
                break;
        }
    }
}

void encode_pon_follow_ups(
    DecisionFeatureSet& result,
    const CompactObservation& observation
) {
    if (observation.current_discard >= TILE_TYPE_COUNT ||
        observation.current_discard_actor >= PLAYER_COUNT) {
        throw std::logic_error("legal pon lacks a public discard context");
    }
    const uint8_t tile_id = observation.current_discard;
    if (observation.hand_counts[tile_id] < 2 ||
        observation.meld_count >= MAX_MELDS) {
        throw std::logic_error("legal pon has an invalid public hand state");
    }

    CompactObservation after_call = observation;
    after_call.hand_counts[tile_id] -= 2;
    after_call.melds[after_call.meld_count++] = {
        tile_id,
        0,
        observation.current_discard_actor,
        MeldType::Pon,
        1,
    };

    result.pon_follow_up.available = 1;
    for (uint8_t discard = 0; discard < TILE_TYPE_COUNT; ++discard) {
        if (after_call.hand_counts[discard] == 0) continue;
        CompactObservation post_discard = after_call;
        --post_discard.hand_counts[discard];
        result.pon_follow_up.discard_mask[discard] = 1;
        result.pon_follow_up.post_discard_hands[discard] =
            PatternFeatureEncoder::encode_hand(post_discard);
    }
}

} // namespace

DecisionFeatureSet DecisionFeatureEncoder::encode(
    const Arena& arena,
    uint8_t actor_id
) {
    if (actor_id >= PLAYER_COUNT) {
        throw std::out_of_range("decision feature actor must be in range 0..3");
    }
    const DecisionContext context = arena.current_decision();
    if (context.player != actor_id) {
        throw std::invalid_argument("actor is not the current decision maker");
    }

    const CompactObservation observation = ObservationEncoder::encode(
        arena,
        actor_id
    );
    DecisionFeatureSet result;
    result.observation = observation;
    result.public_state = PatternFeatureEncoder::encode(observation);
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

    const EncodedDecision encoded = ActionEncoder::encode_legal_actions(
        context.legal_actions
    );
    for (uint8_t action_id : encoded.legal_action_ids()) {
        const Action& action = encoded.action_at(action_id);
        CandidateActionFeatures& candidate = result.actions[action_id];
        candidate.legal = 1;
        candidate.semantics = make_semantics(
            action,
            context.phase,
            actor_id
        );
        candidate.supported_by_v4 = static_cast<uint8_t>(
            candidate.semantics.family != ActionFamily::Unsupported
        );

        switch (action.type) {
            case ActionType::Discard: {
                CompactObservation post_discard = observation;
                if (post_discard.hand_counts[action.tile.id] == 0) {
                    throw std::logic_error(
                        "legal discard is absent from the public hand"
                    );
                }
                --post_discard.hand_counts[action.tile.id];
                candidate.post_hand =
                    PatternFeatureEncoder::encode_hand(post_discard);
                candidate.post_hand_known = 1;
                break;
            }
            case ActionType::Pon:
                encode_pon_follow_ups(result, observation);
                break;
            case ActionType::Tsumo: {
                const HuContext hu_context = arena.current_tsumo_hu_context();
                populate_hu_features(
                    candidate,
                    HuEvaluator::evaluate(
                        arena.players_[actor_id],
                        hu_context
                    ),
                    hu_context.method
                );
                break;
            }
            case ActionType::Ron: {
                const HuContext hu_context =
                    context.phase == RoundPhase::AwaitKakanResponses
                    ? arena.kakan_hu_context(actor_id)
                    : arena.discard_hu_context(actor_id);
                populate_hu_features(
                    candidate,
                    HuEvaluator::evaluate(
                        arena.players_[actor_id],
                        hu_context
                    ),
                    hu_context.method
                );
                break;
            }
            case ActionType::DeclareReady:
                if (!arena.pending_ready_.has_value()) {
                    throw std::logic_error(
                        "ready action has no pending ready state"
                    );
                }
                populate_ready_features(
                    candidate,
                    arena.pending_ready_->kind,
                    arena.pending_ready_->candidate_waits,
                    observation,
                    result.public_state
                );
                break;
            case ActionType::Pass:
                populate_pass_features(candidate, context);
                break;
            case ActionType::AnKan:
            case ActionType::Kakan:
            case ActionType::Daiminkan:
                break;
            case ActionType::Draw:
                throw std::logic_error("draw cannot be a policy action");
        }
    }

    if (result.pon_follow_up.available &&
        !result.actions[PON_ACTION_ID].legal) {
        throw std::logic_error("pon follow-up exists without a legal pon");
    }
    return result;
}

} // namespace RL
} // namespace GYMJ
