#include "arena.h"
#include "chicken_attribution.h"

#include "action_encoder.h"
#include "guiyang_rules.h"
#include "shanten.h"
#include "hu_evaluator.h"
#include "text_log_writer.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace GYMJ {
namespace {

const char* action_type_name(ActionType type) {
    switch (type) {
        case ActionType::Pass: return "Pass";
        case ActionType::Draw: return "Draw";
        case ActionType::Discard: return "Discard";
        case ActionType::Pon: return "Pon";
        case ActionType::AnKan: return "AnKan";
        case ActionType::Kakan: return "Kakan";
        case ActionType::Daiminkan: return "Daiminkan";
        case ActionType::Tsumo: return "Tsumo";
        case ActionType::Ron: return "Ron";
        case ActionType::DeclareReady: return "DeclareReady";
    }
    return "Unknown";
}

const char* phase_name(RoundPhase phase) {
    switch (phase) {
        case RoundPhase::NotStarted: return "NotStarted";
        case RoundPhase::AwaitTurnAction: return "AwaitTurnAction";
        case RoundPhase::AwaitReadyDeclaration: return "AwaitReadyDeclaration";
        case RoundPhase::AwaitDiscardResponses: return "AwaitDiscardResponses";
        case RoundPhase::AwaitKakanResponses: return "AwaitKakanResponses";
        case RoundPhase::ReplacementDraw: return "ReplacementDraw";
        case RoundPhase::Finished: return "Finished";
    }
    return "Unknown";
}

const char* round_end_reason_name(RoundEndReason reason) {
    switch (reason) {
        case RoundEndReason::None: return "None";
        case RoundEndReason::Tsumo: return "Tsumo";
        case RoundEndReason::Ron: return "Ron";
        case RoundEndReason::QiangGang: return "QiangGang";
        case RoundEndReason::ExhaustiveDraw: return "ExhaustiveDraw";
    }
    return "Unknown";
}

const char* hu_pattern_name(HuPattern pattern) {
    switch (pattern) {
        case HuPattern::Standard: return "Standard";
        case HuPattern::AllTriplets: return "AllTriplets";
        case HuPattern::SevenPairs: return "SevenPairs";
        case HuPattern::DragonSevenPairs: return "DragonSevenPairs";
        case HuPattern::DiLong: return "DiLong";
    }
    return "Unknown";
}

const char* score_category_name(ScoreCategory category) {
    switch (category) {
        case ScoreCategory::Hu: return "Hu";
        case ScoreCategory::ChaJiao: return "ChaJiao";
        case ScoreCategory::Chicken: return "Chicken";
        case ScoreCategory::ResponsibilityChicken:
            return "ResponsibilityChicken";
        case ScoreCategory::AnKanBean: return "AnKanBean";
        case ScoreCategory::KakanBean: return "KakanBean";
        case ScoreCategory::DaiminkanBean: return "DaiminkanBean";
    }
    return "Unknown";
}

const char* fan_type_name(FanType type) {
    switch (type) {
        case FanType::PingHu: return "PingHu";
        case FanType::DaDuiZi: return "DaDuiZi";
        case FanType::SuGuaDou: return "SuGuaDou";
        case FanType::QiDui: return "QiDui";
        case FanType::QingYiSe: return "QingYiSe";
        case FanType::QingDanDiao: return "QingDanDiao";
        case FanType::LongQiDui: return "LongQiDui";
        case FanType::DiLong: return "DiLong";
        case FanType::BaoTing: return "BaoTing";
        case FanType::TianTing: return "TianTing";
        case FanType::GangKai: return "GangKai";
        case FanType::RePao: return "RePao";
        case FanType::QiangGang: return "QiangGang";
        case FanType::ShaBao: return "ShaBao";
        case FanType::DiHu: return "DiHu";
        case FanType::TianHu: return "TianHu";
    }
    return "Unknown";
}

std::string format_hand(const PlayerState& player) {
    std::ostringstream output;
    const char suits[] = {'m', 'p', 's'};
    bool first_suit = true;
    for (int suit = 0; suit < 3; ++suit) {
        std::string ranks;
        for (int rank = 0; rank < 9; ++rank) {
            const int tile_id = suit * 9 + rank;
            ranks.append(player.tehai[tile_id], static_cast<char>('1' + rank));
        }
        if (ranks.empty()) continue;
        if (!first_suit) output << ' ';
        output << ranks << suits[suit];
        first_suit = false;
    }
    return first_suit ? "(empty)" : output.str();
}

std::string format_waits(const PlayerState& player) {
    std::ostringstream output;
    bool first = true;
    for (uint8_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (!player.waits[tile_id]) continue;
        if (!first) output << ',';
        output << Tile{tile_id}.to_string();
        first = false;
    }
    return first ? "None" : output.str();
}

std::string format_action(const Action& action) {
    std::ostringstream output;
    output << action_type_name(action.type);
    if (action.tile.id < TILE_TYPE_COUNT) {
        output << '(' << action.tile.to_string() << ')';
    }
    if (action.type == ActionType::Pon ||
        action.type == ActionType::Daiminkan ||
        action.type == ActionType::Ron) {
        output << " from=P" << static_cast<int>(action.target);
    }
    return output.str();
}

void write_pattern(
    std::ostringstream& output,
    const char* name,
    const Algo::SuitPattern& pattern
) {
    output << name << "={melds:" << static_cast<int>(pattern.melds)
           << ",taatsu:" << static_cast<int>(pattern.taatsu)
           << ",pair:" << static_cast<int>(pattern.pairs) << '}';
}

} // namespace

Arena::Arena(const ArenaConfig& config)
    : Arena(config, Wall(config.seed)) {}

Arena::Arena(const ArenaConfig& config, Wall wall)
    : config_(config),
      wall_(std::move(wall)),
      players_{PlayerState(0), PlayerState(1), PlayerState(2), PlayerState(3)},
      current_player_(config.dealer),
      phase_(RoundPhase::NotStarted),
      last_drawn_{255},
      last_draw_kind_(DrawKind::Opening),
      must_discard_(false),
      pending_discard_{ActionType::Pass, 0, Tile{255}, 0},
      response_order_{0, 0, 0},
      response_index_(0) {
    if (config_.dealer >= 4) {
        throw std::invalid_argument("dealer must be in range 0..3");
    }
    if (!config_.log_path.empty()) {
        log_ = std::make_unique<TextLogWriter>(config_.log_path);
    }
}

Arena::~Arena() = default;

void Arena::start_round() {
    if (phase_ != RoundPhase::NotStarted) {
        throw std::logic_error("round has already started");
    }
    if (wall_.remaining() < 53) {
        throw std::invalid_argument("wall needs at least 53 tiles to start a round");
    }

    deal_initial_hands();
    log_round_start();
    current_player_ = config_.dealer;
    initial_ready_index_ = 0;
    advance_initial_ready_declarations();
}

DecisionContext Arena::current_decision() const {
    if (phase_ == RoundPhase::AwaitTurnAction) {
        if (ready_states_[current_player_].declared()) {
            return {
                phase_,
                current_player_,
                legal_actions_for_reported_turn(),
            };
        }

        auto actions = players_[current_player_].get_legal_actions_on_draw(last_drawn_);
        if (wall_.empty()) {
            actions.erase(
                std::remove_if(
                    actions.begin(),
                    actions.end(),
                    [](const Action& action) {
                        return action.type == ActionType::AnKan ||
                               action.type == ActionType::Kakan;
                    }
                ),
                actions.end()
            );
        }
        if (!must_discard_ && can_current_player_tsumo()) {
            actions.push_back({
                ActionType::Tsumo,
                current_player_,
                last_drawn_,
                current_player_,
            });
        }
        if (must_discard_) {
            actions.erase(
                std::remove_if(actions.begin(), actions.end(), [](const Action& action) {
                    return action.type != ActionType::Discard;
                }),
                actions.end()
            );
        }
        return {phase_, current_player_, std::move(actions)};
    }

    if (phase_ == RoundPhase::AwaitReadyDeclaration) {
        if (!pending_ready_.has_value()) {
            throw std::logic_error("ready phase has no pending decision");
        }
        const uint8_t player_id = pending_ready_->player;
        return {
            phase_,
            player_id,
            {
                {
                    ActionType::DeclareReady,
                    player_id,
                    Tile{255},
                    player_id,
                },
                {
                    ActionType::Pass,
                    player_id,
                    Tile{255},
                    player_id,
                },
            },
        };
    }

    if (phase_ == RoundPhase::AwaitDiscardResponses) {
        const uint8_t responder = response_order_[response_index_];
        auto actions = legal_actions_for_discard_response(responder);
        return {phase_, responder, std::move(actions)};
    }

    if (phase_ == RoundPhase::AwaitKakanResponses) {
        const uint8_t responder = response_order_[response_index_];
        auto actions = legal_actions_for_kakan_response(responder);
        return {phase_, responder, std::move(actions)};
    }

    throw std::logic_error("arena is not waiting for a player decision");
}

void Arena::submit_action(const Action& action) {
    const DecisionContext context = current_decision();
    const bool legal = std::any_of(
        context.legal_actions.begin(),
        context.legal_actions.end(),
        [&](const Action& candidate) {
            return actions_equal(candidate, action);
        }
    );
    if (!legal) throw std::invalid_argument("submitted action is not legal");
    log_decision(context, action);
    decision_sequence_++;

    if (phase_ == RoundPhase::AwaitReadyDeclaration) {
        resolve_ready_decision(action);
        return;
    }

    if (phase_ == RoundPhase::AwaitTurnAction) {
        const RoundPhase before = phase_;
        switch (action.type) {
            case ActionType::Discard: {
                const bool first_discard =
                    !progress_[action.actor].has_discarded;
                const bool offer_baoting =
                    should_offer_baoting_after(action);
                const RoundPhase after = offer_baoting
                    ? RoundPhase::AwaitReadyDeclaration
                    : RoundPhase::AwaitDiscardResponses;

                pending_discard_is_hot_ =
                    hot_discard_player_.has_value() &&
                    *hot_discard_player_ == action.actor;
                if (pending_discard_is_hot_) {
                    hot_discard_player_.reset();
                }

                broadcast(action, before, after);
                progress_[action.actor].has_discarded = true;
                pending_discard_ = action;

                if (offer_baoting) {
                    begin_ready_decision(
                        action.actor,
                        ReadyKind::BaoTing,
                        ReadyDecisionOrigin::AfterFirstDiscard
                    );
                } else {
                    if (first_discard &&
                        !ready_states_[action.actor].declared()) {
                        progress_[action.actor].ready_window_closed = true;
                    }
                    begin_discard_responses(action);
                }
                return;
            }
            case ActionType::AnKan:
                progress_[action.actor].opening_hand_changed = true;
                progress_[action.actor].ready_window_closed = true;
                broadcast(action, before, RoundPhase::ReplacementDraw);
                hot_discard_player_ = action.actor;
                phase_ = RoundPhase::ReplacementDraw;
                draw_for_current_player(DrawKind::Replacement);
                return;
            case ActionType::Kakan:
                progress_[action.actor].opening_hand_changed = true;
                progress_[action.actor].ready_window_closed = true;
                begin_kakan_responses(action);
                return;
            case ActionType::Tsumo:
                result_.reason = RoundEndReason::Tsumo;
                result_.hu_players = {action.actor};
                result_.hu_tile = action.tile;
                prepare_hu_settlement(
                    {
                        HuTileOrigin::SelfDraw,
                        action.tile,
                        action.actor,
                        {action.actor},
                    },
                    {current_tsumo_hu_context()}
                );
                broadcast(action, before, RoundPhase::Finished);
                phase_ = RoundPhase::Finished;
                log_round_end();
                return;
            default:
                throw std::logic_error("unexpected turn action");
        }
    }

    if (phase_ == RoundPhase::AwaitDiscardResponses &&
        action.type == ActionType::Pass &&
        std::any_of(context.legal_actions.begin(), context.legal_actions.end(),
                    [](const Action& a) { return a.type == ActionType::Pon; })) {
        progress_[action.actor].passed_pon_tiles |= uint32_t{1} << action.tile.id;
    }
    responses_[action.actor] = action;
    response_index_++;
    if (response_index_ == response_order_.size()) {
        if (phase_ == RoundPhase::AwaitDiscardResponses) {
            resolve_discard_responses();
        } else if (phase_ == RoundPhase::AwaitKakanResponses) {
            resolve_kakan_responses();
        } else {
            throw std::logic_error("responses completed outside response phase");
        }
    }
}

RoundPhase Arena::phase() const {
    return phase_;
}

bool Arena::is_finished() const {
    return phase_ == RoundPhase::Finished;
}

uint8_t Arena::current_player() const {
    return current_player_;
}

uint8_t Arena::dealer() const {
    return config_.dealer;
}

size_t Arena::wall_remaining() const {
    return wall_.remaining();
}

Tile Arena::response_tile() const {
    if (phase_ == RoundPhase::AwaitDiscardResponses) {
        return pending_discard_.tile;
    }
    if (phase_ == RoundPhase::AwaitKakanResponses &&
        pending_kakan_.has_value()) {
        return pending_kakan_->tile;
    }
    throw std::logic_error("arena is not resolving a tile response");
}

uint8_t Arena::response_actor() const {
    if (phase_ == RoundPhase::AwaitDiscardResponses) {
        return pending_discard_.actor;
    }
    if (phase_ == RoundPhase::AwaitKakanResponses &&
        pending_kakan_.has_value()) {
        return pending_kakan_->actor;
    }
    throw std::logic_error("arena is not resolving a tile response");
}

const PlayerState& Arena::player(uint8_t id) const {
    if (id >= 4) throw std::out_of_range("player id must be in range 0..3");
    return players_[id];
}

const ReadyState& Arena::ready_state(uint8_t id) const {
    if (id >= 4) throw std::out_of_range("player id must be in range 0..3");
    return ready_states_[id];
}

const RoundResult& Arena::result() const {
    return result_;
}

const SettlementSnapshot& Arena::settlement_snapshot() const {
    if (!settlement_snapshot_.has_value()) {
        throw std::logic_error("round has no hu settlement snapshot");
    }
    return *settlement_snapshot_;
}

const ChickenResult& Arena::chicken_result() const {
    if (!chicken_result_.has_value()) {
        throw std::logic_error("round has no chicken result");
    }
    return *chicken_result_;
}

const RoundSettlementResult& Arena::round_settlement() const {
    if (!round_settlement_.has_value()) {
        throw std::logic_error("round has no settlement result");
    }
    return *round_settlement_;
}

const std::vector<GameEvent>& Arena::history() const {
    return history_;
}

const std::vector<DiscardEvent>& Arena::discard_history() const {
    return discard_history_;
}

std::vector<ObservedDiscardEvent> Arena::discard_history_for(uint8_t observer_id) const {
    if (observer_id >= 4) {
        throw std::out_of_range("observer id must be in range 0..3");
    }

    std::vector<ObservedDiscardEvent> observed;
    observed.reserve(discard_history_.size());
    for (const auto& event : discard_history_) {
        observed.push_back({
            event.tile,
            static_cast<uint8_t>((event.actor - observer_id + 4) % 4),
            event.was_called,
        });
    }
    return observed;
}

void Arena::deal_initial_hands() {
    for (int tile_index = 0; tile_index < 13; ++tile_index) {
        for (uint8_t player_id = 0; player_id < 4; ++player_id) {
            const Action draw{
                ActionType::Draw,
                player_id,
                wall_.draw(),
                player_id,
            };
            broadcast(draw, RoundPhase::NotStarted, RoundPhase::NotStarted);
        }
    }
}

void Arena::advance_initial_ready_declarations() {
    while (initial_ready_index_ < PLAYER_COUNT - 1) {
        const uint8_t player_id = static_cast<uint8_t>(
            (config_.dealer + 1 + initial_ready_index_) % PLAYER_COUNT
        );
        initial_ready_index_++;

        if (players_[player_id].shanten == 0 &&
            wait_mask(players_[player_id]) != 0) {
            begin_ready_decision(
                player_id,
                ReadyKind::TianTing,
                ReadyDecisionOrigin::BeforeOpeningDraw
            );
            return;
        }
    }

    current_player_ = config_.dealer;
    draw_for_current_player(DrawKind::Opening);
}

void Arena::prepare_normal_turn(uint8_t player_id) {
    current_player_ = player_id;
    if (wall_.empty()) {
        finish_exhaustive_draw();
        return;
    }

    draw_for_current_player(DrawKind::Normal);
}

void Arena::draw_for_current_player(DrawKind kind) {
    if (wall_.empty()) {
        if (kind == DrawKind::Replacement) {
            throw std::logic_error(
                "replacement draw requires a remaining wall tile"
            );
        }
        finish_exhaustive_draw();
        return;
    }

    const RoundPhase before = phase_;
    last_drawn_ = kind == DrawKind::Replacement
        ? wall_.draw_replacement()
        : wall_.draw();
    last_draw_kind_ = kind;
    progress_[current_player_].passed_pon_tiles = 0;
    if (kind == DrawKind::Normal) {
        progress_[current_player_].has_normal_drawn = true;
    }

    const Action draw{
        ActionType::Draw,
        current_player_,
        last_drawn_,
        current_player_,
    };
    broadcast(draw, before, RoundPhase::AwaitTurnAction);
    phase_ = RoundPhase::AwaitTurnAction;
    must_discard_ = false;
}

void Arena::begin_ready_decision(
    uint8_t player_id,
    ReadyKind kind,
    ReadyDecisionOrigin origin
) {
    pending_ready_ = PendingReadyDecision{
        player_id,
        kind,
        origin,
        wait_mask(players_[player_id]),
    };
    phase_ = RoundPhase::AwaitReadyDeclaration;
}

void Arena::resolve_ready_decision(const Action& action) {
    if (!pending_ready_.has_value()) {
        throw std::logic_error("ready decision is missing");
    }

    const PendingReadyDecision decision = *pending_ready_;
    const bool declared = action.type == ActionType::DeclareReady;
    if (declared) {
        ready_states_[decision.player] = {
            decision.kind,
            decision.candidate_waits,
        };
        progress_[decision.player].ready_window_closed = true;
    } else if (decision.origin == ReadyDecisionOrigin::AfterFirstDiscard) {
        progress_[decision.player].ready_window_closed = true;
    }

    if (decision.origin == ReadyDecisionOrigin::BeforeOpeningDraw) {
        broadcast(
            action,
            RoundPhase::AwaitReadyDeclaration,
            RoundPhase::AwaitReadyDeclaration
        );
        pending_ready_.reset();
        advance_initial_ready_declarations();
        return;
    }

    broadcast(
        action,
        RoundPhase::AwaitReadyDeclaration,
        RoundPhase::AwaitDiscardResponses
    );
    pending_ready_.reset();
    begin_discard_responses(pending_discard_);
}

void Arena::begin_discard_responses(const Action& discard) {
    pending_discard_ = discard;
    reset_responses(discard.actor);
    phase_ = RoundPhase::AwaitDiscardResponses;
}

void Arena::begin_kakan_responses(const Action& kakan) {
    pending_kakan_ = kakan;
    reset_responses(kakan.actor);
    record_event(
        kakan,
        RoundPhase::AwaitTurnAction,
        RoundPhase::AwaitKakanResponses,
        false
    );
    phase_ = RoundPhase::AwaitKakanResponses;
}

void Arena::reset_responses(uint8_t actor) {
    responses_.fill(std::nullopt);
    response_index_ = 0;
    for (uint8_t offset = 1; offset <= 3; ++offset) {
        response_order_[offset - 1] = (actor + offset) % 4;
    }
}

void Arena::resolve_discard_responses() {
    std::vector<Action> ron_actions;
    for (uint8_t responder : response_order_) {
        const Action& response = *responses_[responder];
        if (response.type == ActionType::Ron) ron_actions.push_back(response);
    }

    if (!ron_actions.empty()) {
        result_.reason = RoundEndReason::Ron;
        result_.fangpao_player = pending_discard_.actor;
        result_.hu_tile = pending_discard_.tile;
        result_.is_hot_cannon = pending_discard_is_hot_;
        for (const auto& action : ron_actions) {
            result_.hu_players.push_back(action.actor);
        }
        std::vector<HuContext> hu_contexts;
        hu_contexts.reserve(result_.hu_players.size());
        for (uint8_t hu_player : result_.hu_players) {
            hu_contexts.push_back(discard_hu_context(hu_player));
        }
        prepare_hu_settlement(
            {
                HuTileOrigin::Discard,
                pending_discard_.tile,
                pending_discard_.actor,
                result_.hu_players,
            },
            hu_contexts
        );
        for (const auto& action : ron_actions) {
            broadcast(action, RoundPhase::AwaitDiscardResponses, RoundPhase::Finished);
        }
        phase_ = RoundPhase::Finished;
        log_round_end();
        return;
    }

    const Action* selected = nullptr;
    for (uint8_t responder : response_order_) {
        const Action& response = *responses_[responder];
        if (selected == nullptr ||
            GuiyangRules::response_priority(response.type) >
                GuiyangRules::response_priority(selected->type)) {
            selected = &response;
        }
    }

    if (selected != nullptr && selected->type == ActionType::Daiminkan) {
        current_player_ = selected->actor;
        progress_[selected->actor].opening_hand_changed = true;
        progress_[selected->actor].ready_window_closed = true;
        broadcast(
            *selected,
            RoundPhase::AwaitDiscardResponses,
            RoundPhase::ReplacementDraw
        );
        hot_discard_player_ = selected->actor;
        phase_ = RoundPhase::ReplacementDraw;
        draw_for_current_player(DrawKind::Replacement);
        return;
    }

    if (selected != nullptr && selected->type == ActionType::Pon) {
        current_player_ = selected->actor;
        progress_[selected->actor].opening_hand_changed = true;
        progress_[selected->actor].ready_window_closed = true;
        broadcast(
            *selected,
            RoundPhase::AwaitDiscardResponses,
            RoundPhase::AwaitTurnAction
        );
        phase_ = RoundPhase::AwaitTurnAction;
        must_discard_ = true;
        last_drawn_ = selected->tile;
        return;
    }

    prepare_normal_turn((pending_discard_.actor + 1) % 4);
}

void Arena::resolve_kakan_responses() {
    if (!pending_kakan_.has_value()) {
        throw std::logic_error("kakan response has no pending action");
    }

    const Action kakan = *pending_kakan_;
    std::vector<Action> ron_actions;
    for (uint8_t responder : response_order_) {
        const Action& response = *responses_[responder];
        if (response.type == ActionType::Ron) ron_actions.push_back(response);
    }

    if (!ron_actions.empty()) {
        result_.reason = RoundEndReason::QiangGang;
        result_.fangpao_player = kakan.actor;
        result_.hu_tile = kakan.tile;
        for (const auto& action : ron_actions) {
            result_.hu_players.push_back(action.actor);
        }
        std::vector<HuContext> hu_contexts;
        hu_contexts.reserve(result_.hu_players.size());
        for (uint8_t hu_player : result_.hu_players) {
            hu_contexts.push_back(kakan_hu_context(hu_player));
        }
        prepare_hu_settlement(
            {
                HuTileOrigin::Kakan,
                kakan.tile,
                kakan.actor,
                result_.hu_players,
            },
            hu_contexts
        );
        for (const auto& action : ron_actions) {
            broadcast(
                action,
                RoundPhase::AwaitKakanResponses,
                RoundPhase::Finished
            );
        }
        pending_kakan_.reset();
        phase_ = RoundPhase::Finished;
        log_round_end();
        return;
    }

    if (history_.empty() ||
        history_.back().action.type != ActionType::Kakan ||
        history_.back().applied) {
        throw std::logic_error("pending kakan history event is missing");
    }
    apply_to_players(kakan);
    record_event(
        kakan,
        RoundPhase::AwaitKakanResponses,
        RoundPhase::ReplacementDraw,
        true
    );
    pending_kakan_.reset();
    hot_discard_player_ = kakan.actor;
    phase_ = RoundPhase::ReplacementDraw;
    draw_for_current_player(DrawKind::Replacement);
}

void Arena::apply_to_players(const Action& action) {
    for (auto& player_state : players_) {
        player_state.apply_action(action);
    }
    update_discard_history(action);
}

void Arena::record_event(
    const Action& action,
    RoundPhase before,
    RoundPhase after,
    bool applied
) {
    history_.push_back({
        static_cast<uint32_t>(history_.size()),
        action,
        before,
        after,
        applied,
    });
}

void Arena::broadcast(const Action& action, RoundPhase before, RoundPhase after) {
    apply_to_players(action);
    record_event(action, before, after, true);
}

void Arena::update_discard_history(const Action& action) {
    if (action.type == ActionType::Discard) {
        discard_history_.push_back({
            action.tile,
            action.actor,
            false,
            INVALID_PLAYER_ID,
        });
        return;
    }

    if (action.type != ActionType::Pon &&
        action.type != ActionType::Daiminkan) {
        return;
    }

    const auto called = std::find_if(
        discard_history_.rbegin(),
        discard_history_.rend(),
        [&](const DiscardEvent& event) {
            return !event.was_called &&
                   event.actor == action.target &&
                   event.tile == action.tile;
        }
    );
    if (called == discard_history_.rend()) {
        throw std::logic_error("called discard is missing from history");
    }
    called->was_called = true;
    called->called_by = action.actor;
}

void Arena::finish_exhaustive_draw() {
    result_.reason = RoundEndReason::ExhaustiveDraw;
    const auto cha_jiao_settlements =
        RoundSettlementCalculator::evaluate_cha_jiao(
        players_,
        ready_states_
    );
    round_settlement_ = RoundSettlementCalculator::calculate(
        result_,
        {},
        cha_jiao_settlements,
        std::nullopt,
        BeanResult{}
    );
    phase_ = RoundPhase::Finished;
    log_round_end();
}

void Arena::prepare_hu_settlement(
    const HuTileClaim& hu_tile_claim,
    const std::vector<HuContext>& hu_contexts
) {
    if (hu_tile_claim.hu_players.size() != hu_contexts.size()) {
        throw std::invalid_argument(
            "hu contexts do not match hu players"
        );
    }
    settlement_snapshot_ = SettlementSnapshotBuilder::build(
        players_,
        hu_tile_claim
    );
    ChickenContext context = build_chicken_context();
    if (!wall_.empty()) {
        context.indicator = wall_.reveal_chicken_indicator();
    }
    chicken_result_ = ChickenResolver::resolve(
        *settlement_snapshot_,
        context
    );

    std::vector<HuSettlement> hu_settlements;
    hu_settlements.reserve(hu_tile_claim.hu_players.size());
    for (std::size_t index = 0;
         index < hu_tile_claim.hu_players.size();
         ++index) {
        const uint8_t hu_player_id = hu_tile_claim.hu_players[index];
        const auto evaluation = HuEvaluator::evaluate(
            players_[hu_player_id],
            hu_contexts[index]
        );
        if (!evaluation.legal || evaluation.best_candidate() == nullptr) {
            throw std::logic_error(
                "hu settlement has no legal hu evaluation"
            );
        }

        HuSettlement hu;
        hu.player_id = hu_player_id;
        hu.features = evaluation.best_candidate()->features;
        hu.score = evaluation.best_candidate()->score;
        if (hu.score.total_fan == 1 &&
            !chicken_result_->player(hu_player_id).has_any_chicken()) {
            hu.features.is_su_gua_dou = true;
            hu.score = GuiyangScoring::calculate(hu.features);
        }
        hu_settlements.push_back(std::move(hu));
    }

    const BeanResult beans = BeanScoring::calculate(
        players_,
        jiao_states()
    );
    round_settlement_ = RoundSettlementCalculator::calculate(
        result_,
        hu_settlements,
        {},
        chicken_result_,
        beans
    );
    attribute_chicken_settlement(*round_settlement_, history_, discard_history_);
}

ChickenContext Arena::build_chicken_context() const {
    if (!settlement_snapshot_.has_value()) {
        throw std::logic_error(
            "chicken context requires a settlement snapshot"
        );
    }

    ChickenContext context;
    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        context.is_jiao[player_id] = players_[player_id].shanten <= 0;
    }
    if (result_.is_hot_cannon && result_.fangpao_player < PLAYER_COUNT) {
        context.burned_players[result_.fangpao_player] = true;
    }

    ChongfengTracker chongfeng_tracker;
    for (const auto& event : discard_history_) {
        if (!chongfeng_tracker.observe_discard(event.actor, event.tile)) {
            continue;
        }
        const auto& player = settlement_snapshot_->player(event.actor);
        if (!event.was_called && player.discard_count(event.tile) > 0) {
            context.chongfeng_tile_masks[event.actor] |=
                event.tile.id == YAO_JI_TILE_ID
                    ? CHONGFENG_YAO_JI_MASK
                    : CHONGFENG_BA_TONG_MASK;
        } else if (event.was_called) {
            if (event.called_by >= PLAYER_COUNT) {
                throw std::logic_error(
                    "called fixed chicken has no caller"
                );
            }
            context.responsibility_chickens.push_back({
                event.actor,
                event.called_by,
                event.tile,
            });
        }
    }
    return context;
}

void Arena::log_round_start() {
    if (!log_) return;

    std::ostringstream output;
    output << "=== Round Start ===\n"
           << "Seed: " << config_.seed << '\n'
           << "Dealer: P" << static_cast<int>(config_.dealer) << '\n';
    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        const PlayerState& player = players_[player_id];
        output << "P" << static_cast<int>(player_id)
               << " initial hand: " << format_hand(player)
               << " | shanten=" << static_cast<int>(player.shanten)
               << " | waits=" << format_waits(player) << '\n';
    }
    output << '\n';
    log_->write(output.str());
}

void Arena::log_decision(
    const DecisionContext& context,
    const Action& selected_action
) {
    if (!log_) return;

    const PlayerState& player = players_[context.player];
    const int meld_count = static_cast<int>(player.melds[0].size());
    const Algo::ShantenAnalysis analysis = Algo::analyze_shanten(
        player.tehai,
        meld_count
    );
    const RL::EncodedDecision encoded =
        RL::ActionEncoder::encode_legal_actions(context.legal_actions);
    const uint8_t selected_id =
        RL::ActionEncoder::encode_action(selected_action);

    std::ostringstream output;
    output << "[Decision " << decision_sequence_ << "]\n"
           << "Phase: " << phase_name(context.phase) << '\n'
           << "Actor: P" << static_cast<int>(context.player) << '\n'
           << "Hand: " << format_hand(player)
           << " | open_melds=" << meld_count << '\n'
           << "Normal split: ";
    write_pattern(output, "wan", analysis.wan);
    output << ' ';
    write_pattern(output, "tong", analysis.tong);
    output << ' ';
    write_pattern(output, "tiao", analysis.tiao);
    output << '\n'
           << "Shanten: normal=" << static_cast<int>(analysis.result.normal)
           << " seven_pairs="
           << static_cast<int>(analysis.result.seven_pairs)
           << " all_triplets="
           << static_cast<int>(analysis.result.all_triplets)
           << " best=" << static_cast<int>(analysis.result.best) << '\n'
           << "Waits: " << format_waits(player) << '\n'
           << "Legal actions:";

    for (uint8_t action_id : encoded.legal_action_ids()) {
        output << "\n  " << static_cast<int>(action_id) << ": "
               << format_action(encoded.action_at(action_id));
    }
    output << "\nSelected action: " << static_cast<int>(selected_id)
           << ": " << format_action(selected_action) << "\n\n";

    log_->write(output.str());
}

void Arena::log_round_end() {
    if (!log_ || round_end_logged_) return;
    if (!round_settlement_.has_value()) {
        throw std::logic_error("finished round has no settlement to log");
    }

    const RoundSettlementResult& settlement = *round_settlement_;
    std::ostringstream output;
    output << "=== Round End ===\n"
           << "Reason: " << round_end_reason_name(result_.reason) << '\n'
           << "Hu players:";
    if (result_.hu_players.empty()) {
        output << " None";
    } else {
        for (uint8_t player_id : result_.hu_players) {
            output << " P" << static_cast<int>(player_id);
        }
    }
    output << "\nFangpao player: ";
    if (result_.fangpao_player < PLAYER_COUNT) {
        output << 'P' << static_cast<int>(result_.fangpao_player);
    } else {
        output << "None";
    }
    output << "\nHu tile: ";
    if (result_.hu_tile.id < TILE_TYPE_COUNT) {
        output << result_.hu_tile.to_string();
    } else {
        output << "None";
    }
    output << '\n';

    for (const HuSettlement& hu : settlement.hu_settlements) {
        output << "Hu P" << static_cast<int>(hu.player_id)
               << ": pattern=" << hu_pattern_name(hu.features.pattern)
               << " flush=" << (hu.features.is_flush ? "true" : "false")
               << " single_wait="
               << (hu.features.is_single_wait ? "true" : "false")
               << " su_gua_dou="
               << (hu.features.is_su_gua_dou ? "true" : "false")
               << " total_fan=" << hu.score.total_fan
               << " hu_points=" << hu.score.hu_points << '\n'
               << "  Fan items:";
        for (const FanItem& item : hu.score.items) {
            output << ' ' << fan_type_name(item.type)
                   << '=' << item.fan;
        }
        output << '\n';
    }

    for (const ChaJiaoSettlement& cha_jiao :
         settlement.cha_jiao_settlements) {
        output << "ChaJiao P" << static_cast<int>(cha_jiao.player_id)
               << ": best_wait=" << cha_jiao.best_wait.to_string()
               << " pattern=" << hu_pattern_name(cha_jiao.features.pattern)
               << " total_fan=" << cha_jiao.score.total_fan
               << " hu_points=" << cha_jiao.score.hu_points << '\n';
    }

    if (settlement.chicken.has_value()) {
        const ChickenResult& chicken = *settlement.chicken;
        output << "Chicken indicator: ";
        if (chicken.indicator.has_value()) {
            output << chicken.indicator->to_string();
        } else {
            output << "None";
        }
        output << " | flipped: ";
        if (chicken.flipped_chicken.has_value()) {
            output << chicken.flipped_chicken->to_string();
        } else {
            output << "None";
        }
        output << '\n';
        for (const PlayerChickenResult& player : chicken.players) {
            output << "Chicken P" << static_cast<int>(player.player_id)
                   << ": jiao=" << (player.is_jiao ? "true" : "false")
                   << " burned=" << (player.is_burned ? "true" : "false")
                   << " tiles=" << player.total_tile_count
                   << " points=" << player.total_points << '\n';
        }
    } else {
        output << "Chicken: not calculated\n";
    }

    output << "Bean items: " << settlement.beans.items.size() << '\n'
           << "Transfers:";
    for (const ScoreTransfer& transfer : settlement.ledger.transfers()) {
        output << "\n  " << score_category_name(transfer.category)
               << " P" << static_cast<int>(transfer.from)
               << " -> P" << static_cast<int>(transfer.to)
               << " points=" << transfer.points;
        if (transfer.tile.id < TILE_TYPE_COUNT) {
            output << " tile=" << transfer.tile.to_string();
        }
    }
    output << "\nScore deltas:";
    const auto& deltas = settlement.ledger.score_deltas();
    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        output << " P" << static_cast<int>(player_id)
               << '=' << deltas[player_id];
    }
    output << "\n=== End ===\n";

    log_->write(output.str());
    round_end_logged_ = true;
    log_.reset();
}

bool Arena::should_offer_baoting_after(const Action& discard) const {
    const auto& progress = progress_[discard.actor];
    if (ready_states_[discard.actor].declared() ||
        progress.ready_window_closed ||
        progress.has_discarded ||
        progress.opening_hand_changed) {
        return false;
    }
    if (discard.actor != config_.dealer && !progress.has_normal_drawn) {
        return false;
    }
    if (discard.tile.id >= TILE_TYPE_COUNT ||
        players_[discard.actor].tehai[discard.tile.id] == 0) {
        return false;
    }

    uint8_t hand_after_discard[TILE_TYPE_COUNT];
    std::copy(
        players_[discard.actor].tehai,
        players_[discard.actor].tehai + TILE_TYPE_COUNT,
        hand_after_discard
    );
    hand_after_discard[discard.tile.id]--;
    return Algo::calculate_shanten(
        hand_after_discard,
        static_cast<int>(players_[discard.actor].melds[0].size())
    ).best == 0;
}

bool Arena::can_current_player_tsumo() const {
    const PlayerState& player = players_[current_player_];
    if (last_drawn_.id >= TILE_TYPE_COUNT ||
        player.tehai[last_drawn_.id] == 0) {
        return false;
    }

    return HuEvaluator::evaluate(
        player,
        current_tsumo_hu_context()
    ).legal;
}

HuContext Arena::current_tsumo_hu_context() const {
    HuContext context;
    context.method = HuMethod::Tsumo;
    context.hu_tile = last_drawn_;
    context.hu_player_ready_kind = ready_states_[current_player_].kind;

    const auto& progress = progress_[current_player_];
    if (last_draw_kind_ == DrawKind::Opening &&
        current_player_ == config_.dealer) {
        context.timing = HuTiming::TianHu;
    } else if (last_draw_kind_ == DrawKind::Normal &&
               current_player_ != config_.dealer &&
               !progress.has_discarded &&
               !progress.opening_hand_changed) {
        context.timing = HuTiming::DiHu;
    }

    if (last_draw_kind_ == DrawKind::Replacement) {
        context.special = SpecialHu::GangKai;
    }
    return context;
}

HuContext Arena::discard_hu_context(uint8_t responder) const {
    if (responder >= PLAYER_COUNT) {
        throw std::out_of_range("discard responder is out of range");
    }
    const bool discarder_reported =
        ready_states_[pending_discard_.actor].declared();

    HuContext context;
    context.method = HuMethod::Ron;
    context.hu_tile = pending_discard_.tile;
    context.hu_player_ready_kind = ready_states_[responder].kind;
    context.source_ready_kind = ready_states_[pending_discard_.actor].kind;
    if (discarder_reported) {
        context.special = context.special | SpecialHu::ShaBao;
    }
    if (pending_discard_is_hot_) {
        context.special = context.special | SpecialHu::RePao;
        context.bypass_ron_passport = true;
    }
    return context;
}

HuContext Arena::kakan_hu_context(uint8_t responder) const {
    if (!pending_kakan_.has_value()) {
        throw std::logic_error("kakan response has no pending action");
    }
    if (responder >= PLAYER_COUNT) {
        throw std::out_of_range("kakan responder is out of range");
    }

    HuContext context;
    context.method = HuMethod::QiangGang;
    context.hu_tile = pending_kakan_->tile;
    context.hu_player_ready_kind = ready_states_[responder].kind;
    context.source_ready_kind = ready_states_[pending_kakan_->actor].kind;
    if (context.source_ready_kind != ReadyKind::None) {
        context.special = context.special | SpecialHu::ShaBao;
    }
    return context;
}

std::array<bool, PLAYER_COUNT> Arena::jiao_states() const {
    std::array<bool, PLAYER_COUNT> states{};
    for (uint8_t player_id = 0; player_id < PLAYER_COUNT; ++player_id) {
        states[player_id] = players_[player_id].shanten <= 0;
    }
    return states;
}

std::vector<Action> Arena::legal_actions_for_reported_turn() const {
    const PlayerState& player = players_[current_player_];
    const ReadyState& ready = ready_states_[current_player_];
    std::vector<Action> actions;

    if (last_drawn_.id >= TILE_TYPE_COUNT ||
        player.tehai[last_drawn_.id] == 0) {
        throw std::logic_error("reported player has no current drawn tile");
    }

    actions.push_back({
        ActionType::Discard,
        current_player_,
        last_drawn_,
        current_player_,
    });
    if ((ready.locked_waits & (uint32_t{1} << last_drawn_.id)) != 0 &&
        can_current_player_tsumo()) {
        actions.push_back({
            ActionType::Tsumo,
            current_player_,
            last_drawn_,
            current_player_,
        });
    }
    return actions;
}

std::vector<Action> Arena::legal_actions_for_discard_response(
    uint8_t responder
) const {
    const bool responder_reported = ready_states_[responder].declared();
    std::vector<Action> actions;
    if (responder_reported || wall_.empty()) {
        actions.push_back({
            ActionType::Pass,
            responder,
            pending_discard_.tile,
            pending_discard_.actor,
        });
    } else {
        actions = players_[responder].get_legal_actions_on_opponent_discard(
            pending_discard_.actor,
            pending_discard_.tile
        );
        actions.erase(
            std::remove_if(
                actions.begin(),
                actions.end(),
                [&](const Action& action) {
                    return action.type == ActionType::Ron ||
                        (action.type == ActionType::Pon &&
                         (progress_[responder].passed_pon_tiles &
                          (uint32_t{1} << action.tile.id)) != 0);
                }
            ),
            actions.end()
        );
    }

    const bool matches_locked_wait =
        !responder_reported ||
        (ready_states_[responder].locked_waits &
         (uint32_t{1} << pending_discard_.tile.id)) != 0;
    const HuContext hu_context = discard_hu_context(responder);

    if (matches_locked_wait &&
        HuEvaluator::evaluate(
            players_[responder],
            hu_context
        ).legal) {
        actions.push_back({
            ActionType::Ron,
            responder,
            pending_discard_.tile,
            pending_discard_.actor,
        });
    }
    return actions;
}

std::vector<Action> Arena::legal_actions_for_kakan_response(
    uint8_t responder
) const {
    if (!pending_kakan_.has_value()) {
        throw std::logic_error("kakan response has no pending action");
    }

    const Action& kakan = *pending_kakan_;
    std::vector<Action> actions{{
        ActionType::Pass,
        responder,
        kakan.tile,
        kakan.actor,
    }};

    const HuContext hu_context = kakan_hu_context(responder);
    if (HuEvaluator::evaluate(
            players_[responder],
            hu_context
        ).legal) {
        actions.push_back({
            ActionType::Ron,
            responder,
            kakan.tile,
            kakan.actor,
        });
    }
    return actions;
}

uint32_t Arena::wait_mask(const PlayerState& player) {
    uint32_t mask = 0;
    for (uint8_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (player.waits[tile_id]) {
            mask |= uint32_t{1} << tile_id;
        }
    }
    return mask;
}

bool Arena::actions_equal(const Action& lhs, const Action& rhs) {
    return lhs.type == rhs.type &&
           lhs.actor == rhs.actor &&
           lhs.tile == rhs.tile &&
           lhs.target == rhs.target;
}

} // namespace GYMJ
