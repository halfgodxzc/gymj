#pragma once

#include "arena_types.h"
#include "chicken.h"
#include "round_settlement.h"
#include "settlement_snapshot.h"
#include "wall.h"

#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace GYMJ {

class TextLogWriter;
namespace RL {
class DecisionFeatureEncoder;
class PrivilegedStateEncoder;
class PrivilegedPatternEncoder;
}

class Arena {
public:
    explicit Arena(const ArenaConfig& config);
    Arena(const ArenaConfig& config, Wall wall);
    ~Arena();

    void start_round();
    DecisionContext current_decision() const;
    void submit_action(const Action& action);

    RoundPhase phase() const;
    bool is_finished() const;
    uint8_t current_player() const;
    uint8_t dealer() const;
    size_t wall_remaining() const;
    Tile response_tile() const;
    uint8_t response_actor() const;

    const PlayerState& player(uint8_t id) const;
    const ReadyState& ready_state(uint8_t id) const;
    const RoundResult& result() const;
    const SettlementSnapshot& settlement_snapshot() const;
    const ChickenResult& chicken_result() const;
    const RoundSettlementResult& round_settlement() const;
    const std::vector<GameEvent>& history() const;
    const std::vector<DiscardEvent>& discard_history() const;
    std::vector<ObservedDiscardEvent> discard_history_for(uint8_t observer_id) const;

private:
    friend class SearchRoot;
    friend class RL::DecisionFeatureEncoder;
    friend class RL::PrivilegedPatternEncoder;
    friend class RL::PrivilegedStateEncoder;

    enum class DrawKind : uint8_t {
        Opening,
        Normal,
        Replacement
    };

    enum class ReadyDecisionOrigin : uint8_t {
        BeforeOpeningDraw,
        AfterFirstDiscard
    };

    struct PlayerRoundProgress {
        uint32_t passed_pon_tiles = 0;
        bool has_normal_drawn = false;
        bool has_discarded = false;
        bool opening_hand_changed = false;
        bool ready_window_closed = false;
    };

    struct PendingReadyDecision {
        uint8_t player;
        ReadyKind kind;
        ReadyDecisionOrigin origin;
        uint32_t candidate_waits;
    };

    void deal_initial_hands();
    void advance_initial_ready_declarations();
    void prepare_normal_turn(uint8_t player_id);
    void draw_for_current_player(DrawKind kind);
    void begin_ready_decision(
        uint8_t player_id,
        ReadyKind kind,
        ReadyDecisionOrigin origin
    );
    void resolve_ready_decision(const Action& action);
    void begin_discard_responses(const Action& discard);
    void begin_kakan_responses(const Action& kakan);
    void resolve_discard_responses();
    void resolve_kakan_responses();
    void reset_responses(uint8_t actor);
    void apply_to_players(const Action& action);
    void record_event(
        const Action& action,
        RoundPhase before,
        RoundPhase after,
        bool applied
    );
    void broadcast(const Action& action, RoundPhase before, RoundPhase after);
    void update_discard_history(const Action& action);
    void prepare_hu_settlement(
        const HuTileClaim& hu_tile_claim,
        const std::vector<HuContext>& hu_contexts
    );
    ChickenContext build_chicken_context() const;
    void finish_exhaustive_draw();
    void log_round_start();
    void log_decision(
        const DecisionContext& context,
        const Action& selected_action
    );
    void log_round_end();

    bool should_offer_baoting_after(const Action& discard) const;
    bool can_current_player_tsumo() const;
    HuContext current_tsumo_hu_context() const;
    HuContext discard_hu_context(uint8_t responder) const;
    HuContext kakan_hu_context(uint8_t responder) const;
    std::array<bool, PLAYER_COUNT> jiao_states() const;
    std::vector<Action> legal_actions_for_reported_turn() const;
    std::vector<Action> legal_actions_for_discard_response(
        uint8_t responder
    ) const;
    std::vector<Action> legal_actions_for_kakan_response(
        uint8_t responder
    ) const;
    static uint32_t wait_mask(const PlayerState& player);
    static bool actions_equal(const Action& lhs, const Action& rhs);

    ArenaConfig config_;
    Wall wall_;
    std::array<PlayerState, 4> players_;

    uint8_t current_player_;
    RoundPhase phase_;
    RoundResult result_;
    std::optional<SettlementSnapshot> settlement_snapshot_;
    std::optional<ChickenResult> chicken_result_;
    std::optional<RoundSettlementResult> round_settlement_;
    std::vector<GameEvent> history_;
    std::vector<DiscardEvent> discard_history_;
    std::array<PlayerRoundProgress, 4> progress_{};
    std::array<ReadyState, 4> ready_states_{};
    std::optional<PendingReadyDecision> pending_ready_;
    size_t initial_ready_index_ = 0;

    Tile last_drawn_;
    DrawKind last_draw_kind_;
    bool must_discard_;
    std::optional<uint8_t> hot_discard_player_;
    bool pending_discard_is_hot_ = false;

    Action pending_discard_;
    std::optional<Action> pending_kakan_;
    std::array<uint8_t, 3> response_order_;
    size_t response_index_;
    std::array<std::optional<Action>, 4> responses_;
    std::unique_ptr<TextLogWriter> log_;
    uint32_t decision_sequence_ = 0;
    bool round_end_logged_ = false;
};

} // namespace GYMJ
