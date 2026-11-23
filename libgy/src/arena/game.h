#pragma once

#include "arena.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace GYMJ {

enum class GamePhase : uint8_t {
    NotStarted,
    PlayingHand,
    BetweenHands,
    Finished
};

struct GameConfig {
    uint64_t seed = 0;
    uint8_t initial_dealer = 0;
    uint32_t max_hands = 4;
    std::array<int, PLAYER_COUNT> initial_scores{};
};

struct HandRecord {
    uint32_t hand_index = 0;
    uint8_t dealer = INVALID_PLAYER_ID;
    uint32_t dealer_continuation = 0;
    RoundResult round_result;
    RoundSettlementResult settlement;
    std::array<int, PLAYER_COUNT> scores_after{};
    uint8_t next_dealer = INVALID_PLAYER_ID;
    bool dealer_continues = false;
};

class DealerRotation {
public:
    static uint8_t next_dealer(
        uint8_t current_dealer,
        const RoundResult& result,
        const std::array<bool, PLAYER_COUNT>& is_jiao
    );
};

class Game {
public:
    explicit Game(const GameConfig& config);
    Game(const GameConfig& config, std::vector<Wall> scripted_walls);

    void start();
    void finish_hand();
    void start_next_hand();

    DecisionContext current_decision() const;
    void submit_action(const Action& action);

    GamePhase phase() const noexcept;
    bool is_finished() const noexcept;
    uint32_t hand_index() const noexcept;
    uint8_t dealer() const noexcept;
    uint32_t dealer_continuation() const noexcept;
    const std::array<int, PLAYER_COUNT>& scores() const noexcept;
    const std::vector<HandRecord>& history() const noexcept;
    Arena& arena();
    const Arena& arena() const;

private:
    uint64_t seed_for_hand(uint32_t hand_index) const noexcept;
    void create_and_start_hand();

    GameConfig config_;
    GamePhase phase_ = GamePhase::NotStarted;
    uint32_t hand_index_ = 0;
    uint8_t dealer_ = 0;
    uint32_t dealer_continuation_ = 0;
    std::array<int, PLAYER_COUNT> scores_{};
    std::vector<HandRecord> history_;
    std::vector<Wall> scripted_walls_;
    std::unique_ptr<Arena> arena_;
};

} // namespace GYMJ
