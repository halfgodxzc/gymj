#include "game.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace GYMJ {
namespace {

void validate_player(uint8_t player, const char* field_name) {
    if (player >= PLAYER_COUNT) {
        throw std::invalid_argument(
            std::string(field_name) + " is out of range"
        );
    }
}

} // namespace

uint8_t DealerRotation::next_dealer(
    uint8_t current_dealer,
    const RoundResult& result,
    const std::array<bool, PLAYER_COUNT>& is_jiao
) {
    validate_player(current_dealer, "current dealer");

    if (result.reason == RoundEndReason::ExhaustiveDraw) {
        const int jiao_count = static_cast<int>(std::count(
            is_jiao.begin(),
            is_jiao.end(),
            true
        ));
        if (jiao_count == 0 || jiao_count == PLAYER_COUNT ||
            is_jiao[current_dealer]) {
            return current_dealer;
        }
        return static_cast<uint8_t>((current_dealer + 1) % PLAYER_COUNT);
    }

    if (result.reason == RoundEndReason::None || result.hu_players.empty()) {
        throw std::invalid_argument("finished result has no hu player");
    }
    for (uint8_t hu_player : result.hu_players) {
        validate_player(hu_player, "hu player");
    }

    if (result.hu_players.size() == 1) {
        return result.hu_players.front();
    }
    if (result.hu_players.size() < PLAYER_COUNT) {
        validate_player(result.fangpao_player, "multi-ron fangpao player");
        return result.fangpao_player;
    }
    throw std::invalid_argument("round has too many hu players");
}

Game::Game(const GameConfig& config)
    : Game(config, {}) {}

Game::Game(const GameConfig& config, std::vector<Wall> scripted_walls)
    : config_(config),
      dealer_(config.initial_dealer),
      scores_(config.initial_scores),
      scripted_walls_(std::move(scripted_walls)) {
    validate_player(config_.initial_dealer, "initial dealer");
    if (config_.max_hands == 0) {
        throw std::invalid_argument("game must contain at least one hand");
    }
    if (!scripted_walls_.empty() &&
        scripted_walls_.size() != config_.max_hands) {
        throw std::invalid_argument(
            "scripted wall count must match max hands"
        );
    }
}

void Game::start() {
    if (phase_ != GamePhase::NotStarted) {
        throw std::logic_error("game has already started");
    }
    create_and_start_hand();
}

void Game::finish_hand() {
    if (phase_ != GamePhase::PlayingHand || !arena_) {
        throw std::logic_error("game has no active hand to finish");
    }
    if (!arena_->is_finished()) {
        throw std::logic_error("active hand is not finished");
    }

    std::array<bool, PLAYER_COUNT> is_jiao{};
    for (uint8_t player = 0; player < PLAYER_COUNT; ++player) {
        is_jiao[player] = arena_->player(player).shanten <= 0;
    }

    const uint8_t old_dealer = dealer_;
    const uint32_t old_continuation = dealer_continuation_;
    const uint8_t next_dealer = DealerRotation::next_dealer(
        old_dealer,
        arena_->result(),
        is_jiao
    );
    const bool continues = next_dealer == old_dealer;

    const auto& settlement = arena_->round_settlement();
    const auto& deltas = settlement.ledger.score_deltas();
    for (uint8_t player = 0; player < PLAYER_COUNT; ++player) {
        scores_[player] += deltas[player];
    }

    history_.push_back({
        hand_index_,
        old_dealer,
        old_continuation,
        arena_->result(),
        settlement,
        scores_,
        next_dealer,
        continues,
    });

    dealer_ = next_dealer;
    dealer_continuation_ = continues ? old_continuation + 1 : 0;
    hand_index_++;
    phase_ = hand_index_ >= config_.max_hands
        ? GamePhase::Finished
        : GamePhase::BetweenHands;
}

void Game::start_next_hand() {
    if (phase_ != GamePhase::BetweenHands) {
        throw std::logic_error("game is not between hands");
    }
    create_and_start_hand();
}

DecisionContext Game::current_decision() const {
    return arena().current_decision();
}

void Game::submit_action(const Action& action) {
    arena().submit_action(action);
}

GamePhase Game::phase() const noexcept {
    return phase_;
}

bool Game::is_finished() const noexcept {
    return phase_ == GamePhase::Finished;
}

uint32_t Game::hand_index() const noexcept {
    return hand_index_;
}

uint8_t Game::dealer() const noexcept {
    return dealer_;
}

uint32_t Game::dealer_continuation() const noexcept {
    return dealer_continuation_;
}

const std::array<int, PLAYER_COUNT>& Game::scores() const noexcept {
    return scores_;
}

const std::vector<HandRecord>& Game::history() const noexcept {
    return history_;
}

Arena& Game::arena() {
    if (!arena_) throw std::logic_error("game has no arena");
    return *arena_;
}

const Arena& Game::arena() const {
    if (!arena_) throw std::logic_error("game has no arena");
    return *arena_;
}

uint64_t Game::seed_for_hand(uint32_t hand_index) const noexcept {
    constexpr uint64_t SEED_STEP = 0x9e3779b97f4a7c15ULL;
    return config_.seed + SEED_STEP * hand_index;
}

void Game::create_and_start_hand() {
    const ArenaConfig arena_config{
        seed_for_hand(hand_index_),
        dealer_,
    };
    if (scripted_walls_.empty()) {
        arena_ = std::make_unique<Arena>(arena_config);
    } else {
        arena_ = std::make_unique<Arena>(
            arena_config,
            std::move(scripted_walls_[hand_index_])
        );
    }
    arena_->start_round();
    phase_ = GamePhase::PlayingHand;
}

} // namespace GYMJ
