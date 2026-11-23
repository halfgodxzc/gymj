#pragma once

#include "guiyang_mahjong.h"
#include "guiyang_scoring.h"

#include <cstdint>
#include <string>
#include <vector>

namespace GYMJ {

enum class RoundPhase : uint8_t {
    NotStarted,
    AwaitTurnAction,
    AwaitReadyDeclaration,
    AwaitDiscardResponses,
    AwaitKakanResponses,
    ReplacementDraw,
    Finished
};

struct ReadyState {
    ReadyKind kind = ReadyKind::None;
    uint32_t locked_waits = 0;

    bool declared() const {
        return kind != ReadyKind::None;
    }
};

enum class RoundEndReason : uint8_t {
    None,
    Tsumo,
    Ron,
    QiangGang,
    ExhaustiveDraw
};

struct ArenaConfig {
    uint64_t seed = 0;
    uint8_t dealer = 0;
    std::string log_path;
};

struct DecisionContext {
    RoundPhase phase;
    uint8_t player;
    std::vector<Action> legal_actions;
};

struct RoundResult {
    RoundEndReason reason = RoundEndReason::None;
    std::vector<uint8_t> hu_players;
    uint8_t fangpao_player = 255;
    Tile hu_tile{255};
    bool is_hot_cannon = false;
};

struct GameEvent {
    uint32_t sequence;
    Action action;
    RoundPhase phase_before;
    RoundPhase phase_after;
    bool applied;
};

struct DiscardEvent {
    Tile tile;
    uint8_t actor;
    bool was_called;
    uint8_t called_by = 255;
};

struct ObservedDiscardEvent {
    Tile tile;
    uint8_t relative_actor;
    bool was_called;
};

} // namespace GYMJ
