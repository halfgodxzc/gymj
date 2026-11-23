#pragma once

#include "guiyang_mahjong.h"
#include "guiyang_scoring.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace GYMJ {

enum class HuMethod : uint8_t {
    Tsumo,
    Ron,
    QiangGang
};

struct HuContext {
    HuMethod method = HuMethod::Tsumo;
    Tile hu_tile{255};
    ReadyKind hu_player_ready_kind = ReadyKind::None;
    ReadyKind source_ready_kind = ReadyKind::None;
    bool bypass_ron_passport = false;
    HuTiming timing = HuTiming::Normal;
    SpecialHu special = SpecialHu::None;
};

struct HuCandidate {
    HuFeatures features;
    HuScore score;
};

struct HuEvaluation {
    static constexpr std::size_t NO_CANDIDATE =
        std::numeric_limits<std::size_t>::max();

    bool shape_complete = false;
    bool legal = false;
    std::vector<HuCandidate> candidates;
    std::size_t best_candidate_index = NO_CANDIDATE;

    const HuCandidate* best_candidate() const noexcept;
};

class HuEvaluator {
public:
    static HuEvaluation evaluate(
        const PlayerState& player,
        const HuContext& context
    );
};

} // namespace GYMJ
