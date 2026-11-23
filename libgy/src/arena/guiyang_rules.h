#pragma once

#include "guiyang_mahjong.h"

namespace GYMJ {

class GuiyangRules {
public:
    static bool can_ron(
        const PlayerState& player,
        Tile hu_tile,
        bool bypass_passport = false
    );
    static int response_priority(ActionType type);
};

} // namespace GYMJ
