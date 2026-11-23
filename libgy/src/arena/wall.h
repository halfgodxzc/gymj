#pragma once

#include "guiyang_mahjong.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace GYMJ {

class Wall {
public:
    explicit Wall(uint64_t seed);

    static Wall from_tiles(const std::vector<Tile>& tiles, uint64_t seed = 0);

    Tile draw();
    Tile draw_replacement();
    Tile reveal_chicken_indicator();

    size_t remaining() const;
    std::vector<Tile> remaining_tiles() const;
    uint64_t seed() const;
    bool empty() const;

private:
    Wall(std::vector<Tile> tiles, uint64_t seed);

    uint64_t seed_;
    std::vector<Tile> tiles_;
    size_t front_;
    size_t back_;
};

} // namespace GYMJ
