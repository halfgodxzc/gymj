#include "wall.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

void test_standard_wall_contents() {
    GYMJ::Wall wall(123456);
    std::array<int, 27> counts{};

    while (!wall.empty()) {
        counts[wall.draw().id]++;
    }

    assert(wall.remaining() == 0);
    for (int count : counts) {
        assert(count == 4);
    }
}

void test_seed_is_reproducible() {
    GYMJ::Wall first(987654321);
    GYMJ::Wall second(987654321);

    while (!first.empty()) {
        const GYMJ::Tile first_tile = first.draw();
        const GYMJ::Tile second_tile = second.draw();
        assert(first_tile == second_tile);
    }
    assert(second.empty());
}

void test_fixed_wall_and_replacement_draw() {
    const std::vector<GYMJ::Tile> tiles = {
        GYMJ::Tile{1},
        GYMJ::Tile{2},
        GYMJ::Tile{3},
        GYMJ::Tile{4},
    };
    GYMJ::Wall wall = GYMJ::Wall::from_tiles(tiles, 42);

    assert(wall.seed() == 42);
    assert(wall.remaining_tiles() == tiles);
    assert(wall.draw().id == 1);
    assert((wall.remaining_tiles() == std::vector<GYMJ::Tile>{
        GYMJ::Tile{2}, GYMJ::Tile{3}, GYMJ::Tile{4}
    }));
    assert(wall.draw_replacement().id == 4);
    assert(wall.remaining() == 2);
    assert((wall.remaining_tiles() == std::vector<GYMJ::Tile>{
        GYMJ::Tile{2}, GYMJ::Tile{3}
    }));
}

void test_empty_wall_rejects_draw() {
    GYMJ::Wall wall = GYMJ::Wall::from_tiles({});
    bool threw = false;
    try {
        wall.draw();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

void test_chicken_indicator_uses_the_front_of_the_wall() {
    GYMJ::Wall wall = GYMJ::Wall::from_tiles({
        GYMJ::Tile{3},
        GYMJ::Tile{4},
    });

    assert(wall.reveal_chicken_indicator() == GYMJ::Tile{3});
    assert(wall.remaining() == 1);
    assert((wall.remaining_tiles() == std::vector<GYMJ::Tile>{GYMJ::Tile{4}}));
    assert(wall.draw() == GYMJ::Tile{4});
    assert(wall.remaining_tiles().empty());
}

int main() {
    test_standard_wall_contents();
    test_seed_is_reproducible();
    test_fixed_wall_and_replacement_draw();
    test_empty_wall_rejects_draw();
    test_chicken_indicator_uses_the_front_of_the_wall();
    return 0;
}
