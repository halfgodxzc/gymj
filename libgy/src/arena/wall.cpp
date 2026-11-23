#include "wall.h"

#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace GYMJ {
namespace {

uint64_t bounded_random(std::mt19937_64& engine, uint64_t bound) {
    const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
    while (true) {
        const uint64_t value = engine();
        if (value >= threshold) return value % bound;
    }
}

std::vector<Tile> make_shuffled_tiles(uint64_t seed) {
    std::vector<Tile> tiles;
    tiles.reserve(108);
    for (uint8_t id = 0; id < 27; ++id) {
        for (int copy = 0; copy < 4; ++copy) {
            tiles.push_back(Tile{id});
        }
    }

    std::mt19937_64 engine(seed);
    for (size_t i = tiles.size(); i > 1; --i) {
        const size_t other = static_cast<size_t>(bounded_random(engine, i));
        std::swap(tiles[i - 1], tiles[other]);
    }
    return tiles;
}

} // namespace

Wall::Wall(uint64_t seed)
    : Wall(make_shuffled_tiles(seed), seed) {}

Wall::Wall(std::vector<Tile> tiles, uint64_t seed)
    : seed_(seed),
      tiles_(std::move(tiles)),
      front_(0),
      back_(tiles_.size()) {}

Wall Wall::from_tiles(const std::vector<Tile>& tiles, uint64_t seed) {
    return Wall(tiles, seed);
}

Tile Wall::draw() {
    if (empty()) throw std::out_of_range("cannot draw from an empty wall");
    return tiles_[front_++];
}

Tile Wall::draw_replacement() {
    if (empty()) throw std::out_of_range("cannot draw from an empty wall");
    return tiles_[--back_];
}

Tile Wall::reveal_chicken_indicator() {
    if (empty()) {
        throw std::out_of_range(
            "cannot reveal a chicken indicator from an empty wall"
        );
    }
    return tiles_[front_++];
}

size_t Wall::remaining() const {
    return back_ - front_;
}

std::vector<Tile> Wall::remaining_tiles() const {
    return {tiles_.cbegin() + static_cast<std::ptrdiff_t>(front_),
            tiles_.cbegin() + static_cast<std::ptrdiff_t>(back_)};
}

uint64_t Wall::seed() const {
    return seed_;
}

bool Wall::empty() const {
    return front_ >= back_;
}

} // namespace GYMJ
