#include "agari.h"
#include "guiyang_mahjong.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

void add_tiles(uint8_t hand[27], std::initializer_list<int> tiles) {
    for (int tile : tiles) {
        hand[tile]++;
    }
}

} // namespace

void test_standard_all_triplets_hu() {
    using namespace GYMJ;

    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4,
    });
    assert(Algo::is_standard_hu(hand, 0));
    assert(Algo::is_all_triplets(hand, {}));
}

void test_standard_sequence_hu() {
    using namespace GYMJ;

    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        3, 4, 5,
        6, 6,
    });
    assert(Algo::is_standard_hu(hand, 0));
}

void test_seven_pairs() {
    using namespace GYMJ;

    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6,
    });
    assert(Algo::is_seven_pairs(hand));
}

void test_open_all_triplets_and_flush() {
    using namespace GYMJ;

    uint8_t hand[27] = {0};
    std::vector<Meld> melds = {
        {MeldType::Pon, Tile{0}, 1},
    };
    add_tiles(hand, {
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4,
    });
    assert(Algo::is_all_triplets(hand, melds));
    assert(Algo::is_flush(hand, melds));
}

void test_not_standard_hu() {
    using namespace GYMJ;

    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 1, 2,
        9, 10, 11,
        18, 19, 20,
        3, 4, 5,
        6, 7,
    });
    assert(!Algo::is_standard_hu(hand, 0));
}

int main() {
    test_standard_all_triplets_hu();
    test_standard_sequence_hu();
    test_seven_pairs();
    test_open_all_triplets_and_flush();
    test_not_standard_hu();

    return 0;
}
