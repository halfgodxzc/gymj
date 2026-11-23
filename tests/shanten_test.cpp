#include "shanten.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace {

void add_tiles(uint8_t hand[27], std::initializer_list<int> tiles) {
    for (int tile : tiles) {
        hand[tile]++;
    }
}

bool has_pattern(
    const std::vector<GYMJ::Algo::SuitPattern>& patterns,
    uint8_t melds,
    uint8_t taatsu,
    uint8_t pairs
) {
    return std::any_of(patterns.begin(), patterns.end(), [&](const auto& pattern) {
        return pattern.melds == melds &&
               pattern.taatsu == taatsu &&
               pattern.pairs == pairs;
    });
}

} // namespace

void test_112233_keeps_global_choices() {
    uint8_t suit[9] = {2, 2, 2, 0, 0, 0, 0, 0, 0};
    const auto& patterns = GYMJ::Algo::get_suit_patterns(suit);

    assert(has_pattern(patterns, 2, 0, 0));
    assert(has_pattern(patterns, 1, 1, 0));
    assert(has_pattern(patterns, 0, 3, 0));
    assert(has_pattern(patterns, 0, 2, 1));
}

void test_normal_complete_and_tenpai() {
    uint8_t complete[27] = {0};
    add_tiles(complete, {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26, 26,
    });
    assert(GYMJ::Algo::calculate_normal_shanten(complete, 0) == -1);

    complete[26]--;
    assert(GYMJ::Algo::calculate_normal_shanten(complete, 0) == 0);
}

void test_open_normal_shanten() {
    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 1, 2,
        9, 10, 11,
        18, 19, 20,
        26,
    });
    assert(GYMJ::Algo::calculate_normal_shanten(hand, 1) == 0);
}

void test_seven_pairs_shanten() {
    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6,
    });
    assert(GYMJ::Algo::calculate_seven_pairs_shanten(hand, 0) == 0);
    hand[6]++;
    assert(GYMJ::Algo::calculate_seven_pairs_shanten(hand, 0) == -1);
    assert(GYMJ::Algo::calculate_seven_pairs_shanten(hand, 1) == 8);
}

void test_all_triplets_shanten() {
    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 0, 0,
        1, 1, 1,
        9, 9, 9,
        18, 18, 18,
        26,
    });
    assert(GYMJ::Algo::calculate_all_triplets_shanten(hand, 0) == 0);
    hand[26]++;
    assert(GYMJ::Algo::calculate_all_triplets_shanten(hand, 0) == -1);
}

void test_best_shanten_chooses_special_shape() {
    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 0,
        2, 2,
        4, 4,
        6, 6,
        9, 9,
        11, 11,
        18,
    });

    const auto result = GYMJ::Algo::calculate_shanten(hand, 0);
    assert(result.seven_pairs == 0);
    assert(result.best == 0);
}

void test_fourth_copy_cannot_be_a_fifth_copy_wait() {
    uint8_t hand[27] = {0};
    add_tiles(hand, {
        2, 3, 4,
        14, 14, 14, 14,
        19, 20, 21,
        24, 24, 24,
    });

    const auto result = GYMJ::Algo::calculate_shanten(hand, 0);
    assert(result.normal == 1);
    assert(result.best == 1);
}

void test_drawn_hand_can_be_tenpai_after_discard() {
    uint8_t hand[27] = {0};
    add_tiles(hand, {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        17,
        18, 19, 20,
        26,
    });

    assert(GYMJ::Algo::calculate_normal_shanten(hand, 0) == 0);
    assert(GYMJ::Algo::calculate_shanten(hand, 0).best == 0);
}

void test_logging_switch() {
    GYMJ::Algo::set_shanten_logging(true);
    assert(GYMJ::Algo::is_shanten_logging_enabled());
    GYMJ::Algo::set_shanten_logging(false);
    assert(!GYMJ::Algo::is_shanten_logging_enabled());
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--log") == 0) {
        GYMJ::Algo::set_shanten_logging(true);
    }

    test_112233_keeps_global_choices();
    test_normal_complete_and_tenpai();
    test_open_normal_shanten();
    test_seven_pairs_shanten();
    test_all_triplets_shanten();
    test_best_shanten_chooses_special_shape();
    test_fourth_copy_cannot_be_a_fifth_copy_wait();
    test_drawn_hand_can_be_tenpai_after_discard();
    if (argc == 1) {
        test_logging_switch();
    }
    return 0;
}
