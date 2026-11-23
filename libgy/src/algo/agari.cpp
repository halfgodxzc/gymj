#include "agari.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace GYMJ {
namespace Algo {
namespace {

struct SuitEntry {
    uint8_t mentsu_count;
    bool has_pair;
};

using SuitEntries = std::vector<SuitEntry>;

std::unordered_map<uint32_t, SuitEntries> suit_table;
std::once_flag init_flag;

int tile_count(const uint8_t tehai[TILE_TYPE_COUNT]) {
    int count = 0;
    for (std::size_t i = 0; i < TILE_TYPE_COUNT; ++i) {
        count += tehai[i];
    }
    return count;
}

uint32_t encode_suit(const uint8_t suit_tehai[TILES_PER_SUIT]) {
    uint32_t key = 0;
    for (std::size_t i = 0; i < TILES_PER_SUIT; ++i) {
        key |= static_cast<uint32_t>(suit_tehai[i]) << (i * 3);
    }
    return key;
}

bool entry_exists(const SuitEntries& entries, uint8_t mentsu_count, bool has_pair) {
    return std::any_of(entries.begin(), entries.end(), [=](const SuitEntry& entry) {
        return entry.mentsu_count == mentsu_count && entry.has_pair == has_pair;
    });
}

void add_entry(SuitEntries& entries, uint8_t mentsu_count, bool has_pair) {
    if (!entry_exists(entries, mentsu_count, has_pair)) {
        entries.push_back({mentsu_count, has_pair});
    }
}

void analyze_suit(
    uint8_t suit_tehai[TILES_PER_SUIT],
    uint8_t mentsu_count,
    bool has_pair,
    SuitEntries& entries
) {
    int first = -1;
    for (std::size_t i = 0; i < TILES_PER_SUIT; ++i) {
        if (suit_tehai[i] > 0) {
            first = static_cast<int>(i);
            break;
        }
    }
    if (first < 0) {
        add_entry(entries, mentsu_count, has_pair);
        return;
    }

    if (!has_pair && suit_tehai[first] >= 2) {
        suit_tehai[first] -= 2;
        analyze_suit(suit_tehai, mentsu_count, true, entries);
        suit_tehai[first] += 2;
    }

    if (suit_tehai[first] >= 3) {
        suit_tehai[first] -= 3;
        analyze_suit(suit_tehai, mentsu_count + 1, has_pair, entries);
        suit_tehai[first] += 3;
    }

    if (first <= 6 && suit_tehai[first + 1] > 0 && suit_tehai[first + 2] > 0) {
        suit_tehai[first]--;
        suit_tehai[first + 1]--;
        suit_tehai[first + 2]--;
        analyze_suit(suit_tehai, mentsu_count + 1, has_pair, entries);
        suit_tehai[first]++;
        suit_tehai[first + 1]++;
        suit_tehai[first + 2]++;
    }
}

void enumerate_suits(int pos, uint8_t suit_tehai[TILES_PER_SUIT]) {
    if (pos == static_cast<int>(TILES_PER_SUIT)) {
        SuitEntries entries;
        uint8_t temp[TILES_PER_SUIT];
        std::copy(suit_tehai, suit_tehai + TILES_PER_SUIT, temp);
        analyze_suit(temp, 0, false, entries);

        if (!entries.empty()) {
            suit_table.emplace(encode_suit(suit_tehai), std::move(entries));
        }
        return;
    }

    for (uint8_t count = 0; count <= 4; ++count) {
        suit_tehai[pos] = count;
        enumerate_suits(pos + 1, suit_tehai);
    }
}

void initialize_table() {
    uint8_t suit_tehai[TILES_PER_SUIT] = {0};
    enumerate_suits(0, suit_tehai);
}

} // namespace

void ensure_agari_table_initialized() {
    std::call_once(init_flag, initialize_table);
}

bool is_standard_hu(
    const uint8_t tehai[TILE_TYPE_COUNT],
    int meld_count
) {
    ensure_agari_table_initialized();

    if (meld_count < 0 || meld_count > COMPLETE_HAND_MELD_COUNT) return false;
    if (tile_count(tehai) !=
        COMPLETE_HAND_TILE_COUNT - meld_count * TILES_PER_MELD) {
        return false;
    }

    uint8_t wan[TILES_PER_SUIT];
    uint8_t tong[TILES_PER_SUIT];
    uint8_t tiao[TILES_PER_SUIT];
    std::copy(tehai, tehai + TILES_PER_SUIT, wan);
    std::copy(
        tehai + TILES_PER_SUIT,
        tehai + 2 * TILES_PER_SUIT,
        tong
    );
    std::copy(
        tehai + 2 * TILES_PER_SUIT,
        tehai + TILE_TYPE_COUNT,
        tiao
    );

    auto wan_it = suit_table.find(encode_suit(wan));
    auto tong_it = suit_table.find(encode_suit(tong));
    auto tiao_it = suit_table.find(encode_suit(tiao));
    if (wan_it == suit_table.end() || tong_it == suit_table.end() || tiao_it == suit_table.end()) {
        return false;
    }

    for (const auto& wan_entry : wan_it->second) {
        for (const auto& tong_entry : tong_it->second) {
            for (const auto& tiao_entry : tiao_it->second) {
                int pair_count = (wan_entry.has_pair ? 1 : 0) +
                                 (tong_entry.has_pair ? 1 : 0) +
                                 (tiao_entry.has_pair ? 1 : 0);
                if (pair_count != 1) continue;

                int total_mentsu = wan_entry.mentsu_count +
                                   tong_entry.mentsu_count +
                                   tiao_entry.mentsu_count +
                                   meld_count;
                if (total_mentsu == COMPLETE_HAND_MELD_COUNT) return true;
            }
        }
    }

    return false;
}

bool is_seven_pairs(const uint8_t tehai[TILE_TYPE_COUNT]) {
    if (tile_count(tehai) != COMPLETE_HAND_TILE_COUNT) return false;

    int pairs = 0;
    for (std::size_t i = 0; i < TILE_TYPE_COUNT; ++i) {
        if (tehai[i] == 2) {
            pairs++;
        } else if (tehai[i] == 4) {
            pairs += 2;
        } else if (tehai[i] != 0) {
            return false;
        }
    }

    return pairs == 7;
}

bool is_flush(
    const uint8_t tehai[TILE_TYPE_COUNT],
    const std::vector<Meld>& melds
) {
    bool has_suit[3] = {false, false, false};
    for (std::size_t i = 0; i < TILE_TYPE_COUNT; ++i) {
        if (tehai[i] > 0) {
            has_suit[i / TILES_PER_SUIT] = true;
        }
    }
    for (const auto& meld : melds) {
        has_suit[meld.tile.suit()] = true;
    }

    int suit_count = 0;
    for (bool has_tile : has_suit) {
        if (has_tile) suit_count++;
    }
    return suit_count == 1;
}

bool is_all_triplets(
    const uint8_t tehai[TILE_TYPE_COUNT],
    const std::vector<Meld>& melds
) {
    int meld_count = static_cast<int>(melds.size());
    if (meld_count > COMPLETE_HAND_MELD_COUNT) return false;
    if (tile_count(tehai) !=
        COMPLETE_HAND_TILE_COUNT - meld_count * TILES_PER_MELD) {
        return false;
    }

    int pair_count = 0;
    int triplet_count = 0;
    for (std::size_t i = 0; i < TILE_TYPE_COUNT; ++i) {
        if (tehai[i] == 0) continue;
        if (tehai[i] == 2) {
            pair_count++;
        } else if (tehai[i] == 3) {
            triplet_count++;
        } else {
            return false;
        }
    }

    return pair_count == 1 &&
           triplet_count == COMPLETE_HAND_MELD_COUNT - meld_count;
}

std::vector<Tile> get_waits(
    const uint8_t tehai[TILE_TYPE_COUNT],
    int meld_count
) {
    std::vector<Tile> waits;

    for (uint8_t i = 0; i < TILE_TYPE_COUNT; ++i) {
        if (tehai[i] >= MAX_TILE_COPIES) continue;

        uint8_t temp[TILE_TYPE_COUNT];
        std::copy(tehai, tehai + TILE_TYPE_COUNT, temp);
        temp[i]++;

        if (is_standard_hu(temp, meld_count) || (meld_count == 0 && is_seven_pairs(temp))) {
            waits.push_back(Tile{i});
        }
    }

    return waits;
}

} // namespace Algo
} // namespace GYMJ
