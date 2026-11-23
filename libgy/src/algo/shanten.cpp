#include "shanten.h"

#include "agari.h"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace GYMJ {
namespace Algo {
namespace {

using SuitPatterns = std::vector<SuitPattern>;

std::unordered_map<uint32_t, SuitPatterns> suit_table;
std::mutex suit_table_mutex;
std::atomic<bool> logging_enabled{false};
std::mutex log_mutex;

struct NormalEvaluation {
    int8_t shanten;
    SuitPattern wan;
    SuitPattern tong;
    SuitPattern tiao;
};

template <typename IsComplete>
bool has_legal_completion(
    const uint8_t tehai[TILE_TYPE_COUNT],
    IsComplete is_complete
) {
    uint8_t completed[TILE_TYPE_COUNT];
    std::copy(tehai, tehai + TILE_TYPE_COUNT, completed);

    for (std::size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (completed[tile_id] >= MAX_TILE_COPIES) continue;

        completed[tile_id]++;
        const bool complete = is_complete(completed);
        completed[tile_id]--;
        if (complete) return true;
    }
    return false;
}

int tile_count(const uint8_t tehai[TILE_TYPE_COUNT]) {
    int count = 0;
    for (std::size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        count += tehai[tile_id];
    }
    return count;
}

bool is_all_triplets_complete(
    const uint8_t tehai[TILE_TYPE_COUNT],
    int meld_count
) {
    if (tile_count(tehai) !=
        COMPLETE_HAND_TILE_COUNT - meld_count * TILES_PER_MELD) {
        return false;
    }

    int pairs = 0;
    int concealed_melds = 0;
    for (std::size_t tile_id = 0; tile_id < TILE_TYPE_COUNT; ++tile_id) {
        if (tehai[tile_id] == 0) continue;
        if (tehai[tile_id] == 2) {
            pairs++;
        } else if (tehai[tile_id] == 3) {
            concealed_melds++;
        } else {
            return false;
        }
    }
    return pairs == 1 &&
           concealed_melds + meld_count == COMPLETE_HAND_MELD_COUNT;
}

template <typename IsComplete>
int8_t correct_impossible_tenpai(
    int8_t shanten,
    const uint8_t tehai[TILE_TYPE_COUNT],
    int meld_count,
    IsComplete is_complete
) {
    const int waiting_tile_count =
        COMPLETE_HAND_TILE_COUNT - 1 - meld_count * TILES_PER_MELD;
    if (shanten != 0 || tile_count(tehai) != waiting_tile_count ||
        has_legal_completion(tehai, is_complete)) {
        return shanten;
    }
    return 1;
}

uint32_t encode_suit(const uint8_t suit_tehai[9]) {
    uint32_t key = 0;
    for (int i = 0; i < 9; ++i) {
        key |= static_cast<uint32_t>(suit_tehai[i]) << (i * 3);
    }
    return key;
}

bool same_pattern(const SuitPattern& lhs, const SuitPattern& rhs) {
    return lhs.melds == rhs.melds &&
           lhs.taatsu == rhs.taatsu &&
           lhs.pairs == rhs.pairs;
}

bool dominates(const SuitPattern& lhs, const SuitPattern& rhs) {
    const bool no_worse = lhs.melds >= rhs.melds &&
                          lhs.taatsu >= rhs.taatsu &&
                          lhs.pairs >= rhs.pairs;
    const bool strictly_better = lhs.melds > rhs.melds ||
                                 lhs.taatsu > rhs.taatsu ||
                                 lhs.pairs > rhs.pairs;
    return no_worse && strictly_better;
}

void add_pattern(SuitPatterns& patterns, uint8_t melds, uint8_t taatsu, uint8_t pairs) {
    if (melds > 4 || taatsu > 4 || pairs > 1 || melds + taatsu > 4) return;

    SuitPattern candidate{melds, taatsu, pairs};
    if (std::any_of(patterns.begin(), patterns.end(), [&](const SuitPattern& pattern) {
            return same_pattern(pattern, candidate);
        })) {
        return;
    }
    patterns.push_back(candidate);
}

void analyze_suit(
    uint8_t tiles[9],
    uint8_t melds,
    uint8_t taatsu,
    uint8_t pairs,
    SuitPatterns& patterns
) {
    int first = -1;
    for (int i = 0; i < 9; ++i) {
        if (tiles[i] > 0) {
            first = i;
            break;
        }
    }

    if (first < 0) {
        add_pattern(patterns, melds, taatsu, pairs);
        return;
    }

    // A tile may remain unused in this local decomposition.
    tiles[first]--;
    analyze_suit(tiles, melds, taatsu, pairs, patterns);
    tiles[first]++;

    if (tiles[first] >= 3) {
        tiles[first] -= 3;
        analyze_suit(tiles, melds + 1, taatsu, pairs, patterns);
        tiles[first] += 3;
    }

    if (first <= 6 && tiles[first + 1] > 0 && tiles[first + 2] > 0) {
        tiles[first]--;
        tiles[first + 1]--;
        tiles[first + 2]--;
        analyze_suit(tiles, melds + 1, taatsu, pairs, patterns);
        tiles[first]++;
        tiles[first + 1]++;
        tiles[first + 2]++;
    }

    if (tiles[first] >= 2) {
        tiles[first] -= 2;
        analyze_suit(tiles, melds, taatsu + 1, pairs, patterns);
        if (pairs == 0) {
            analyze_suit(tiles, melds, taatsu, 1, patterns);
        }
        tiles[first] += 2;
    }

    if (first <= 7 && tiles[first + 1] > 0) {
        tiles[first]--;
        tiles[first + 1]--;
        analyze_suit(tiles, melds, taatsu + 1, pairs, patterns);
        tiles[first]++;
        tiles[first + 1]++;
    }

    if (first <= 6 && tiles[first + 2] > 0) {
        tiles[first]--;
        tiles[first + 2]--;
        analyze_suit(tiles, melds, taatsu + 1, pairs, patterns);
        tiles[first]++;
        tiles[first + 2]++;
    }
}

SuitPatterns build_patterns(const uint8_t suit_tehai[9]) {
    uint8_t temp[9];
    std::copy(suit_tehai, suit_tehai + 9, temp);

    SuitPatterns patterns;
    analyze_suit(temp, 0, 0, 0, patterns);

    SuitPatterns filtered;
    for (const auto& candidate : patterns) {
        const bool is_dominated = std::any_of(
            patterns.begin(),
            patterns.end(),
            [&](const SuitPattern& other) {
                return dominates(other, candidate);
            }
        );
        if (!is_dominated) filtered.push_back(candidate);
    }
    return filtered;
}

int8_t clamp_shanten(int value) {
    return static_cast<int8_t>(std::max(-1, std::min(8, value)));
}

int normal_shanten_for(
    const SuitPattern& wan,
    const SuitPattern& tong,
    const SuitPattern& tiao,
    int meld_count
) {
    int melds = meld_count + wan.melds + tong.melds + tiao.melds;
    melds = std::min(melds, 4);

    int taatsu = wan.taatsu + tong.taatsu + tiao.taatsu;
    int pairs = wan.pairs + tong.pairs + tiao.pairs;
    const int has_pair = pairs > 0 ? 1 : 0;

    // Extra pair candidates can serve as incomplete groups.
    taatsu += std::max(0, pairs - 1);
    taatsu = std::min(taatsu, 4 - melds);

    return 8 - 2 * melds - taatsu - has_pair;
}

NormalEvaluation evaluate_normal_shanten(const uint8_t tehai[27], int meld_count) {
    NormalEvaluation result{8, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    if (meld_count < 0 || meld_count > 4) return result;

    const auto& wan_patterns = get_suit_patterns(tehai);
    const auto& tong_patterns = get_suit_patterns(tehai + 9);
    const auto& tiao_patterns = get_suit_patterns(tehai + 18);

    int best = std::numeric_limits<int>::max();
    for (const auto& wan : wan_patterns) {
        for (const auto& tong : tong_patterns) {
            for (const auto& tiao : tiao_patterns) {
                const int candidate = normal_shanten_for(wan, tong, tiao, meld_count);
                if (candidate < best) {
                    best = candidate;
                    result.wan = wan;
                    result.tong = tong;
                    result.tiao = tiao;
                }
            }
        }
    }
    result.shanten = correct_impossible_tenpai(
        clamp_shanten(best),
        tehai,
        meld_count,
        [meld_count](const uint8_t completed[TILE_TYPE_COUNT]) {
            return is_standard_hu(completed, meld_count);
        }
    );
    return result;
}

std::string format_hand(const uint8_t tehai[27]) {
    std::ostringstream output;
    const char suits[] = {'m', 'p', 's'};
    bool first_suit = true;

    for (int suit = 0; suit < 3; ++suit) {
        std::string tiles;
        for (int rank = 0; rank < 9; ++rank) {
            for (int count = 0; count < tehai[suit * 9 + rank]; ++count) {
                tiles += static_cast<char>('1' + rank);
            }
        }
        if (tiles.empty()) continue;
        if (!first_suit) output << ' ';
        output << tiles << suits[suit];
        first_suit = false;
    }
    return first_suit ? "(empty)" : output.str();
}

void write_log(
    const uint8_t tehai[27],
    int meld_count,
    const ShantenAnalysis& analysis
) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::clog << "[Shanten] Hand: " << format_hand(tehai)
              << " | Open melds: " << meld_count << '\n';
    std::clog << "[Shanten] Normal split: "
              << "wan={melds:" << static_cast<int>(analysis.wan.melds)
              << ", taatsu:" << static_cast<int>(analysis.wan.taatsu)
              << ", pair:" << static_cast<int>(analysis.wan.pairs) << "} "
              << "tong={melds:" << static_cast<int>(analysis.tong.melds)
              << ", taatsu:" << static_cast<int>(analysis.tong.taatsu)
              << ", pair:" << static_cast<int>(analysis.tong.pairs) << "} "
              << "tiao={melds:" << static_cast<int>(analysis.tiao.melds)
              << ", taatsu:" << static_cast<int>(analysis.tiao.taatsu)
              << ", pair:" << static_cast<int>(analysis.tiao.pairs) << "}\n";
    std::clog << "[Shanten] Result: normal="
              << static_cast<int>(analysis.result.normal)
              << " seven_pairs="
              << static_cast<int>(analysis.result.seven_pairs)
              << " all_triplets="
              << static_cast<int>(analysis.result.all_triplets)
              << " best=" << static_cast<int>(analysis.result.best) << '\n';
}

} // namespace

void set_shanten_logging(bool enabled) {
    logging_enabled.store(enabled);
}

bool is_shanten_logging_enabled() {
    return logging_enabled.load();
}

const std::vector<SuitPattern>& get_suit_patterns(const uint8_t suit_tehai[9]) {
    const uint32_t key = encode_suit(suit_tehai);
    std::lock_guard<std::mutex> lock(suit_table_mutex);
    auto it = suit_table.find(key);
    if (it == suit_table.end()) {
        it = suit_table.emplace(key, build_patterns(suit_tehai)).first;
    }
    return it->second;
}

int8_t calculate_normal_shanten(const uint8_t tehai[27], int meld_count) {
    return evaluate_normal_shanten(tehai, meld_count).shanten;
}

int8_t calculate_seven_pairs_shanten(const uint8_t tehai[27], int meld_count) {
    if (meld_count != 0) return 8;

    int pair_units = 0;
    for (int i = 0; i < 27; ++i) {
        pair_units += tehai[i] / 2;
    }
    return correct_impossible_tenpai(
        clamp_shanten(6 - std::min(pair_units, 7)),
        tehai,
        meld_count,
        [](const uint8_t completed[TILE_TYPE_COUNT]) {
            return is_seven_pairs(completed);
        }
    );
}

int8_t calculate_all_triplets_shanten(const uint8_t tehai[27], int meld_count) {
    if (meld_count < 0 || meld_count > 4) return 8;

    int best = 8;
    for (int pair_tile = -1; pair_tile < 27; ++pair_tile) {
        int melds = meld_count;
        int taatsu = 0;
        int has_pair = 0;

        for (int i = 0; i < 27; ++i) {
            int count = tehai[i];
            if (i == pair_tile && count >= 2) {
                count -= 2;
                has_pair = 1;
            }
            melds += count / 3;
            if (count % 3 >= 2) taatsu++;
        }

        if (melds > 4) continue;
        taatsu = std::min(taatsu, 4 - melds);
        best = std::min(best, 8 - 2 * melds - taatsu - has_pair);
    }
    return correct_impossible_tenpai(
        clamp_shanten(best),
        tehai,
        meld_count,
        [meld_count](const uint8_t completed[TILE_TYPE_COUNT]) {
            return is_all_triplets_complete(completed, meld_count);
        }
    );
}

ShantenAnalysis analyze_shanten(const uint8_t tehai[27], int meld_count) {
    const NormalEvaluation normal = evaluate_normal_shanten(tehai, meld_count);
    ShantenAnalysis analysis{
        {
            normal.shanten,
            calculate_seven_pairs_shanten(tehai, meld_count),
            calculate_all_triplets_shanten(tehai, meld_count),
            8,
        },
        normal.wan,
        normal.tong,
        normal.tiao,
    };
    analysis.result.best = std::min({
        analysis.result.normal,
        analysis.result.seven_pairs,
        analysis.result.all_triplets,
    });
    return analysis;
}

ShantenResult calculate_shanten(const uint8_t tehai[27], int meld_count) {
    const ShantenAnalysis analysis = analyze_shanten(tehai, meld_count);
    if (logging_enabled.load()) {
        write_log(tehai, meld_count, analysis);
    }
    return analysis.result;
}

} // namespace Algo
} // namespace GYMJ
