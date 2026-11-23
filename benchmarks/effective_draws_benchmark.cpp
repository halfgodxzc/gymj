#include "action_feature_encoder.h"
#include "effective_draws.h"
#include "shanten.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <iostream>

namespace {

using TileCounts = std::array<uint8_t, GYMJ::TILE_TYPE_COUNT>;
constexpr uint8_t EXTRA_TILE_ID = 13;

TileCounts make_counts(std::initializer_list<uint8_t> tiles) {
    TileCounts counts{};
    for (uint8_t tile_id : tiles) ++counts[tile_id];
    return counts;
}

template <typename Function>
double microseconds_per_call(int iterations, Function function) {
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        function();
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    return std::chrono::duration<double, std::micro>(elapsed).count() /
           iterations;
}

} // namespace

int main(int argc, char* argv[]) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 10000;
    if (iterations <= 0) return 1;

    const TileCounts hand = make_counts({
        0, 1, 2,
        6, 6, 7, 8,
        9, 10, 11,
        18, 19, 20,
    });
    const TileCounts visible = hand;
    GYMJ::RL::CompactObservation observation;
    observation.hand_counts = hand;
    observation.visible_counts = visible;
    ++observation.hand_counts[EXTRA_TILE_ID];
    ++observation.visible_counts[EXTRA_TILE_ID];

    std::array<uint64_t, 2> legal_mask{};
    int legal_discard_kinds = 0;
    for (size_t tile_id = 0; tile_id < GYMJ::TILE_TYPE_COUNT; ++tile_id) {
        if (observation.hand_counts[tile_id] == 0) continue;
        legal_mask[tile_id / 64] |= uint64_t{1} << (tile_id % 64);
        ++legal_discard_kinds;
    }
    volatile int sink = 0;

    GYMJ::Algo::calculate_shanten(hand.data(), 0);
    GYMJ::Algo::analyze_effective_draws(
        hand.data(),
        visible.data(),
        0
    );
    GYMJ::RL::ActionFeatureEncoder::encode(observation, legal_mask);

    const double shanten_us = microseconds_per_call(iterations, [&]() {
        const auto result = GYMJ::Algo::calculate_shanten(hand.data(), 0);
        sink += result.best;
    });
    const double effective_draws_us = microseconds_per_call(
        iterations,
        [&]() {
            const auto result = GYMJ::Algo::analyze_effective_draws(
                hand.data(),
                visible.data(),
                0
            );
            sink += result.normal.copies;
        }
    );
    const double shanten_decision_us = microseconds_per_call(
        iterations,
        [&]() {
            for (size_t tile_id = 0;
                 tile_id < GYMJ::TILE_TYPE_COUNT;
                 ++tile_id) {
                if (observation.hand_counts[tile_id] == 0) continue;
                auto after_discard = observation.hand_counts;
                --after_discard[tile_id];
                const auto result = GYMJ::Algo::calculate_shanten(
                    after_discard.data(),
                    0
                );
                sink += result.best;
            }
        }
    );
    const double feature_decision_us = microseconds_per_call(
        iterations,
        [&]() {
            const auto features = GYMJ::RL::ActionFeatureEncoder::encode(
                observation,
                legal_mask
            );
            sink += static_cast<int>(features[EXTRA_TILE_ID][
                static_cast<size_t>(
                    GYMJ::RL::ActionFeatureIndex::NormalEffectiveDrawCopies
                )
            ]);
        }
    );

    std::cout << std::fixed << std::setprecision(3)
              << "shanten_us_per_candidate=" << shanten_us << '\n'
              << "effective_draws_us_per_candidate="
              << effective_draws_us << '\n'
              << "candidate_slowdown="
              << effective_draws_us / shanten_us << "x\n"
              << "legal_discard_kinds=" << legal_discard_kinds << '\n'
              << "shanten_us_per_decision=" << shanten_decision_us << '\n'
              << "action_features_us_per_decision="
              << feature_decision_us << '\n'
              << "decision_slowdown="
              << feature_decision_us / shanten_decision_us << "x\n";
    return sink == -1 ? 1 : 0;
}
