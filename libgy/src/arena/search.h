#pragma once

#include "arena.h"
#include "action_encoder.h"
#include "decision_feature_encoder.h"
#include <functional>

#include <array>
#include <memory>
#include <string>

namespace GYMJ {

struct SearchConfig {
    uint32_t simulations = 128;
    uint32_t tree_depth = 3;
    uint32_t max_steps = 1000;
    double exploration = 2.0;
    double uniform_prior = 0.25;
};

struct SearchResult {
    uint8_t action_id = 255;
    std::array<uint32_t, RL::ACTION_SPACE_SIZE> visits{};
    std::array<double, RL::ACTION_SPACE_SIZE> mean_scores{};
    uint32_t simulations = 0;
    uint32_t nodes = 0;
    uint32_t deepest_path = 0;
    uint64_t rollout_steps = 0;
};

// Contains public state and the acting player's own private information only.
// Other hands are sampled uniformly subject to current physical constraints;
// this is deliberately not a learned posterior over opponents' past choices.
class SearchRoot {
public:
    using Policy = std::function<std::array<double, RL::ACTION_SPACE_SIZE>(const RL::DecisionFeatureSet&)>;
    explicit SearchRoot(const Arena& arena);
    static std::string unsupported_reason(const Arena& arena);
    std::unique_ptr<Arena> sample(uint64_t seed) const;
    SearchResult search(
        const std::array<double, RL::ACTION_SPACE_SIZE>& prior,
        const SearchConfig& config, uint64_t seed) const;
    SearchResult search_policy(
        const std::array<double, RL::ACTION_SPACE_SIZE>& prior,
        const SearchConfig& config, uint64_t seed, const Policy& policy) const;
    uint8_t observer() const { return observer_; }
    size_t remaining() const { return remaining_; }
    std::string fingerprint() const;

private:
    SearchResult search_impl(const std::array<double, RL::ACTION_SPACE_SIZE>& prior,
        const SearchConfig& config, uint64_t seed, const Policy& policy) const;
    std::unique_ptr<Arena> clone_template() const;
    std::unique_ptr<Arena> public_template_;
    uint8_t observer_;
    size_t remaining_;
    std::vector<Tile> unseen_;
};

} // namespace GYMJ
