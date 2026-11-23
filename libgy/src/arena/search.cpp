#include "search.h"

#include "observation_encoder.h"
#include "shanten.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace GYMJ {
namespace {

using Hand = std::array<uint8_t, TILE_TYPE_COUNT>;
using Random = std::mt19937_64;

bool search_actions(const DecisionContext& decision) {
    if (decision.legal_actions.size() < 2) return false;
    for (const auto& action : decision.legal_actions) {
        if (action.type != ActionType::Discard && action.type != ActionType::Pon &&
            action.type != ActionType::Pass) return false;
    }
    return decision.phase == RoundPhase::AwaitTurnAction ||
           decision.phase == RoundPhase::AwaitDiscardResponses;
}

std::string information_key(const Arena& arena, uint8_t observer) {
    const auto observation = RL::ObservationEncoder::encode(arena, observer);
    std::string key;
    auto byte = [&](uint8_t value) { key.push_back(static_cast<char>(value)); };
    byte(observer);
    byte(static_cast<uint8_t>(observation.phase));
    byte(observation.dealer);
    byte(observation.wall_remaining);
    byte(observation.current_discard);
    byte(observation.current_discard_actor);
    for (auto count : observation.hand_counts) byte(count);
    for (auto kind : observation.ready_kinds) byte(kind);
    for (const auto& event : arena.history()) {
        byte(static_cast<uint8_t>(event.action.type));
        byte(event.action.actor);
        byte(event.action.target);
        byte(event.action.type == ActionType::Draw && event.action.actor != observer
            ? 255 : event.action.tile.id);
        byte(event.applied ? 1 : 0);
    }
    const auto legal = RL::ActionEncoder::encode_legal_actions(arena.current_decision().legal_actions);
    for (auto id : legal.legal_action_ids()) byte(id);
    return key;
}

Hand hand_of(const PlayerState& player) {
    Hand hand{};
    std::copy(std::begin(player.tehai), std::end(player.tehai), hand.begin());
    return hand;
}

int effective_copies(Hand hand, const Hand& visible, int melds, int shanten) {
    int copies = 0;
    for (size_t tile = 0; tile < TILE_TYPE_COUNT; ++tile) {
        if (visible[tile] >= MAX_TILE_COPIES) continue;
        ++hand[tile];
        if (Algo::calculate_shanten(hand.data(), melds).best < shanten) {
            copies += MAX_TILE_COPIES - visible[tile];
        }
        --hand[tile];
    }
    return copies;
}

struct DiscardChoice {
    uint8_t tile = 255;
    int shanten = 99;
    int copies = -1;
};

DiscardChoice best_discard(Hand hand, const Hand& visible, int melds, Random& random) {
    DiscardChoice best;
    uint32_t ties = 0;
    for (uint8_t tile = 0; tile < TILE_TYPE_COUNT; ++tile) {
        if (!hand[tile]) continue;
        --hand[tile];
        const int shanten = Algo::calculate_shanten(hand.data(), melds).best;
        if (shanten <= best.shanten) {
            const int copies = effective_copies(hand, visible, melds, shanten);
            if (shanten < best.shanten || copies > best.copies) {
                best = {tile, shanten, copies};
                ties = 1;
            } else if (copies == best.copies && random() % ++ties == 0) {
                best.tile = tile;
            }
        }
        ++hand[tile];
    }
    if (best.tile == 255) throw std::logic_error("rollout has no discard");
    return best;
}

Action rollout_action(const Arena& arena, const DecisionContext& decision, Random& random) {
    const auto& actions = decision.legal_actions;
    if (actions.empty()) throw std::logic_error("empty rollout decision");
    if (actions.size() == 1) return actions.front();
    for (const auto& action : actions) {
        if (action.type == ActionType::Ron || action.type == ActionType::Tsumo ||
            action.type == ActionType::DeclareReady) return action;
    }
    const auto& player = arena.player(decision.player);
    const auto hand = hand_of(player);
    const auto visible = RL::ObservationEncoder::encode(arena, decision.player).visible_counts;
    const int melds = static_cast<int>(player.melds[0].size());
    if (decision.phase == RoundPhase::AwaitTurnAction) {
        const auto best = best_discard(hand, visible, melds, random);
        // A cheap, fixed rollout model, not the R5 neural policy. Only take a
        // minority of non-worsening kans; the real search agent keeps R5 here.
        for (const auto& action : actions) {
            if (action.type != ActionType::AnKan && action.type != ActionType::Kakan) continue;
            auto after = hand;
            after[action.tile.id] -= action.type == ActionType::AnKan ? 4 : 1;
            const int new_melds = melds + (action.type == ActionType::AnKan ? 1 : 0);
            if (Algo::calculate_shanten(after.data(), new_melds).best <= best.shanten &&
                random() % 100 < 15) return action;
        }
        for (const auto& action : actions) {
            if (action.type == ActionType::Discard && action.tile.id == best.tile) return action;
        }
        throw std::logic_error("rollout selected illegal discard");
    }
    const int pass_shanten = Algo::calculate_shanten(hand.data(), melds).best;
    const int pass_copies = effective_copies(hand, visible, melds, pass_shanten);
    for (const auto& action : actions) {
        if (action.type == ActionType::Daiminkan) {
            auto after = hand;
            after[action.tile.id] -= 3;
            if (Algo::calculate_shanten(after.data(), melds + 1).best <= pass_shanten &&
                random() % 100 < 15) return action;
        }
        if (action.type != ActionType::Pon) continue;
        auto after = hand;
        after[action.tile.id] -= 2;
        // The exposed discard was already visible; two concealed copies move
        // into a meld, so the visible count does not change.
        const auto best = best_discard(after, visible, melds + 1, random);
        if (best.shanten < pass_shanten ||
            (best.shanten == pass_shanten && best.copies > pass_copies) ||
            (best.shanten == pass_shanten && best.copies == pass_copies && random() % 2)) {
            return action;
        }
    }
    for (const auto& action : actions) if (action.type == ActionType::Pass) return action;
    throw std::logic_error("rollout response has no Pass");
}

struct Edge {
    uint8_t id;
    double prior;
    uint32_t visits = 0;
    double sum = 0;
};
struct Node {
    std::vector<Edge> edges;
    uint32_t visits = 0;
};

Node make_node(const DecisionContext& decision, const Arena& arena, Random& random,
               const std::array<double, RL::ACTION_SPACE_SIZE>* prior, double mix) {
    Node node;
    const auto preferred = prior ? uint8_t{255} :
        RL::ActionEncoder::encode_action(rollout_action(arena, decision, random));
    double total = 0;
    for (const auto& action : decision.legal_actions) {
        const uint8_t id = RL::ActionEncoder::encode_action(action);
        const double weight = prior ? (*prior)[id] : (id == preferred ? 1.0 : 0.0);
        if (!std::isfinite(weight) || weight < 0) throw std::invalid_argument("invalid MCTS prior");
        node.edges.push_back({id, weight});
        total += weight;
    }
    if (!(total > 0)) throw std::invalid_argument("MCTS prior has no legal mass");
    for (auto& edge : node.edges) {
        edge.prior = (1 - mix) * edge.prior / total + mix / node.edges.size();
    }
    return node;
}

size_t select_edge(const Node& node, double exploration, bool force_unvisited, Random& random) {
    if (force_unvisited) {
        std::vector<size_t> unvisited;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            if (!node.edges[i].visits) unvisited.push_back(i);
        }
        if (!unvisited.empty()) return unvisited[random() % unvisited.size()];
    }
    double best = -std::numeric_limits<double>::infinity();
    size_t selected = 0;
    for (size_t i = 0; i < node.edges.size(); ++i) {
        const auto& edge = node.edges[i];
        const double value = edge.visits ? edge.sum / edge.visits : 0.0;
        const double score = value + exploration * edge.prior *
            std::sqrt(static_cast<double>(node.visits + 1)) / (1 + edge.visits);
        if (score > best) { best = score; selected = i; }
    }
    return selected;
}

} // namespace

std::string SearchRoot::unsupported_reason(const Arena& arena) {
    if (arena.phase_ != RoundPhase::AwaitTurnAction &&
        arena.phase_ != RoundPhase::AwaitDiscardResponses) return "phase";
    for (const auto& ready : arena.ready_states_) if (ready.declared()) return "declared_ready";
    if (arena.phase_ == RoundPhase::AwaitDiscardResponses && arena.response_index_ != 0) {
        return "earlier_private_responses";
    }
    if (!search_actions(arena.current_decision())) return "action_family_or_forced";
    return "";
}

SearchRoot::SearchRoot(const Arena& arena) {
    const auto reason = unsupported_reason(arena);
    if (!reason.empty()) throw std::invalid_argument("unsupported search root: " + reason);
    observer_ = arena.current_decision().player;
    remaining_ = arena.wall_remaining();
    public_template_ = std::make_unique<Arena>(ArenaConfig{0, arena.dealer(), ""}, Wall::from_tiles({}));
    auto& target = *public_template_;
    // Copy only public metadata. Neither original hidden hands, wall tiles,
    // private response choices, nor the original wall seed are copied.
    target.current_player_ = arena.current_player_;
    target.phase_ = arena.phase_;
    target.discard_history_ = arena.discard_history_;
    target.progress_ = arena.progress_;
    target.initial_ready_index_ = arena.initial_ready_index_;
    target.last_drawn_ = arena.phase_ == RoundPhase::AwaitTurnAction ? arena.last_drawn_ : Tile{255};
    target.last_draw_kind_ = arena.last_draw_kind_;
    target.must_discard_ = arena.must_discard_;
    target.hot_discard_player_ = arena.hot_discard_player_;
    target.pending_discard_is_hot_ = arena.pending_discard_is_hot_;
    target.pending_discard_ = arena.pending_discard_;
    target.response_order_ = arena.response_order_;
    target.response_index_ = arena.response_index_;
    target.decision_sequence_ = arena.decision_sequence_;
    target.history_ = arena.history_;
    for (auto& event : target.history_) {
        if (event.action.type == ActionType::Draw && event.action.actor != observer_) {
            event.action.tile = Tile{255};
        }
    }
    const auto& own = arena.player(observer_);
    for (uint8_t id = 0; id < 4; ++id) {
        auto& player = target.players_[id];
        for (uint8_t relative = 0; relative < 4; ++relative) {
            const uint8_t source = (id + relative + 4 - observer_) % 4;
            player.kawa[relative] = own.kawa[source];
            player.melds[relative] = own.melds[source];
        }
    }
    std::copy(std::begin(own.tehai), std::end(own.tehai), target.players_[observer_].tehai);
    const auto visible = RL::ObservationEncoder::encode(arena, observer_).visible_counts;
    for (uint8_t tile = 0; tile < TILE_TYPE_COUNT; ++tile) {
        if (visible[tile] > MAX_TILE_COPIES) throw std::logic_error("invalid visible count");
        for (uint8_t copy = visible[tile]; copy < MAX_TILE_COPIES; ++copy) unseen_.push_back(Tile{tile});
    }
    size_t required = remaining_;
    for (uint8_t id = 0; id < 4; ++id) {
        if (id != observer_) required += 13 - TILES_PER_MELD * target.players_[id].melds[0].size();
    }
    if (required != unseen_.size()) throw std::logic_error("search root tile conservation failed");
}

std::unique_ptr<Arena> SearchRoot::clone_template() const {
    const auto& source = *public_template_;
    auto target = std::make_unique<Arena>(source.config_, Wall::from_tiles({}));
    target->players_ = source.players_;
    target->current_player_ = source.current_player_;
    target->phase_ = source.phase_;
    target->history_ = source.history_;
    target->discard_history_ = source.discard_history_;
    target->progress_ = source.progress_;
    target->initial_ready_index_ = source.initial_ready_index_;
    target->last_drawn_ = source.last_drawn_;
    target->last_draw_kind_ = source.last_draw_kind_;
    target->must_discard_ = source.must_discard_;
    target->hot_discard_player_ = source.hot_discard_player_;
    target->pending_discard_is_hot_ = source.pending_discard_is_hot_;
    target->pending_discard_ = source.pending_discard_;
    target->response_order_ = source.response_order_;
    target->response_index_ = source.response_index_;
    target->decision_sequence_ = source.decision_sequence_;
    return target;
}

std::unique_ptr<Arena> SearchRoot::sample(uint64_t seed) const {
    auto target = clone_template();
    auto pool = unseen_;
    Random random(seed);
    std::shuffle(pool.begin(), pool.end(), random);
    size_t offset = 0;
    for (uint8_t id = 0; id < 4; ++id) {
        if (id == observer_) continue;
        const size_t count = 13 - TILES_PER_MELD * target->players_[id].melds[0].size();
        for (size_t i = 0; i < count; ++i) ++target->players_[id].tehai[pool.at(offset++).id];
    }
    target->wall_ = Wall::from_tiles(std::vector<Tile>(pool.begin() + offset, pool.end()), seed);
    for (auto& player : target->players_) {
        player.apply_action({ActionType::Pass, player.player_id, Tile{255}, player.player_id});
    }
    if (target->wall_remaining() != remaining_) throw std::logic_error("sampled wall size differs");
    return target;
}

std::string SearchRoot::fingerprint() const {
    // The fixed seed operates only on the public unseen pool, never on the real wall.
    return information_key(*sample(0), observer_);
}

SearchResult SearchRoot::search(const std::array<double, RL::ACTION_SPACE_SIZE>& prior,
                              const SearchConfig& config, uint64_t seed) const {
    return search_impl(prior, config, seed, {});
}

SearchResult SearchRoot::search_policy(const std::array<double, RL::ACTION_SPACE_SIZE>& prior,
    const SearchConfig& config, uint64_t seed, const Policy& policy) const {
    if (!policy) throw std::invalid_argument("neural search requires a policy");
    return search_impl(prior, config, seed, policy);
}

SearchResult SearchRoot::search_impl(const std::array<double, RL::ACTION_SPACE_SIZE>& prior,
    const SearchConfig& config, uint64_t seed, const Policy& policy) const {
    if (!config.simulations || !config.tree_depth || !config.max_steps ||
        !std::isfinite(config.exploration) || config.exploration < 0 ||
        !std::isfinite(config.uniform_prior) || config.uniform_prior < 0 || config.uniform_prior > 1) {
        throw std::invalid_argument("invalid search configuration");
    }
    Random random(seed);
    auto initial = sample(random());
    const auto root_decision = initial->current_decision();
    if (config.simulations < root_decision.legal_actions.size()) {
        throw std::invalid_argument("budget must cover all legal root actions");
    }
    std::unordered_map<std::string, Node> tree;
    const auto root_key = information_key(*initial, observer_);
    tree.emplace(root_key, make_node(root_decision, *initial, random, &prior, config.uniform_prior));
    SearchResult result;
    for (uint32_t simulation = 0; simulation < config.simulations; ++simulation) {
        auto arena = sample(random());
        std::vector<std::pair<Node*, size_t>> path;
        bool tree_policy = true;
        uint32_t steps = 0;
        while (!arena->is_finished()) {
            if (++steps > config.max_steps) throw std::runtime_error("MCTS rollout exceeded step budget");
            const auto decision = arena->current_decision();
            Action selected;
            if (tree_policy && decision.player == observer_ && search_actions(decision) &&
                path.size() < config.tree_depth) {
                const auto key = information_key(*arena, observer_);
                auto found = tree.find(key);
                const bool expanded = found == tree.end();
                if (expanded) {
                    const auto neural_prior = policy ? policy(RL::DecisionFeatureEncoder::encode(*arena, decision.player))
                                                     : std::array<double, RL::ACTION_SPACE_SIZE>{};
                    found = tree.emplace(key, make_node(decision, *arena, random,
                        policy ? &neural_prior : nullptr, config.uniform_prior)).first;
                }
                auto& node = found->second;
                const size_t edge = select_edge(node, config.exploration, path.empty(), random);
                const auto encoded = RL::ActionEncoder::encode_legal_actions(decision.legal_actions);
                selected = encoded.action_at(node.edges[edge].id);
                path.push_back({&node, edge});
                if (expanded) tree_policy = false;
            } else {
                if (policy && decision.legal_actions.size() > 1) {
                    const auto probabilities = policy(RL::DecisionFeatureEncoder::encode(*arena, decision.player));
                    double best = -1;
                    for (const auto& action : decision.legal_actions) {
                        const double p = probabilities[RL::ActionEncoder::encode_action(action)];
                        if (!std::isfinite(p) || p < 0) throw std::invalid_argument("invalid neural probability");
                        if (p > best) { best = p; selected = action; }
                    }
                    if (!(best > 0)) throw std::invalid_argument("neural policy has no legal mass");
                } else {
                    selected = policy ? decision.legal_actions.front() : rollout_action(*arena, decision, random);
                }
            }
            arena->submit_action(selected);
        }
        const auto scores = arena->round_settlement().ledger.score_deltas();
        if (std::accumulate(scores.begin(), scores.end(), 0) != 0) throw std::logic_error("nonzero sum rollout");
        const double value = static_cast<double>(scores[observer_]) / 30.0;
        for (const auto& item : path) {
            ++item.first->visits;
            auto& edge = item.first->edges[item.second];
            ++edge.visits;
            edge.sum += value;
        }
        result.deepest_path = std::max(result.deepest_path, static_cast<uint32_t>(path.size()));
        result.rollout_steps += steps;
        ++result.simulations;
    }
    const auto& root = tree.at(root_key);
    const Edge* best = nullptr;
    for (const auto& edge : root.edges) {
        result.visits[edge.id] = edge.visits;
        result.mean_scores[edge.id] = edge.visits ? edge.sum / edge.visits * 30.0 : 0.0;
        if (!best || edge.visits > best->visits ||
            (edge.visits == best->visits && edge.prior > best->prior)) best = &edge;
    }
    result.action_id = best->id;
    result.nodes = static_cast<uint32_t>(tree.size());
    return result;
}

} // namespace GYMJ
