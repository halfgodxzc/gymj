#include "action_encoder.h"

#include <stdexcept>

namespace GYMJ {
namespace RL {
namespace {

void validate_tile(const Action& action) {
    if (action.tile.id >= 27) {
        throw std::invalid_argument("tile action must use a tile id in range 0..26");
    }
}

bool action_uses_tile(ActionType type) {
    return type != ActionType::Pass &&
           type != ActionType::DeclareReady;
}

} // namespace

bool EncodedDecision::is_legal(size_t action_id) const {
    if (action_id >= ACTION_SPACE_SIZE) return false;
    const size_t word = action_id / 64;
    const size_t bit = action_id % 64;
    return (mask_words_[word] & (uint64_t{1} << bit)) != 0;
}

const Action& EncodedDecision::action_at(size_t action_id) const {
    if (!is_legal(action_id)) {
        throw std::out_of_range("action id is not legal in this decision");
    }
    return *actions_[action_id];
}

std::array<uint8_t, ACTION_SPACE_SIZE> EncodedDecision::expanded_mask() const {
    std::array<uint8_t, ACTION_SPACE_SIZE> mask{};
    for (size_t action_id = 0; action_id < ACTION_SPACE_SIZE; ++action_id) {
        mask[action_id] = is_legal(action_id) ? 1 : 0;
    }
    return mask;
}

std::vector<uint8_t> EncodedDecision::legal_action_ids() const {
    std::vector<uint8_t> ids;
    for (size_t action_id = 0; action_id < ACTION_SPACE_SIZE; ++action_id) {
        if (is_legal(action_id)) {
            ids.push_back(static_cast<uint8_t>(action_id));
        }
    }
    return ids;
}

const std::array<uint64_t, 2>& EncodedDecision::mask_words() const {
    return mask_words_;
}

uint8_t ActionEncoder::encode_action(const Action& action) {
    if (action_uses_tile(action.type)) validate_tile(action);

    switch (action.type) {
        case ActionType::Discard:
            return action.tile.id;
        case ActionType::AnKan:
            return static_cast<uint8_t>(27 + action.tile.id);
        case ActionType::Kakan:
            return static_cast<uint8_t>(54 + action.tile.id);
        case ActionType::Pon:
            return 81;
        case ActionType::Daiminkan:
            return 82;
        case ActionType::Tsumo:
            return 83;
        case ActionType::Ron:
            return 84;
        case ActionType::Pass:
            return 85;
        case ActionType::DeclareReady:
            return 86;
        case ActionType::Draw:
            throw std::invalid_argument("Draw is an Arena event, not a policy action");
    }
    throw std::invalid_argument("unknown action type");
}

EncodedDecision ActionEncoder::encode_legal_actions(
    const std::vector<Action>& legal_actions
) {
    EncodedDecision encoded;
    for (const auto& action : legal_actions) {
        const uint8_t action_id = encode_action(action);
        if (encoded.actions_[action_id].has_value()) {
            throw std::logic_error("multiple legal actions map to the same action id");
        }

        encoded.actions_[action_id] = action;
        const size_t word = action_id / 64;
        const size_t bit = action_id % 64;
        encoded.mask_words_[word] |= uint64_t{1} << bit;
    }
    return encoded;
}

} // namespace RL
} // namespace GYMJ
