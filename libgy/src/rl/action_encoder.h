#pragma once

#include "guiyang_mahjong.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace GYMJ {
namespace RL {

constexpr uint32_t ACTION_ENCODING_VERSION = 2;
constexpr size_t ACTION_SPACE_SIZE = 87;

class EncodedDecision {
public:
    bool is_legal(size_t action_id) const;
    const Action& action_at(size_t action_id) const;
    std::array<uint8_t, ACTION_SPACE_SIZE> expanded_mask() const;
    std::vector<uint8_t> legal_action_ids() const;
    const std::array<uint64_t, 2>& mask_words() const;

private:
    friend class ActionEncoder;

    std::array<uint64_t, 2> mask_words_{};
    std::array<std::optional<Action>, ACTION_SPACE_SIZE> actions_{};
};

class ActionEncoder {
public:
    static uint8_t encode_action(const Action& action);
    static EncodedDecision encode_legal_actions(
        const std::vector<Action>& legal_actions
    );
};

} // namespace RL
} // namespace GYMJ
