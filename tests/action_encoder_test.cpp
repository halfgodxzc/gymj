#include "action_encoder.h"
#include "arena.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

bool actions_equal(const GYMJ::Action& lhs, const GYMJ::Action& rhs) {
    return lhs.type == rhs.type &&
           lhs.actor == rhs.actor &&
           lhs.tile == rhs.tile &&
           lhs.target == rhs.target;
}

} // namespace

void test_action_id_layout() {
    using namespace GYMJ;
    using GYMJ::RL::ActionEncoder;

    assert(ActionEncoder::encode_action({
        ActionType::Discard, 0, Tile{0}, 0
    }) == 0);
    assert(ActionEncoder::encode_action({
        ActionType::Discard, 0, Tile{26}, 0
    }) == 26);
    assert(ActionEncoder::encode_action({
        ActionType::AnKan, 0, Tile{0}, 0
    }) == 27);
    assert(ActionEncoder::encode_action({
        ActionType::AnKan, 0, Tile{26}, 0
    }) == 53);
    assert(ActionEncoder::encode_action({
        ActionType::Kakan, 0, Tile{0}, 0
    }) == 54);
    assert(ActionEncoder::encode_action({
        ActionType::Kakan, 0, Tile{26}, 0
    }) == 80);
    assert(ActionEncoder::encode_action({
        ActionType::Pon, 1, Tile{5}, 0
    }) == 81);
    assert(ActionEncoder::encode_action({
        ActionType::Daiminkan, 1, Tile{5}, 0
    }) == 82);
    assert(ActionEncoder::encode_action({
        ActionType::Tsumo, 1, Tile{5}, 1
    }) == 83);
    assert(ActionEncoder::encode_action({
        ActionType::Ron, 1, Tile{5}, 0
    }) == 84);
    assert(ActionEncoder::encode_action({
        ActionType::Pass, 1, Tile{5}, 0
    }) == 85);
    assert(ActionEncoder::encode_action({
        ActionType::DeclareReady, 1, Tile{255}, 1
    }) == 86);
}

void test_mask_and_action_round_trip() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    const std::vector<Action> legal_actions = {
        {ActionType::Discard, 2, Tile{0}, 2},
        {ActionType::AnKan, 2, Tile{26}, 2},
        {ActionType::Kakan, 2, Tile{10}, 2},
        {ActionType::Pass, 2, Tile{7}, 1},
        {ActionType::DeclareReady, 2, Tile{255}, 2},
    };
    const EncodedDecision encoded =
        ActionEncoder::encode_legal_actions(legal_actions);

    assert(encoded.is_legal(0));
    assert(encoded.is_legal(53));
    assert(encoded.is_legal(64));
    assert(encoded.is_legal(85));
    assert(encoded.is_legal(86));
    assert(!encoded.is_legal(1));
    assert(!encoded.is_legal(87));

    const auto mask = encoded.expanded_mask();
    int legal_count = 0;
    for (uint8_t value : mask) legal_count += value;
    assert(legal_count == 5);

    assert((encoded.mask_words()[0] & (uint64_t{1} << 0)) != 0);
    assert((encoded.mask_words()[0] & (uint64_t{1} << 53)) != 0);
    assert((encoded.mask_words()[1] & (uint64_t{1} << 0)) != 0);
    assert((encoded.mask_words()[1] & (uint64_t{1} << 21)) != 0);
    assert((encoded.mask_words()[1] & (uint64_t{1} << 22)) != 0);

    for (const auto& action : legal_actions) {
        const uint8_t action_id = ActionEncoder::encode_action(action);
        assert(actions_equal(encoded.action_at(action_id), action));
    }
}

void test_invalid_and_colliding_actions_are_rejected() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    bool draw_threw = false;
    try {
        ActionEncoder::encode_action({
            ActionType::Draw, 0, Tile{0}, 0
        });
    } catch (const std::invalid_argument&) {
        draw_threw = true;
    }
    assert(draw_threw);

    bool tile_threw = false;
    try {
        ActionEncoder::encode_action({
            ActionType::Discard, 0, Tile{27}, 0
        });
    } catch (const std::invalid_argument&) {
        tile_threw = true;
    }
    assert(tile_threw);

    bool collision_threw = false;
    try {
        ActionEncoder::encode_legal_actions({
            {ActionType::Pon, 1, Tile{0}, 0},
            {ActionType::Pon, 1, Tile{1}, 2},
        });
    } catch (const std::logic_error&) {
        collision_threw = true;
    }
    assert(collision_threw);
}

void test_real_arena_decision_round_trip() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{123456, 0});
    arena.start_round();
    while (arena.phase() == RoundPhase::AwaitReadyDeclaration) {
        const auto ready = arena.current_decision();
        const auto encoded_ready = ActionEncoder::encode_legal_actions(
            ready.legal_actions
        );
        arena.submit_action(encoded_ready.action_at(85));
    }

    const DecisionContext decision = arena.current_decision();
    const EncodedDecision encoded =
        ActionEncoder::encode_legal_actions(decision.legal_actions);
    assert(encoded.legal_action_ids().size() == decision.legal_actions.size());

    for (const auto& action : decision.legal_actions) {
        const uint8_t action_id = ActionEncoder::encode_action(action);
        assert(actions_equal(encoded.action_at(action_id), action));
    }

    const uint8_t selected_id = encoded.legal_action_ids().front();
    arena.submit_action(encoded.action_at(selected_id));
    assert(arena.phase() == RoundPhase::AwaitDiscardResponses);

    const DecisionContext response = arena.current_decision();
    const EncodedDecision encoded_response =
        ActionEncoder::encode_legal_actions(response.legal_actions);
    assert(encoded_response.is_legal(85));
    assert(encoded_response.action_at(85).type == ActionType::Pass);
}

int main() {
    static_assert(GYMJ::RL::ACTION_ENCODING_VERSION == 2);
    static_assert(GYMJ::RL::ACTION_SPACE_SIZE == 87);

    test_action_id_layout();
    test_mask_and_action_round_trip();
    test_invalid_and_colliding_actions_are_rejected();
    test_real_arena_decision_round_trip();
    return 0;
}
