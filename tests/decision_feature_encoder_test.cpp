#include "decision_feature_encoder.h"
#include "privileged_pattern_encoder.h"

#include "shanten.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <random>
#include <vector>

namespace {

std::vector<GYMJ::Tile> scripted_wall(
    std::array<std::vector<uint8_t>, 4> hands,
    const std::vector<uint8_t>& front_draws,
    std::optional<uint8_t> replacement_draw = std::nullopt
) {
    std::array<int, GYMJ::TILE_TYPE_COUNT> remaining;
    remaining.fill(GYMJ::MAX_TILE_COPIES);
    auto reserve = [&](uint8_t tile_id) {
        assert(tile_id < GYMJ::TILE_TYPE_COUNT);
        assert(remaining[tile_id] > 0);
        --remaining[tile_id];
    };

    for (const auto& hand : hands) {
        assert(hand.size() <= 13);
        for (uint8_t tile_id : hand) reserve(tile_id);
    }
    for (uint8_t tile_id : front_draws) reserve(tile_id);
    if (replacement_draw.has_value()) reserve(*replacement_draw);

    for (auto& hand : hands) {
        for (uint8_t tile_id = 0;
             hand.size() < 13 && tile_id < GYMJ::TILE_TYPE_COUNT;
             ++tile_id) {
            while (hand.size() < 13 && remaining[tile_id] > 0) {
                hand.push_back(tile_id);
                --remaining[tile_id];
            }
        }
        assert(hand.size() == 13);
    }

    std::vector<GYMJ::Tile> wall;
    for (size_t index = 0; index < 13; ++index) {
        for (uint8_t player = 0; player < GYMJ::PLAYER_COUNT; ++player) {
            wall.push_back(GYMJ::Tile{hands[player][index]});
        }
    }
    for (uint8_t tile_id : front_draws) {
        wall.push_back(GYMJ::Tile{tile_id});
    }
    for (uint8_t tile_id = 0; tile_id < GYMJ::TILE_TYPE_COUNT; ++tile_id) {
        while (remaining[tile_id]-- > 0) {
            wall.push_back(GYMJ::Tile{tile_id});
        }
    }
    if (replacement_draw.has_value()) {
        wall.push_back(GYMJ::Tile{*replacement_draw});
    }
    assert(wall.size() == 108);
    return wall;
}

GYMJ::Action find_action(
    const GYMJ::DecisionContext& decision,
    GYMJ::ActionType type,
    int tile_id = -1
) {
    const auto found = std::find_if(
        decision.legal_actions.begin(),
        decision.legal_actions.end(),
        [&](const GYMJ::Action& action) {
            return action.type == type &&
                   (tile_id < 0 || action.tile.id == tile_id);
        }
    );
    assert(found != decision.legal_actions.end());
    return *found;
}

void start_round_passing_initial_ready(GYMJ::Arena& arena) {
    arena.start_round();
    while (arena.phase() == GYMJ::RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
}

void decline_ready_if_offered(GYMJ::Arena& arena) {
    if (arena.phase() != GYMJ::RoundPhase::AwaitReadyDeclaration) return;
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Pass
    ));
}

void pass_all_discard_responses(GYMJ::Arena& arena) {
    decline_ready_if_offered(arena);
    for (int i = 0; i < 3; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
    decline_ready_if_offered(arena);
}

void advance_to_player_one_kakan(
    GYMJ::Arena& arena,
    uint8_t pon_discard,
    const std::array<uint8_t, 3>& turn_discards
) {
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Pon,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Pass
        ));
    }
    arena.submit_action(find_action(
        arena.current_decision(),
        GYMJ::ActionType::Discard,
        pon_discard
    ));
    pass_all_discard_responses(arena);

    const std::array<uint8_t, 3> players{2, 3, 0};
    for (size_t i = 0; i < players.size(); ++i) {
        assert(arena.current_player() == players[i]);
        arena.submit_action(find_action(
            arena.current_decision(),
            GYMJ::ActionType::Discard,
            turn_discards[i]
        ));
        pass_all_discard_responses(arena);
    }
    assert(arena.current_player() == 1);
}

} // namespace

void test_discard_candidates_have_structured_post_states() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{13001, 0});
    start_round_passing_initial_ready(arena);
    const uint8_t actor = arena.current_decision().player;
    const auto features = DecisionFeatureEncoder::encode(arena, actor);
    const auto decision = arena.current_decision();
    assert(features.observation.current_actor == 0);
    assert(features.observation.hand_counts ==
           ObservationEncoder::encode(arena, actor).hand_counts);

    for (const Action& action : decision.legal_actions) {
        const uint8_t action_id = ActionEncoder::encode_action(action);
        const auto& candidate = features.actions[action_id];
        assert(candidate.legal == 1);
        if (action.type == ActionType::Discard) {
            assert(candidate.supported_by_v4 == 1);
            assert(candidate.semantics.family == ActionFamily::Turn);
            assert(candidate.semantics.tile == action.tile.id);
            assert(candidate.post_hand_known == 1);
            assert(candidate.post_hand.post_discard_state == 1);

            std::array<uint8_t, TILE_TYPE_COUNT> expected{};
            std::copy(
                arena.player(actor).tehai,
                arena.player(actor).tehai + TILE_TYPE_COUNT,
                expected.begin()
            );
            --expected[action.tile.id];
            assert(candidate.post_hand.routes[0].shanten ==
                   Algo::calculate_normal_shanten(
                       expected.data(),
                       static_cast<int>(arena.player(actor).melds[0].size())
                   ));
        } else if (action.type == ActionType::AnKan ||
                   action.type == ActionType::Kakan) {
            assert(candidate.supported_by_v4 == 0);
            assert(candidate.semantics.family == ActionFamily::Unsupported);
        }
    }
}

void test_pon_keeps_every_follow_up_discard_and_pass_opportunity() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {0, 0, 1, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19};
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {26}))
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        0
    ));
    if (arena.phase() == RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }

    assert(arena.current_decision().player == 1);
    const auto features = DecisionFeatureEncoder::encode(arena, 1);
    const auto& pon = features.actions[81];
    const auto& pass = features.actions[85];
    assert(pon.legal == 1);
    assert(pon.supported_by_v4 == 1);
    assert(pon.semantics.family == ActionFamily::Response);
    assert(pon.semantics.relative_source == 3);
    assert(pon.semantics.relative_target == 0);
    assert(pon.post_hand_known == 0);
    assert(features.pon_follow_up.available == 1);
    assert(features.pon_follow_up.discard_mask[0] == 0);
    assert(features.pon_follow_up.discard_mask[1] == 1);
    assert(features.pon_follow_up.discard_mask[19] == 1);
    assert(features.pon_follow_up.post_discard_hands[1]
               .post_discard_state == 1);
    assert(pass.pass.available == 1);
    assert(pass.pass.declines_pon == 1);
    assert(pass.pass.declines_ron == 0);

    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Pon,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }
    const auto after_pon = DecisionFeatureEncoder::encode(arena, 1);
    assert(after_pon.progress.must_discard_after_call == 1);
    for (const auto& candidate : after_pon.actions) {
        if (candidate.legal) {
            assert(candidate.semantics.type == ActionType::Discard);
        }
    }
}

void test_tsumo_action_contains_exact_tianhu_score() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {26}))
    );
    start_round_passing_initial_ready(arena);

    const auto features = DecisionFeatureEncoder::encode(arena, 0);
    const auto& tsumo = features.actions[83];
    assert(tsumo.legal == 1);
    assert(tsumo.hu.available == 1);
    assert(tsumo.hu.is_tsumo == 1);
    assert(tsumo.hu.is_qianggang == 0);
    assert(tsumo.hu.timing_fan == 9);
    assert(tsumo.hu.total_fan == 9);
    assert(tsumo.hu.hu_points == 45);
}

void test_gangkai_tsumo_action_contains_exact_special_score() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {0};
    hands[1] = {
        0, 0, 0,
        1, 2, 3,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {25}, 26))
    );
    start_round_passing_initial_ready(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        0
    ));
    decline_ready_if_offered(arena);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Daiminkan,
        0
    ));
    for (int i = 0; i < 2; ++i) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }

    const auto features = DecisionFeatureEncoder::encode(arena, 1);
    const auto& tsumo = features.actions[83];
    assert(features.progress.last_draw_kind ==
           DecisionDrawKind::Replacement);
    assert(tsumo.legal == 1);
    assert(tsumo.hu.available == 1);
    assert(tsumo.hu.is_tsumo == 1);
    assert(tsumo.hu.special_fan == 2);
    assert(tsumo.hu.total_fan == 2);
    assert(tsumo.hu.hu_points == 10);
}

void test_qianggang_ron_action_contains_exact_special_score() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    const std::array<std::vector<uint8_t>, 4> hands{{
        {0, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17},
        {0, 0, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24},
        {1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 26, 26},
        {6, 6, 6, 7, 7, 7, 8, 8, 8, 12, 12, 12, 13},
    }};
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(
            hands,
            {25, 6, 13, 24, 0}
        ))
    );
    advance_to_player_one_kakan(arena, 24, {6, 13, 24});
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Kakan,
        0
    ));

    assert(arena.current_decision().player == 2);
    const auto features = DecisionFeatureEncoder::encode(arena, 2);
    const auto& ron = features.actions[84];
    const auto& pass = features.actions[85];
    assert(features.progress.pending_kakan == 1);
    assert(ron.legal == 1);
    assert(ron.hu.available == 1);
    assert(ron.hu.is_qianggang == 1);
    assert(ron.hu.special_fan == 3);
    assert(ron.hu.total_fan == 3);
    assert(ron.hu.hu_points == 15);
    assert(ron.semantics.relative_source == 3);
    assert(ron.semantics.relative_target == 0);
    assert(pass.pass.declines_ron == 1);
}

void test_ready_action_exposes_locked_wait_value() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[1] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {7}))
    );
    arena.start_round();
    assert(arena.current_decision().player == 1);

    const auto features = DecisionFeatureEncoder::encode(arena, 1);
    const auto& ready = features.actions[86];
    const auto& pass = features.actions[85];
    assert(ready.legal == 1);
    assert(ready.semantics.family == ActionFamily::Ready);
    assert(ready.ready.available == 1);
    assert(ready.ready.kind == ReadyKind::TianTing);
    assert((ready.ready.wait_mask & (uint32_t{1} << 26)) != 0);
    assert(ready.ready.wait_kinds >= 1);
    assert(ready.ready.wait_copies >= 1);
    assert(ready.ready.max_base_hu_points >= 5);
    assert(pass.pass.declines_ready == 1);
}

void test_ron_action_contains_ready_and_kill_report_score() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    std::array<std::vector<uint8_t>, 4> hands;
    hands[0] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        26,
    };
    hands[1] = {
        0, 1, 2,
        3, 4, 5,
        9, 10, 11,
        18, 19, 20,
        7,
    };
    Arena arena(
        ArenaConfig{0, 0},
        Wall::from_tiles(scripted_wall(hands, {7}))
    );
    arena.start_round();
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::DeclareReady
    ));
    while (arena.phase() == RoundPhase::AwaitReadyDeclaration) {
        arena.submit_action(find_action(
            arena.current_decision(),
            ActionType::Pass
        ));
    }
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard,
        7
    ));
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::DeclareReady
    ));

    assert(arena.current_decision().player == 1);
    const auto features = DecisionFeatureEncoder::encode(arena, 1);
    const auto& ron = features.actions[84];
    const auto& pass = features.actions[85];
    assert(ron.legal == 1);
    assert(ron.hu.available == 1);
    assert(ron.hu.ready_fan == 5);
    assert(ron.hu.special_fan == 3);
    assert(ron.hu.total_fan == 8);
    assert(ron.hu.hu_points == 40);
    assert(ron.semantics.relative_source == 3);
    assert(ron.semantics.relative_target == 0);
    assert(pass.pass.declines_ron == 1);
}

void test_decision_sequence_advances_without_logging() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    Arena arena(ArenaConfig{44004, 0});
    start_round_passing_initial_ready(arena);
    const uint8_t actor = arena.current_decision().player;
    const auto before = DecisionFeatureEncoder::encode(arena, actor);
    arena.submit_action(find_action(
        arena.current_decision(),
        ActionType::Discard
    ));
    if (arena.phase() == RoundPhase::AwaitReadyDeclaration) {
        const auto after_discard = DecisionFeatureEncoder::encode(
            arena,
            arena.current_decision().player
        );
        assert(after_discard.progress.decision_sequence ==
               before.progress.decision_sequence + 1);
    } else {
        const auto response = DecisionFeatureEncoder::encode(
            arena,
            arena.current_decision().player
        );
        assert(response.progress.decision_sequence ==
               before.progress.decision_sequence + 1);
    }
}

void test_random_rounds_preserve_feature_invariants() {
    using namespace GYMJ;
    using namespace GYMJ::RL;

    for (uint64_t seed = 55000; seed < 55012; ++seed) {
        Arena arena(ArenaConfig{seed, static_cast<uint8_t>(seed % 4)});
        arena.start_round();
        std::mt19937_64 random(seed ^ UINT64_C(0xA0761D6478BD642F));
        uint32_t expected_sequence = 0;
        int decision_count = 0;

        while (!arena.is_finished()) {
            assert(decision_count++ < 500);
            const DecisionContext decision = arena.current_decision();
            const DecisionFeatureSet actor_features =
                DecisionFeatureEncoder::encode(arena, decision.player);
            const PrivilegedPatternFeatures critic_features =
                PrivilegedPatternEncoder::encode(arena, decision.player);
            assert(actor_features.progress.decision_sequence ==
                   expected_sequence);
            assert(critic_features.progress.decision_sequence ==
                   expected_sequence);

            std::array<uint8_t, ACTION_SPACE_SIZE> expected_legal{};
            for (const Action& action : decision.legal_actions) {
                expected_legal[ActionEncoder::encode_action(action)] = 1;
            }
            for (size_t action_id = 0; action_id < ACTION_SPACE_SIZE;
                 ++action_id) {
                assert(actor_features.actions[action_id].legal ==
                       expected_legal[action_id]);
            }

            int wall_count = 0;
            for (uint8_t copies : critic_features.remaining_wall_counts) {
                wall_count += copies;
            }
            assert(wall_count == static_cast<int>(arena.wall_remaining()));

            const size_t selected = static_cast<size_t>(
                random() % decision.legal_actions.size()
            );
            arena.submit_action(decision.legal_actions[selected]);
            ++expected_sequence;
        }
    }
}

int main() {
    static_assert(GYMJ::RL::DECISION_FEATURE_VERSION == 3);
    test_discard_candidates_have_structured_post_states();
    test_pon_keeps_every_follow_up_discard_and_pass_opportunity();
    test_tsumo_action_contains_exact_tianhu_score();
    test_gangkai_tsumo_action_contains_exact_special_score();
    test_qianggang_ron_action_contains_exact_special_score();
    test_ready_action_exposes_locked_wait_value();
    test_ron_action_contains_ready_and_kill_report_score();
    test_decision_sequence_advances_without_logging();
    test_random_rounds_preserve_feature_invariants();
    return 0;
}
