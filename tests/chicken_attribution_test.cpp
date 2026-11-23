#include "chicken_attribution.h"

#include <cassert>
#include <stdexcept>

using namespace GYMJ;

namespace {
struct Fixture {
    SettlementSnapshot snapshot;
    ChickenContext context;
    std::vector<GameEvent> history;
    std::vector<DiscardEvent> discards;
    ChongfengTracker tracker;

    Fixture() {
        for (uint8_t p = 0; p < PLAYER_COUNT; ++p) {
            snapshot.players[p].player_id = p;
            context.is_jiao[p] = true;
        }
    }
    uint32_t event(ActionType type, uint8_t p, Tile tile, uint8_t target = 255,
                   bool applied = true) {
        auto sequence = static_cast<uint32_t>(history.size());
        history.push_back({sequence, {type, p, tile, target},
                           RoundPhase::AwaitTurnAction, RoundPhase::AwaitTurnAction, applied});
        return sequence;
    }
    uint32_t discard(uint8_t p, Tile tile, uint8_t caller = 255,
                     ActionType call_type = ActionType::Pon) {
        auto sequence = event(ActionType::Discard, p, tile);
        bool qualified = tracker.observe_discard(p, tile);
        discards.push_back({tile, p, caller != 255, caller});
        if (caller == 255) {
            snapshot.players[p].discards.push_back(tile);
            if (qualified) context.chongfeng_tile_masks[p] |= tile.id == YAO_JI_TILE_ID
                ? CHONGFENG_YAO_JI_MASK : CHONGFENG_BA_TONG_MASK;
        } else {
            event(call_type, caller, tile, p, false); // submitted, not yet applied
            event(ActionType::Pass, (caller + 1) % PLAYER_COUNT, tile, p, false);
            event(call_type, caller, tile, p);
            snapshot.players[caller].melds.push_back({call_type == ActionType::Pon
                ? MeldType::Pon : MeldType::Daiminkan, tile, p});
            if (qualified) context.responsibility_chickens.push_back({p, caller, tile});
        }
        return sequence;
    }
    RoundSettlementResult settle() {
        auto chicken = ChickenResolver::resolve(snapshot, context);
        RoundResult round;
        round.reason = RoundEndReason::Tsumo;
        round.hu_players = {3};
        HuSettlement hu;
        hu.player_id = 3;
        hu.score.hu_points = 5;
        auto result = RoundSettlementCalculator::calculate(round, {hu}, {}, chicken, {});
        const auto deltas = result.ledger.score_deltas();
        const auto transfer_count = result.ledger.transfers().size();
        attribute_chicken_settlement(result, history, discards);
        assert(result.ledger.score_deltas() == deltas);
        assert(result.ledger.transfers().size() == transfer_count);
        std::array<int, PLAYER_COUNT> expected{};
        for (const auto& t : result.ledger.transfers()) {
            if (t.category == ScoreCategory::Chicken || t.category == ScoreCategory::ResponsibilityChicken) {
                expected[t.from] -= t.points;
                expected[t.to] += t.points;
            }
        }
        auto actual = result.chicken_unattributed_deltas;
        for (const auto& a : result.chicken_attributions) actual[a.player_id] += a.points;
        assert(actual == expected);
        const auto size = result.chicken_attributions.size();
        attribute_chicken_settlement(result, history, discards);
        assert(result.chicken_attributions.size() == size);
        return result;
    }
};

std::array<int, PLAYER_COUNT> totals(const RoundSettlementResult& result) {
    std::array<int, PLAYER_COUNT> values{};
    for (const auto& a : result.chicken_attributions) values[a.player_id] += a.points;
    return values;
}

void test_chongfeng_and_negative_other_owner() {
    Fixture f;
    f.discard(0, Tile{YAO_JI_TILE_ID});
    f.discard(1, Tile{BA_TONG_TILE_ID});
    f.context.is_jiao[1] = false;
    auto result = f.settle();
    // P0 also receives P1's negative chicken. Do not count that as P0's own chicken.
    assert((totals(result) == std::array<int, 4>{15, -15, 0, 0}));
    assert((result.ledger.score_deltas() == std::array<int, 4>{15, -25, -5, 15}));
}

void test_responsibility_matrix() {
    for (bool source_jiao : {false, true}) {
        for (bool owner_jiao : {false, true}) {
            for (bool gold : {false, true}) {
                Fixture f;
                f.discard(0, Tile{YAO_JI_TILE_ID}, 1);
                f.context.is_jiao[0] = source_jiao;
                f.context.is_jiao[1] = owner_jiao;
                if (gold) f.context.indicator = Tile{26};
                auto result = f.settle();
                const int multiplier = gold ? 2 : 1;
                const int responsibility = owner_jiao ? 5 : source_jiao ? -5 : 0;
                const int pon = owner_jiao ? 18 : -6 * (2 + int(source_jiao));
                assert((totals(result) == std::array<int, 4>{
                    -responsibility * multiplier, (pon + responsibility) * multiplier, 0, 0}));
                std::array<int, 4> expected{-5, -5, -5, 15};
                for (int p : {0, 2, 3}) {
                    const int payment = owner_jiao ? -6 :
                        ((p != 0 || source_jiao) ? 6 : 0);
                    expected[p] += payment * multiplier;
                    expected[1] -= payment * multiplier;
                }
                expected[0] -= responsibility * multiplier;
                expected[1] += responsibility * multiplier;
                assert(result.ledger.score_deltas() == expected);
                for (const auto& a : result.chicken_attributions) {
                    assert(a.event_sequence == (a.player_id == 0 ? 0 : 3));
                    assert(a.kind != ChickenAttributionKind::Chongfeng);
                }
            }
        }
    }
}

void test_burned_and_daiminkan() {
    Fixture f;
    f.discard(0, Tile{YAO_JI_TILE_ID}, 1);
    f.context.burned_players[1] = true;
    auto result = f.settle();
    // Preserve current engine semantics: burn zeroes ordinary chickens, not responsibility.
    assert((totals(result) == std::array<int, 4>{-5, 5, 0, 0}));
    Fixture g;
    g.discard(0, Tile{YAO_JI_TILE_ID}, 1, ActionType::Daiminkan);
    result = g.settle();
    assert((totals(result) == std::array<int, 4>{-5, 0, 0, 0}));
}

void test_follow_repeat_and_kakan() {
    Fixture f;
    f.discard(0, Tile{YAO_JI_TILE_ID});
    f.discard(1, Tile{YAO_JI_TILE_ID}); // opening follow
    f.discard(0, Tile{YAO_JI_TILE_ID}); // not another charge
    auto result = f.settle();
    assert((totals(result) == std::array<int, 4>{15, 15, 0, 0}));
    for (const auto& a : result.chicken_attributions) assert(a.event_sequence != 2);

    Fixture g;
    g.discard(0, Tile{BA_TONG_TILE_ID}, 1);
    g.snapshot.players[1].melds[0].type = MeldType::Kakan;
    g.event(ActionType::Kakan, 1, Tile{BA_TONG_TILE_ID}, 0);
    result = g.settle();
    assert((totals(result) == std::array<int, 4>{-5, 23, 0, 0}));
    assert(result.chicken_unattributed_deltas[1] == 6); // fourth tile, three payers
}

void test_nonqualified_pon_and_absent_chicken() {
    Fixture f;
    f.discard(0, Tile{YAO_JI_TILE_ID});
    f.discard(1, Tile{0});
    f.discard(1, Tile{YAO_JI_TILE_ID}, 2); // not an opening follow
    auto result = f.settle();
    assert((totals(result) == std::array<int, 4>{15, 0, 18, 0}));
    assert(result.chicken->responsibility_chickens.empty());
    RoundResult round;
    round.reason = RoundEndReason::ExhaustiveDraw;
    auto draw = RoundSettlementCalculator::calculate(round, {}, {}, std::nullopt, {});
    attribute_chicken_settlement(draw, f.history, f.discards);
    assert(draw.chicken_attributions.empty());
    assert((draw.chicken_unattributed_deltas == std::array<int, 4>{}));

    f.history[0].sequence = 99;
    bool threw = false;
    try { attribute_chicken_settlement(result, f.history, f.discards); }
    catch (const std::logic_error&) { threw = true; }
    assert(threw);
}

void test_gold_batong_burn_and_unrelated_flipped() {
    Fixture f;
    f.discard(0, Tile{BA_TONG_TILE_ID});
    f.discard(0, Tile{YAO_JI_TILE_ID}); // independent global first
    f.context.indicator = Tile{15};
    auto result = f.settle();
    assert((totals(result) == std::array<int, 4>{45, 0, 0, 0}));
    f.context.burned_players[0] = true;
    result = f.settle();
    assert((totals(result) == std::array<int, 4>{}));

    Fixture g;
    g.context.indicator = Tile{0}; // 2m is only a flipped, not fixed, chicken
    g.discard(0, Tile{1}, 1);
    result = g.settle();
    assert(result.chicken_attributions.empty());
    assert(result.chicken_unattributed_deltas[1] == 9);
}

void test_later_identical_discard_does_not_steal_responsibility() {
    Fixture f;
    f.discard(0, Tile{YAO_JI_TILE_ID}, 1);
    f.discard(0, Tile{YAO_JI_TILE_ID});
    auto result = f.settle();
    assert((totals(result) == std::array<int, 4>{-5, 23, 0, 0}));
    for (const auto& a : result.chicken_attributions) {
        assert(a.event_sequence == (a.player_id == 0 ? 0 : 3));
    }
    // Removing the applied call must fail instead of attaching to its proposal.
    f.history[3].applied = false;
    bool threw = false;
    try { attribute_chicken_settlement(result, f.history, f.discards); }
    catch (const std::logic_error&) { threw = true; }
    assert(threw);
}
} // namespace

int main() {
    test_chongfeng_and_negative_other_owner();
    test_responsibility_matrix();
    test_burned_and_daiminkan();
    test_follow_repeat_and_kakan();
    test_nonqualified_pon_and_absent_chicken();
    test_gold_batong_burn_and_unrelated_flipped();
    test_later_identical_discard_does_not_steal_responsibility();
}
