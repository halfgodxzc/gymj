#include "arena.h"
#include "search.h"
#include "action_encoder.h"
#include "action_feature_encoder.h"
#include "chicken.h"
#include "decision_feature_encoder.h"
#include <pybind11/functional.h>
#include "flush_progress.h"
#include "game.h"
#include "guiyang_scoring.h"
#include "observation_encoder.h"
#include "pattern_feature_encoder.h"
#include "privileged_pattern_encoder.h"
#include "privileged_state_encoder.h"
#include "round_settlement.h"
#include "shanten.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

#ifndef GYMJ_BINDING_PART
#define GYMJ_BINDING_PART 0
#endif

namespace {

#if GYMJ_BINDING_PART == 8
std::vector<GYMJ::Tile> make_tiles(const std::vector<uint8_t>& tile_ids) {
    std::vector<GYMJ::Tile> tiles;
    tiles.reserve(tile_ids.size());
    for (uint8_t tile_id : tile_ids) {
        if (tile_id >= GYMJ::TILE_TYPE_COUNT) {
            throw std::invalid_argument("tile id must be in range 0..26");
        }
        tiles.push_back(GYMJ::Tile{tile_id});
    }
    return tiles;
}

std::vector<uint8_t> hand_counts(const GYMJ::PlayerState& player) {
    return std::vector<uint8_t>(
        player.tehai,
        player.tehai + GYMJ::TILE_TYPE_COUNT
    );
}
#endif

#if GYMJ_BINDING_PART == 1 || GYMJ_BINDING_PART == 5
using ShantenHand = std::array<uint8_t, GYMJ::TILE_TYPE_COUNT>;

ShantenHand make_shanten_hand(const std::vector<int>& counts) {
    if (counts.size() != GYMJ::TILE_TYPE_COUNT) {
        throw std::invalid_argument("hand counts must contain exactly 27 entries");
    }

    ShantenHand hand{};
    for (std::size_t tile_id = 0; tile_id < counts.size(); ++tile_id) {
        const int count = counts[tile_id];
        if (count < 0 || count > GYMJ::MAX_TILE_COPIES) {
            throw std::invalid_argument("each hand count must be in range 0..4");
        }
        hand[tile_id] = static_cast<uint8_t>(count);
    }
    return hand;
}

void validate_meld_count(int meld_count) {
    if (meld_count < 0 || meld_count > GYMJ::COMPLETE_HAND_MELD_COUNT) {
        throw std::invalid_argument("meld count must be in range 0..4");
    }
}
#endif

#if GYMJ_BINDING_PART == 5
GYMJ::RL::ActionFeatureMatrix encode_action_features_from_arrays(
    const std::vector<int>& hand_counts,
    const std::vector<int>& visible_counts,
    const std::vector<int>& meld_tiles,
    const std::vector<int>& meld_owners,
    const std::vector<int>& meld_valid,
    int meld_count,
    const std::array<uint64_t, 2>& legal_mask_words,
    int current_discard
) {
    if (meld_tiles.size() != GYMJ::RL::MAX_MELDS ||
        meld_owners.size() != GYMJ::RL::MAX_MELDS ||
        meld_valid.size() != GYMJ::RL::MAX_MELDS ||
        meld_count < 0 || meld_count > GYMJ::RL::MAX_MELDS) {
        throw std::invalid_argument("meld arrays must have 16 entries and a valid count");
    }
    if (current_discard < 0 || current_discard > GYMJ::RL::NO_TILE) {
        throw std::invalid_argument("current discard is outside its valid range");
    }

    GYMJ::RL::CompactObservation observation;
    observation.hand_counts = make_shanten_hand(hand_counts);
    observation.visible_counts = make_shanten_hand(visible_counts);
    observation.current_discard = static_cast<uint8_t>(current_discard);
    observation.meld_count = static_cast<uint8_t>(meld_count);
    for (int index = 0; index < meld_count; ++index) {
        if (meld_valid[index] < 0 || meld_valid[index] > 1 ||
            meld_tiles[index] < 0 || meld_tiles[index] > GYMJ::RL::NO_TILE ||
            meld_owners[index] < 0 || meld_owners[index] > GYMJ::RL::NO_PLAYER ||
            (meld_valid[index] && meld_tiles[index] == GYMJ::RL::NO_TILE)) {
            throw std::invalid_argument("meld fields are outside their valid ranges");
        }
        auto& meld = observation.melds[index];
        meld.tile = static_cast<uint8_t>(meld_tiles[index]);
        meld.relative_owner = static_cast<uint8_t>(meld_owners[index]);
        meld.valid = static_cast<uint8_t>(meld_valid[index]);
    }
    return GYMJ::RL::ActionFeatureEncoder::encode(
        observation,
        legal_mask_words
    );
}
#endif

#if GYMJ_BINDING_PART == 1
py::dict select_flush_target_from_arrays(
    const std::vector<int>& hand_counts,
    const std::vector<int>& visible_counts,
    int meld_count,
    uint8_t fixed_meld_suit_mask,
    uint64_t seed,
    bool include_details
) {
    validate_meld_count(meld_count);
    const ShantenHand hand = make_shanten_hand(hand_counts);
    const ShantenHand visible = make_shanten_hand(visible_counts);
    const GYMJ::Algo::FlushTargetAnalysis analysis =
        GYMJ::Algo::analyze_flush_targets(
            hand.data(),
            visible.data(),
            meld_count,
            fixed_meld_suit_mask
        );

    py::dict result;
    result["best_suit"] = GYMJ::Algo::select_flush_target_suit(
        analysis,
        seed
    );
    if (!include_details) return result;

    result["best_suit_mask"] = analysis.best_suit_mask;
    py::list suits;
    for (size_t suit = 0; suit < GYMJ::Algo::FLUSH_SUIT_COUNT; ++suit) {
        const GYMJ::Algo::FlushSuitCandidate& candidate =
            analysis.suits[suit];
        py::dict details;
        details["suit"] = suit;
        details["shanten"] = candidate.shanten;
        details["concealed_tiles"] = candidate.concealed_tiles;
        details["effective_draw_kinds"] =
            candidate.effective_draws.kinds;
        details["effective_draw_copies"] =
            candidate.effective_draws.copies;
        details["meld_compatible"] =
            candidate.meld_compatible != 0;
        suits.append(std::move(details));
    }
    result["suits"] = std::move(suits);
    return result;
}
#endif

} // namespace

#if GYMJ_BINDING_PART == 0
namespace GymjPythonBindings {
void bind_algo(py::module_& module);
void bind_enums_and_chicken(py::module_& module);
void bind_scoring(py::module_& module);
void bind_core_types(py::module_& module);
void bind_action_and_pattern_features(py::module_& module);
void bind_decision_features(py::module_& module);
void bind_observations(py::module_& module);
void bind_arena(py::module_& module);
}

PYBIND11_MODULE(_gymj, module) {
    module.doc() = "Guiyang Mahjong C++ simulator";
    GymjPythonBindings::bind_algo(module);
    GymjPythonBindings::bind_enums_and_chicken(module);
    GymjPythonBindings::bind_scoring(module);
    GymjPythonBindings::bind_core_types(module);
    GymjPythonBindings::bind_action_and_pattern_features(module);
    GymjPythonBindings::bind_decision_features(module);
    GymjPythonBindings::bind_observations(module);
    GymjPythonBindings::bind_arena(module);
}
#elif GYMJ_BINDING_PART == 1
namespace GymjPythonBindings {
void bind_algo(py::module_& module) {
    module.attr("ACTION_ENCODING_VERSION") = GYMJ::RL::ACTION_ENCODING_VERSION;
    module.attr("ACTION_SPACE_SIZE") = GYMJ::RL::ACTION_SPACE_SIZE;
    module.attr("ACTION_FEATURE_VERSION") = GYMJ::RL::ACTION_FEATURE_VERSION;
    module.attr("ACTION_FEATURE_COUNT") = GYMJ::RL::ACTION_FEATURE_COUNT;
    module.attr("OBSERVATION_VERSION") = GYMJ::RL::OBSERVATION_VERSION;
    module.attr("PRIVILEGED_STATE_VERSION") =
        GYMJ::RL::PRIVILEGED_STATE_VERSION;
    module.attr("PATTERN_FEATURE_VERSION") =
        GYMJ::RL::PATTERN_FEATURE_VERSION;
    module.attr("DECISION_FEATURE_VERSION") =
        GYMJ::RL::DECISION_FEATURE_VERSION;
    module.attr("PRIVILEGED_PATTERN_VERSION") =
        GYMJ::RL::PRIVILEGED_PATTERN_VERSION;
    module.attr("MAX_PRIVILEGED_WALL_TILES") =
        GYMJ::RL::MAX_PRIVILEGED_WALL_TILES;
    module.attr("PRIVILEGED_FUTURE_DRAW_COUNT") =
        GYMJ::RL::PRIVILEGED_FUTURE_DRAW_COUNT;
    module.attr("PRIVILEGED_TILE_PAD") = GYMJ::RL::PRIVILEGED_TILE_PAD;
    module.attr("MAX_DISCARD_EVENTS") = GYMJ::RL::MAX_DISCARD_EVENTS;
    module.attr("MAX_MELDS") = GYMJ::RL::MAX_MELDS;
    module.attr("NO_TILE") = GYMJ::RL::NO_TILE;
    module.attr("NO_PLAYER") = GYMJ::RL::NO_PLAYER;
    module.attr("GUIYANG_HU_BASE_POINTS") =
        GYMJ::GUIYANG_HU_BASE_POINTS;
    module.attr("YAO_JI_TILE_ID") = GYMJ::YAO_JI_TILE_ID;
    module.attr("BA_TONG_TILE_ID") = GYMJ::BA_TONG_TILE_ID;
    module.attr("TILE_TYPE_COUNT") = GYMJ::TILE_TYPE_COUNT;
    module.attr("MAX_TILE_COPIES") = GYMJ::MAX_TILE_COPIES;
    module.attr("WAN_FLUSH_SUIT") = GYMJ::Algo::WAN_FLUSH_INDEX;
    module.attr("TONG_FLUSH_SUIT") = GYMJ::Algo::TONG_FLUSH_INDEX;
    module.attr("TIAO_FLUSH_SUIT") = GYMJ::Algo::TIAO_FLUSH_INDEX;
    module.attr("NO_FLUSH_SUIT") = GYMJ::Algo::NO_FLUSH_SUIT;
    module.attr("PATTERN_STANDARD") = static_cast<uint16_t>(
        GYMJ::RL::PatternStandard
    );
    module.attr("PATTERN_ALL_TRIPLETS") = static_cast<uint16_t>(
        GYMJ::RL::PatternAllTriplets
    );
    module.attr("PATTERN_SEVEN_PAIRS") = static_cast<uint16_t>(
        GYMJ::RL::PatternSevenPairs
    );
    module.attr("PATTERN_DRAGON_SEVEN_PAIRS") =
        static_cast<uint16_t>(GYMJ::RL::PatternDragonSevenPairs);
    module.attr("PATTERN_DI_LONG") = static_cast<uint16_t>(
        GYMJ::RL::PatternDiLong
    );
    module.attr("PATTERN_FLUSH") = static_cast<uint16_t>(
        GYMJ::RL::PatternFlush
    );
    module.attr("PATTERN_SINGLE_WAIT") = static_cast<uint16_t>(
        GYMJ::RL::PatternSingleWait
    );

    py::class_<GYMJ::Algo::SuitPattern>(module, "SuitPattern")
        .def_readonly("melds", &GYMJ::Algo::SuitPattern::melds)
        .def_readonly("taatsu", &GYMJ::Algo::SuitPattern::taatsu)
        .def_readonly("pairs", &GYMJ::Algo::SuitPattern::pairs);

    py::class_<GYMJ::Algo::ShantenResult>(module, "ShantenResult")
        .def_readonly("normal", &GYMJ::Algo::ShantenResult::normal)
        .def_readonly(
            "seven_pairs",
            &GYMJ::Algo::ShantenResult::seven_pairs
        )
        .def_readonly(
            "all_triplets",
            &GYMJ::Algo::ShantenResult::all_triplets
        )
        .def_readonly("best", &GYMJ::Algo::ShantenResult::best);

    py::class_<GYMJ::Algo::ShantenAnalysis>(module, "ShantenAnalysis")
        .def_readonly("result", &GYMJ::Algo::ShantenAnalysis::result)
        .def_readonly("wan", &GYMJ::Algo::ShantenAnalysis::wan)
        .def_readonly("tong", &GYMJ::Algo::ShantenAnalysis::tong)
        .def_readonly("tiao", &GYMJ::Algo::ShantenAnalysis::tiao);

    module.def(
        "calculate_normal_shanten",
        [](const std::vector<int>& counts, int meld_count) {
            validate_meld_count(meld_count);
            const ShantenHand hand = make_shanten_hand(counts);
            return GYMJ::Algo::calculate_normal_shanten(
                hand.data(),
                meld_count
            );
        },
        py::arg("hand_counts"),
        py::arg("meld_count") = 0
    );
    module.def(
        "calculate_seven_pairs_shanten",
        [](const std::vector<int>& counts, int meld_count) {
            validate_meld_count(meld_count);
            const ShantenHand hand = make_shanten_hand(counts);
            return GYMJ::Algo::calculate_seven_pairs_shanten(
                hand.data(),
                meld_count
            );
        },
        py::arg("hand_counts"),
        py::arg("meld_count") = 0
    );
    module.def(
        "calculate_all_triplets_shanten",
        [](const std::vector<int>& counts, int meld_count) {
            validate_meld_count(meld_count);
            const ShantenHand hand = make_shanten_hand(counts);
            return GYMJ::Algo::calculate_all_triplets_shanten(
                hand.data(),
                meld_count
            );
        },
        py::arg("hand_counts"),
        py::arg("meld_count") = 0
    );
    module.def(
        "calculate_shanten",
        [](const std::vector<int>& counts, int meld_count) {
            validate_meld_count(meld_count);
            const ShantenHand hand = make_shanten_hand(counts);
            return GYMJ::Algo::calculate_shanten(
                hand.data(),
                meld_count
            );
        },
        py::arg("hand_counts"),
        py::arg("meld_count") = 0
    );
    module.def(
        "analyze_shanten",
        [](const std::vector<int>& counts, int meld_count) {
            validate_meld_count(meld_count);
            const ShantenHand hand = make_shanten_hand(counts);
            return GYMJ::Algo::analyze_shanten(
                hand.data(),
                meld_count
            );
        },
        py::arg("hand_counts"),
        py::arg("meld_count") = 0
    );
    module.def(
        "select_flush_target",
        &select_flush_target_from_arrays,
        py::arg("hand_counts"),
        py::arg("visible_counts"),
        py::arg("meld_count") = 0,
        py::arg("fixed_meld_suit_mask") = 0,
        py::arg("seed") = 0,
        py::arg("include_details") = false
    );
    module.def(
        "set_shanten_logging",
        &GYMJ::Algo::set_shanten_logging,
        py::arg("enabled")
    );
    module.def(
        "is_shanten_logging_enabled",
        &GYMJ::Algo::is_shanten_logging_enabled
    );

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 2
namespace GymjPythonBindings {
void bind_enums_and_chicken(py::module_& module) {

    py::enum_<GYMJ::ActionType>(module, "ActionType")
        .value("PASS", GYMJ::ActionType::Pass)
        .value("DRAW", GYMJ::ActionType::Draw)
        .value("DISCARD", GYMJ::ActionType::Discard)
        .value("PON", GYMJ::ActionType::Pon)
        .value("ANKAN", GYMJ::ActionType::AnKan)
        .value("KAKAN", GYMJ::ActionType::Kakan)
        .value("DAIMINKAN", GYMJ::ActionType::Daiminkan)
        .value("TSUMO", GYMJ::ActionType::Tsumo)
        .value("RON", GYMJ::ActionType::Ron)
        .value("DECLARE_READY", GYMJ::ActionType::DeclareReady);

    py::enum_<GYMJ::MeldType>(module, "MeldType")
        .value("PON", GYMJ::MeldType::Pon)
        .value("ANKAN", GYMJ::MeldType::AnKan)
        .value("KAKAN", GYMJ::MeldType::Kakan)
        .value("DAIMINKAN", GYMJ::MeldType::Daiminkan);

    py::enum_<GYMJ::RoundPhase>(module, "RoundPhase")
        .value("NOT_STARTED", GYMJ::RoundPhase::NotStarted)
        .value("AWAIT_TURN_ACTION", GYMJ::RoundPhase::AwaitTurnAction)
        .value(
            "AWAIT_READY_DECLARATION",
            GYMJ::RoundPhase::AwaitReadyDeclaration
        )
        .value("AWAIT_DISCARD_RESPONSES", GYMJ::RoundPhase::AwaitDiscardResponses)
        .value("AWAIT_KAKAN_RESPONSES", GYMJ::RoundPhase::AwaitKakanResponses)
        .value("REPLACEMENT_DRAW", GYMJ::RoundPhase::ReplacementDraw)
        .value("FINISHED", GYMJ::RoundPhase::Finished);

    py::enum_<GYMJ::RoundEndReason>(module, "RoundEndReason")
        .value("NONE", GYMJ::RoundEndReason::None)
        .value("TSUMO", GYMJ::RoundEndReason::Tsumo)
        .value("RON", GYMJ::RoundEndReason::Ron)
        .value("QIANG_GANG", GYMJ::RoundEndReason::QiangGang)
        .value("EXHAUSTIVE_DRAW", GYMJ::RoundEndReason::ExhaustiveDraw);

    py::enum_<GYMJ::GamePhase>(module, "GamePhase")
        .value("NOT_STARTED", GYMJ::GamePhase::NotStarted)
        .value("PLAYING_HAND", GYMJ::GamePhase::PlayingHand)
        .value("BETWEEN_HANDS", GYMJ::GamePhase::BetweenHands)
        .value("FINISHED", GYMJ::GamePhase::Finished);

    py::enum_<GYMJ::ReadyKind>(module, "ReadyKind")
        .value("NONE", GYMJ::ReadyKind::None)
        .value("TIAN_TING", GYMJ::ReadyKind::TianTing)
        .value("BAO_TING", GYMJ::ReadyKind::BaoTing);

    py::class_<GYMJ::ReadyState>(module, "ReadyState")
        .def_readonly("kind", &GYMJ::ReadyState::kind)
        .def_readonly("locked_waits", &GYMJ::ReadyState::locked_waits)
        .def_property_readonly("declared", &GYMJ::ReadyState::declared);

    py::enum_<GYMJ::ChickenKind>(module, "ChickenKind")
        .value("FLIPPED", GYMJ::ChickenKind::Flipped)
        .value("YAO_JI", GYMJ::ChickenKind::YaoJi)
        .value("BA_TONG", GYMJ::ChickenKind::BaTong)
        .value(
            "CHONGFENG_YAO_JI",
            GYMJ::ChickenKind::ChongfengYaoJi
        )
        .value(
            "CHONGFENG_BA_TONG",
            GYMJ::ChickenKind::ChongfengBaTong
        );

    py::enum_<GYMJ::ScoreCategory>(module, "ScoreCategory")
        .value("HU", GYMJ::ScoreCategory::Hu)
        .value("CHA_JIAO", GYMJ::ScoreCategory::ChaJiao)
        .value("CHICKEN", GYMJ::ScoreCategory::Chicken)
        .value(
            "RESPONSIBILITY_CHICKEN",
            GYMJ::ScoreCategory::ResponsibilityChicken
        )
        .value("ANKAN_BEAN", GYMJ::ScoreCategory::AnKanBean)
        .value("KAKAN_BEAN", GYMJ::ScoreCategory::KakanBean)
        .value("DAIMINKAN_BEAN", GYMJ::ScoreCategory::DaiminkanBean);

    py::enum_<GYMJ::BeanKind>(module, "BeanKind")
        .value("ANKAN", GYMJ::BeanKind::AnKan)
        .value("KAKAN", GYMJ::BeanKind::Kakan)
        .value("DAIMINKAN", GYMJ::BeanKind::Daiminkan);

    py::class_<GYMJ::ChickenTileCounts>(module, "ChickenTileCounts")
        .def_readonly("hand", &GYMJ::ChickenTileCounts::hand)
        .def_readonly("discards", &GYMJ::ChickenTileCounts::discards)
        .def_readonly("melds", &GYMJ::ChickenTileCounts::melds)
        .def_property_readonly("total", &GYMJ::ChickenTileCounts::total);

    py::class_<GYMJ::ChickenItem>(module, "ChickenItem")
        .def_readonly("kind", &GYMJ::ChickenItem::kind)
        .def_readonly("tile", &GYMJ::ChickenItem::tile)
        .def_readonly("counts", &GYMJ::ChickenItem::counts)
        .def_readonly("is_gold", &GYMJ::ChickenItem::is_gold)
        .def_readonly(
            "points_per_tile",
            &GYMJ::ChickenItem::points_per_tile
        )
        .def_property_readonly(
            "tile_count",
            &GYMJ::ChickenItem::tile_count
        )
        .def_property_readonly(
            "total_points",
            &GYMJ::ChickenItem::total_points
        );

    py::class_<GYMJ::PlayerChickenResult>(
        module,
        "PlayerChickenResult"
    )
        .def_readonly("player_id", &GYMJ::PlayerChickenResult::player_id)
        .def_readonly("is_jiao", &GYMJ::PlayerChickenResult::is_jiao)
        .def_readonly("is_burned", &GYMJ::PlayerChickenResult::is_burned)
        .def_readonly("items", &GYMJ::PlayerChickenResult::items)
        .def_readonly(
            "total_tile_count",
            &GYMJ::PlayerChickenResult::total_tile_count
        )
        .def_readonly(
            "raw_total_points",
            &GYMJ::PlayerChickenResult::raw_total_points
        )
        .def_readonly(
            "total_points",
            &GYMJ::PlayerChickenResult::total_points
        )
        .def_property_readonly(
            "has_any_chicken",
            &GYMJ::PlayerChickenResult::has_any_chicken
        );

    py::class_<GYMJ::ResponsibilityChickenResult>(
        module,
        "ResponsibilityChickenResult"
    )
        .def_readonly(
            "source_player",
            &GYMJ::ResponsibilityChickenResult::source_player
        )
        .def_readonly(
            "owner_player",
            &GYMJ::ResponsibilityChickenResult::owner_player
        )
        .def_readonly("tile", &GYMJ::ResponsibilityChickenResult::tile)
        .def_readonly(
            "is_gold",
            &GYMJ::ResponsibilityChickenResult::is_gold
        )
        .def_readonly("points", &GYMJ::ResponsibilityChickenResult::points);

    py::class_<GYMJ::ChickenResult>(module, "ChickenResult")
        .def_readonly("indicator", &GYMJ::ChickenResult::indicator)
        .def_readonly(
            "flipped_chicken",
            &GYMJ::ChickenResult::flipped_chicken
        )
        .def_readonly("players", &GYMJ::ChickenResult::players)
        .def_readonly(
            "responsibility_chickens",
            &GYMJ::ChickenResult::responsibility_chickens
        )
        .def(
            "player",
            &GYMJ::ChickenResult::player,
            py::return_value_policy::reference_internal
        );

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 3
namespace GymjPythonBindings {
void bind_scoring(py::module_& module) {

    py::enum_<GYMJ::HuPattern>(module, "HuPattern")
        .value("STANDARD", GYMJ::HuPattern::Standard)
        .value("ALL_TRIPLETS", GYMJ::HuPattern::AllTriplets)
        .value("SEVEN_PAIRS", GYMJ::HuPattern::SevenPairs)
        .value(
            "DRAGON_SEVEN_PAIRS",
            GYMJ::HuPattern::DragonSevenPairs
        )
        .value("DI_LONG", GYMJ::HuPattern::DiLong);

    py::enum_<GYMJ::HuTiming>(module, "HuTiming")
        .value("NORMAL", GYMJ::HuTiming::Normal)
        .value("DI_HU", GYMJ::HuTiming::DiHu)
        .value("TIAN_HU", GYMJ::HuTiming::TianHu);

    py::enum_<GYMJ::SpecialHu>(module, "SpecialHu")
        .value("NONE", GYMJ::SpecialHu::None)
        .value("GANG_KAI", GYMJ::SpecialHu::GangKai)
        .value("RE_PAO", GYMJ::SpecialHu::RePao)
        .value("QIANG_GANG", GYMJ::SpecialHu::QiangGang)
        .value("SHA_BAO", GYMJ::SpecialHu::ShaBao);

    py::enum_<GYMJ::FanType>(module, "FanType")
        .value("PING_HU", GYMJ::FanType::PingHu)
        .value("DA_DUI_ZI", GYMJ::FanType::DaDuiZi)
        .value("SU_GUA_DOU", GYMJ::FanType::SuGuaDou)
        .value("QI_DUI", GYMJ::FanType::QiDui)
        .value("QING_YI_SE", GYMJ::FanType::QingYiSe)
        .value("QING_DAN_DIAO", GYMJ::FanType::QingDanDiao)
        .value("LONG_QI_DUI", GYMJ::FanType::LongQiDui)
        .value("DI_LONG", GYMJ::FanType::DiLong)
        .value("BAO_TING", GYMJ::FanType::BaoTing)
        .value("TIAN_TING", GYMJ::FanType::TianTing)
        .value("GANG_KAI", GYMJ::FanType::GangKai)
        .value("RE_PAO", GYMJ::FanType::RePao)
        .value("QIANG_GANG", GYMJ::FanType::QiangGang)
        .value("SHA_BAO", GYMJ::FanType::ShaBao)
        .value("DI_HU", GYMJ::FanType::DiHu)
        .value("TIAN_HU", GYMJ::FanType::TianHu);

    py::class_<GYMJ::HuFeatures>(module, "HuFeatures")
        .def(py::init<>())
        .def_readwrite("pattern", &GYMJ::HuFeatures::pattern)
        .def_readwrite("is_flush", &GYMJ::HuFeatures::is_flush)
        .def_readwrite(
            "is_single_wait",
            &GYMJ::HuFeatures::is_single_wait
        )
        .def_readwrite("ready_kind", &GYMJ::HuFeatures::ready_kind)
        .def_property(
            "is_reported",
            [](const GYMJ::HuFeatures& features) {
                return features.ready_kind != GYMJ::ReadyKind::None;
            },
            [](GYMJ::HuFeatures& features, bool reported) {
                features.ready_kind = reported
                    ? GYMJ::ReadyKind::BaoTing
                    : GYMJ::ReadyKind::None;
            }
        )
        .def_readwrite(
            "is_su_gua_dou",
            &GYMJ::HuFeatures::is_su_gua_dou
        )
        .def_readwrite("timing", &GYMJ::HuFeatures::timing)
        .def_readwrite("special", &GYMJ::HuFeatures::special);

    py::class_<GYMJ::FanItem>(module, "FanItem")
        .def_readonly("type", &GYMJ::FanItem::type)
        .def_readonly("fan", &GYMJ::FanItem::fan);

    py::class_<GYMJ::HuScore>(module, "HuScore")
        .def_readonly("shape_fan", &GYMJ::HuScore::shape_fan)
        .def_readonly("ready_fan", &GYMJ::HuScore::ready_fan)
        .def_readonly("timing_fan", &GYMJ::HuScore::timing_fan)
        .def_readonly("special_fan", &GYMJ::HuScore::special_fan)
        .def_readonly("total_fan", &GYMJ::HuScore::total_fan)
        .def_readonly("hu_points", &GYMJ::HuScore::hu_points)
        .def_readonly("items", &GYMJ::HuScore::items);

    py::class_<GYMJ::ScoreTransfer>(module, "ScoreTransfer")
        .def_readonly("chicken_owner", &GYMJ::ScoreTransfer::chicken_owner)
        .def_readonly("category", &GYMJ::ScoreTransfer::category)
        .def_readonly("from_player", &GYMJ::ScoreTransfer::from)
        .def_readonly("to_player", &GYMJ::ScoreTransfer::to)
        .def_readonly("points", &GYMJ::ScoreTransfer::points)
        .def_readonly("tile", &GYMJ::ScoreTransfer::tile);

    py::class_<GYMJ::ScoreLedger>(module, "ScoreLedger")
        .def_property_readonly(
            "transfers",
            &GYMJ::ScoreLedger::transfers,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly(
            "score_deltas",
            &GYMJ::ScoreLedger::score_deltas,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly("net_total", &GYMJ::ScoreLedger::net_total);

    py::class_<GYMJ::BeanItem>(module, "BeanItem")
        .def_readonly("kind", &GYMJ::BeanItem::kind)
        .def_readonly("owner", &GYMJ::BeanItem::owner)
        .def_readonly("source", &GYMJ::BeanItem::source)
        .def_readonly("tile", &GYMJ::BeanItem::tile)
        .def_readonly("owner_is_jiao", &GYMJ::BeanItem::owner_is_jiao)
        .def_readonly(
            "points_per_payer",
            &GYMJ::BeanItem::points_per_payer
        );

    py::class_<GYMJ::BeanResult>(module, "BeanResult")
        .def_readonly("items", &GYMJ::BeanResult::items)
        .def_readonly("ledger", &GYMJ::BeanResult::ledger);

    py::class_<GYMJ::HuSettlement>(module, "HuSettlement")
        .def_readonly("player_id", &GYMJ::HuSettlement::player_id)
        .def_readonly("features", &GYMJ::HuSettlement::features)
        .def_readonly("score", &GYMJ::HuSettlement::score);

    py::class_<GYMJ::ChaJiaoSettlement>(module, "ChaJiaoSettlement")
        .def_readonly("player_id", &GYMJ::ChaJiaoSettlement::player_id)
        .def_readonly("best_wait", &GYMJ::ChaJiaoSettlement::best_wait)
        .def_readonly("features", &GYMJ::ChaJiaoSettlement::features)
        .def_readonly("score", &GYMJ::ChaJiaoSettlement::score);

    py::enum_<GYMJ::ChickenAttributionKind>(module, "ChickenAttributionKind")
        .value("CHONGFENG", GYMJ::ChickenAttributionKind::Chongfeng)
        .value("PON", GYMJ::ChickenAttributionKind::Pon)
        .value("RESPONSIBILITY", GYMJ::ChickenAttributionKind::Responsibility);
    py::class_<GYMJ::ChickenAttribution>(module, "ChickenAttribution")
        .def_readonly("player_id", &GYMJ::ChickenAttribution::player_id)
        .def_readonly("event_sequence", &GYMJ::ChickenAttribution::event_sequence)
        .def_readonly("transfer_index", &GYMJ::ChickenAttribution::transfer_index)
        .def_readonly("kind", &GYMJ::ChickenAttribution::kind)
        .def_readonly("tile", &GYMJ::ChickenAttribution::tile)
        .def_readonly("points", &GYMJ::ChickenAttribution::points);

    py::class_<GYMJ::RoundSettlementResult>(
        module,
        "RoundSettlementResult"
    )
        .def_readonly(
            "hu_settlements",
            &GYMJ::RoundSettlementResult::hu_settlements
        )
        .def_readonly(
            "cha_jiao_settlements",
            &GYMJ::RoundSettlementResult::cha_jiao_settlements
        )
        .def_readonly("chicken", &GYMJ::RoundSettlementResult::chicken)
        .def_readonly("beans", &GYMJ::RoundSettlementResult::beans)
        .def_readonly("chicken_attributions", &GYMJ::RoundSettlementResult::chicken_attributions)
        .def_readonly("chicken_unattributed_deltas", &GYMJ::RoundSettlementResult::chicken_unattributed_deltas)
        .def_readonly("ledger", &GYMJ::RoundSettlementResult::ledger);

    py::class_<GYMJ::GuiyangScoring>(module, "GuiyangScoring")
        .def_static("calculate", &GYMJ::GuiyangScoring::calculate);

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 4
namespace GymjPythonBindings {
void bind_core_types(py::module_& module) {

    py::class_<GYMJ::Tile>(module, "Tile")
        .def_property_readonly("id", [](const GYMJ::Tile& tile) {
            return tile.id;
        })
        .def_property_readonly("suit", &GYMJ::Tile::suit)
        .def_property_readonly("rank", &GYMJ::Tile::rank)
        .def("__repr__", [](const GYMJ::Tile& tile) {
            return "Tile('" + tile.to_string() + "')";
        });

    py::class_<GYMJ::Action>(module, "Action")
        .def_readonly("type", &GYMJ::Action::type)
        .def_readonly("actor", &GYMJ::Action::actor)
        .def_readonly("tile", &GYMJ::Action::tile)
        .def_readonly("target", &GYMJ::Action::target);

    py::class_<GYMJ::Meld>(module, "Meld")
        .def_readonly("type", &GYMJ::Meld::type)
        .def_readonly("tile", &GYMJ::Meld::tile)
        .def_readonly("source", &GYMJ::Meld::source);

    py::class_<GYMJ::DecisionContext>(module, "DecisionContext")
        .def_readonly("phase", &GYMJ::DecisionContext::phase)
        .def_readonly("player", &GYMJ::DecisionContext::player)
        .def_readonly("legal_actions", &GYMJ::DecisionContext::legal_actions);

    py::class_<GYMJ::RoundResult>(module, "RoundResult")
        .def_readonly("reason", &GYMJ::RoundResult::reason)
        .def_readonly("hu_players", &GYMJ::RoundResult::hu_players)
        .def_readonly(
            "fangpao_player",
            &GYMJ::RoundResult::fangpao_player
        )
        .def_readonly("hu_tile", &GYMJ::RoundResult::hu_tile)
        .def_readonly("is_hot_cannon", &GYMJ::RoundResult::is_hot_cannon);

    py::class_<GYMJ::GameConfig>(module, "GameConfig")
        .def(py::init<>())
        .def_readwrite("seed", &GYMJ::GameConfig::seed)
        .def_readwrite("initial_dealer", &GYMJ::GameConfig::initial_dealer)
        .def_readwrite("max_hands", &GYMJ::GameConfig::max_hands)
        .def_readwrite("initial_scores", &GYMJ::GameConfig::initial_scores);

    py::class_<GYMJ::DealerRotation>(module, "DealerRotation")
        .def_static(
            "next_dealer",
            &GYMJ::DealerRotation::next_dealer,
            py::arg("current_dealer"),
            py::arg("result"),
            py::arg("is_jiao")
        );

    py::class_<GYMJ::HandRecord>(module, "HandRecord")
        .def_readonly("hand_index", &GYMJ::HandRecord::hand_index)
        .def_readonly("dealer", &GYMJ::HandRecord::dealer)
        .def_readonly(
            "dealer_continuation",
            &GYMJ::HandRecord::dealer_continuation
        )
        .def_readonly("round_result", &GYMJ::HandRecord::round_result)
        .def_readonly("settlement", &GYMJ::HandRecord::settlement)
        .def_readonly("scores_after", &GYMJ::HandRecord::scores_after)
        .def_readonly("next_dealer", &GYMJ::HandRecord::next_dealer)
        .def_readonly(
            "dealer_continues",
            &GYMJ::HandRecord::dealer_continues
        );

    py::class_<GYMJ::GameEvent>(module, "GameEvent")
        .def_readonly("sequence", &GYMJ::GameEvent::sequence)
        .def_readonly("action", &GYMJ::GameEvent::action)
        .def_readonly("phase_before", &GYMJ::GameEvent::phase_before)
        .def_readonly("phase_after", &GYMJ::GameEvent::phase_after)
        .def_readonly("applied", &GYMJ::GameEvent::applied);

    py::class_<GYMJ::DiscardEvent>(module, "DiscardEvent")
        .def_readonly("tile", &GYMJ::DiscardEvent::tile)
        .def_readonly("actor", &GYMJ::DiscardEvent::actor)
        .def_readonly("was_called", &GYMJ::DiscardEvent::was_called)
        .def_readonly("called_by", &GYMJ::DiscardEvent::called_by);

    py::class_<GYMJ::ObservedDiscardEvent>(module, "ObservedDiscardEvent")
        .def_readonly("tile", &GYMJ::ObservedDiscardEvent::tile)
        .def_readonly(
            "relative_actor",
            &GYMJ::ObservedDiscardEvent::relative_actor
        )
        .def_readonly(
            "was_called",
            &GYMJ::ObservedDiscardEvent::was_called
        );

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 5
namespace GymjPythonBindings {
void bind_action_and_pattern_features(py::module_& module) {

    py::class_<GYMJ::RL::EncodedDecision>(module, "EncodedDecision")
        .def("is_legal", &GYMJ::RL::EncodedDecision::is_legal)
        .def("action_at", [](const GYMJ::RL::EncodedDecision& decision, size_t action_id) {
            return decision.action_at(action_id);
        })
        .def_property_readonly(
            "legal_mask",
            &GYMJ::RL::EncodedDecision::expanded_mask
        )
        .def_property_readonly(
            "legal_action_ids",
            &GYMJ::RL::EncodedDecision::legal_action_ids
        )
        .def_property_readonly(
            "mask_words",
            &GYMJ::RL::EncodedDecision::mask_words
        );

    py::class_<GYMJ::RL::ActionEncoder>(module, "ActionEncoder")
        .def_static("encode_action", &GYMJ::RL::ActionEncoder::encode_action)
        .def_static(
            "encode_legal_actions",
            &GYMJ::RL::ActionEncoder::encode_legal_actions
        );

    py::enum_<GYMJ::RL::ActionFeatureIndex>(module, "ActionFeatureIndex")
        .value("POST_STATE_KNOWN", GYMJ::RL::ActionFeatureIndex::PostStateKnown)
        .value("REQUIRES_FOLLOW_UP_DISCARD", GYMJ::RL::ActionFeatureIndex::RequiresFollowUpDiscard)
        .value("MELD_COUNT_DELTA", GYMJ::RL::ActionFeatureIndex::MeldCountDelta)
        .value("NORMAL_SHANTEN", GYMJ::RL::ActionFeatureIndex::NormalShanten)
        .value("SEVEN_PAIRS_SHANTEN", GYMJ::RL::ActionFeatureIndex::SevenPairsShanten)
        .value("ALL_TRIPLETS_SHANTEN", GYMJ::RL::ActionFeatureIndex::AllTripletsShanten)
        .value("NORMAL_EFFECTIVE_DRAW_KINDS", GYMJ::RL::ActionFeatureIndex::NormalEffectiveDrawKinds)
        .value("SEVEN_PAIRS_EFFECTIVE_DRAW_KINDS", GYMJ::RL::ActionFeatureIndex::SevenPairsEffectiveDrawKinds)
        .value("ALL_TRIPLETS_EFFECTIVE_DRAW_KINDS", GYMJ::RL::ActionFeatureIndex::AllTripletsEffectiveDrawKinds)
        .value("NORMAL_EFFECTIVE_DRAW_COPIES", GYMJ::RL::ActionFeatureIndex::NormalEffectiveDrawCopies)
        .value("SEVEN_PAIRS_EFFECTIVE_DRAW_COPIES", GYMJ::RL::ActionFeatureIndex::SevenPairsEffectiveDrawCopies)
        .value("ALL_TRIPLETS_EFFECTIVE_DRAW_COPIES", GYMJ::RL::ActionFeatureIndex::AllTripletsEffectiveDrawCopies)
        .value("WAN_FLUSH_OFF_SUIT_TILES", GYMJ::RL::ActionFeatureIndex::WanFlushOffSuitTiles)
        .value("TONG_FLUSH_OFF_SUIT_TILES", GYMJ::RL::ActionFeatureIndex::TongFlushOffSuitTiles)
        .value("TIAO_FLUSH_OFF_SUIT_TILES", GYMJ::RL::ActionFeatureIndex::TiaoFlushOffSuitTiles)
        .value("WAN_FLUSH_MELD_COMPATIBLE", GYMJ::RL::ActionFeatureIndex::WanFlushMeldCompatible)
        .value("TONG_FLUSH_MELD_COMPATIBLE", GYMJ::RL::ActionFeatureIndex::TongFlushMeldCompatible)
        .value("TIAO_FLUSH_MELD_COMPATIBLE", GYMJ::RL::ActionFeatureIndex::TiaoFlushMeldCompatible)
        .value("CONCEALED_TILES_CONSUMED", GYMJ::RL::ActionFeatureIndex::ConcealedTilesConsumed)
        .value("TAKES_REPLACEMENT_DRAW", GYMJ::RL::ActionFeatureIndex::TakesReplacementDraw);

    py::class_<GYMJ::RL::ActionFeatureEncoder>(module, "ActionFeatureEncoder")
        .def_static(
            "encode",
            &GYMJ::RL::ActionFeatureEncoder::encode,
            py::arg("observation"),
            py::arg("legal_mask_words")
        )
        .def_static(
            "encode_from_arrays",
            &encode_action_features_from_arrays,
            py::arg("hand_counts"),
            py::arg("visible_counts"),
            py::arg("meld_tiles"),
            py::arg("meld_owners"),
            py::arg("meld_valid"),
            py::arg("meld_count"),
            py::arg("legal_mask_words"),
            py::arg("current_discard") = GYMJ::RL::NO_TILE
        );

    py::enum_<GYMJ::RL::PatternRoute>(module, "PatternRoute")
        .value("NORMAL", GYMJ::RL::PatternRoute::Normal)
        .value("SEVEN_PAIRS", GYMJ::RL::PatternRoute::SevenPairs)
        .value("ALL_TRIPLETS", GYMJ::RL::PatternRoute::AllTriplets);

    py::class_<GYMJ::Algo::EffectiveDrawCount>(
        module,
        "EffectiveDrawCount"
    )
        .def_readonly("kinds", &GYMJ::Algo::EffectiveDrawCount::kinds)
        .def_readonly("copies", &GYMJ::Algo::EffectiveDrawCount::copies)
        .def_readonly(
            "remaining_by_tile",
            &GYMJ::Algo::EffectiveDrawCount::remaining_by_tile
        );

    py::class_<GYMJ::RL::RoutePatternFeatures>(
        module,
        "RoutePatternFeatures"
    )
        .def_readonly("shanten", &GYMJ::RL::RoutePatternFeatures::shanten)
        .def_readonly(
            "effective_draws",
            &GYMJ::RL::RoutePatternFeatures::effective_draws
        );

    py::class_<GYMJ::RL::FlushPatternFeatures>(
        module,
        "FlushPatternFeatures"
    )
        .def_readonly("shanten", &GYMJ::RL::FlushPatternFeatures::shanten)
        .def_readonly(
            "concealed_tiles",
            &GYMJ::RL::FlushPatternFeatures::concealed_tiles
        )
        .def_readonly(
            "off_suit_tiles",
            &GYMJ::RL::FlushPatternFeatures::off_suit_tiles
        )
        .def_readonly(
            "meld_compatible",
            &GYMJ::RL::FlushPatternFeatures::meld_compatible
        )
        .def_readonly(
            "effective_draws",
            &GYMJ::RL::FlushPatternFeatures::effective_draws
        );

    py::class_<GYMJ::RL::WaitValueFeatures>(module, "WaitValueFeatures")
        .def_readonly(
            "pattern_mask",
            &GYMJ::RL::WaitValueFeatures::pattern_mask
        )
        .def_readonly("base_fan", &GYMJ::RL::WaitValueFeatures::base_fan)
        .def_readonly(
            "base_hu_points",
            &GYMJ::RL::WaitValueFeatures::base_hu_points
        )
        .def_readonly(
            "remaining_copies",
            &GYMJ::RL::WaitValueFeatures::remaining_copies
        );

    py::class_<GYMJ::RL::HandPatternFeatures>(
        module,
        "HandPatternFeatures"
    )
        .def_readonly(
            "post_discard_state",
            &GYMJ::RL::HandPatternFeatures::post_discard_state
        )
        .def_readonly("routes", &GYMJ::RL::HandPatternFeatures::routes)
        .def_readonly("flushes", &GYMJ::RL::HandPatternFeatures::flushes)
        .def_readonly(
            "best_flush_suit_mask",
            &GYMJ::RL::HandPatternFeatures::best_flush_suit_mask
        )
        .def_readonly(
            "shape_wait_mask",
            &GYMJ::RL::HandPatternFeatures::shape_wait_mask
        )
        .def_readonly(
            "live_wait_mask",
            &GYMJ::RL::HandPatternFeatures::live_wait_mask
        )
        .def_readonly(
            "live_wait_kinds",
            &GYMJ::RL::HandPatternFeatures::live_wait_kinds
        )
        .def_readonly(
            "live_wait_copies",
            &GYMJ::RL::HandPatternFeatures::live_wait_copies
        )
        .def_readonly(
            "wait_values",
            &GYMJ::RL::HandPatternFeatures::wait_values
        );

    py::class_<GYMJ::RL::PublicPlayerFeatures>(
        module,
        "PublicPlayerFeatures"
    )
        .def_readonly(
            "river_counts",
            &GYMJ::RL::PublicPlayerFeatures::river_counts
        )
        .def_readonly(
            "called_discard_counts",
            &GYMJ::RL::PublicPlayerFeatures::called_discard_counts
        )
        .def_readonly(
            "meld_tile_counts",
            &GYMJ::RL::PublicPlayerFeatures::meld_tile_counts
        )
        .def_readonly(
            "public_owned_tile_counts",
            &GYMJ::RL::PublicPlayerFeatures::public_owned_tile_counts
        )
        .def_readonly(
            "meld_type_counts",
            &GYMJ::RL::PublicPlayerFeatures::meld_type_counts
        )
        .def_readonly(
            "bean_type_counts",
            &GYMJ::RL::PublicPlayerFeatures::bean_type_counts
        )
        .def_readonly(
            "discard_count",
            &GYMJ::RL::PublicPlayerFeatures::discard_count
        )
        .def_readonly(
            "meld_count",
            &GYMJ::RL::PublicPlayerFeatures::meld_count
        )
        .def_readonly(
            "fixed_chicken_count",
            &GYMJ::RL::PublicPlayerFeatures::fixed_chicken_count
        )
        .def_readonly(
            "known_tenpai_status",
            &GYMJ::RL::PublicPlayerFeatures::known_tenpai_status
        )
        .def_readonly(
            "signed_fixed_chicken_count",
            &GYMJ::RL::PublicPlayerFeatures::signed_fixed_chicken_count
        )
        .def_readonly(
            "chongfeng_chicken_count",
            &GYMJ::RL::PublicPlayerFeatures::chongfeng_chicken_count
        )
        .def_readonly(
            "responsibility_chicken_given_count",
            &GYMJ::RL::PublicPlayerFeatures::responsibility_chicken_given_count
        )
        .def_readonly(
            "responsibility_chicken_received_count",
            &GYMJ::RL::PublicPlayerFeatures::responsibility_chicken_received_count
        );

    py::class_<GYMJ::RL::PublicPatternFeatures>(
        module,
        "PublicPatternFeatures"
    )
        .def_readonly("hand", &GYMJ::RL::PublicPatternFeatures::hand)
        .def_readonly("players", &GYMJ::RL::PublicPatternFeatures::players);

    py::class_<GYMJ::RL::PatternFeatureEncoder>(
        module,
        "PatternFeatureEncoder"
    )
        .def_static("encode_hand", &GYMJ::RL::PatternFeatureEncoder::encode_hand)
        .def_static("encode", &GYMJ::RL::PatternFeatureEncoder::encode);

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 6
namespace GymjPythonBindings {
void bind_decision_features(py::module_& module) {

    py::enum_<GYMJ::RL::ActionFamily>(module, "ActionFamily")
        .value("TURN", GYMJ::RL::ActionFamily::Turn)
        .value("RESPONSE", GYMJ::RL::ActionFamily::Response)
        .value("READY", GYMJ::RL::ActionFamily::Ready)
        .value("UNSUPPORTED", GYMJ::RL::ActionFamily::Unsupported);

    py::enum_<GYMJ::RL::DecisionDrawKind>(module, "DecisionDrawKind")
        .value("NONE", GYMJ::RL::DecisionDrawKind::None)
        .value("OPENING", GYMJ::RL::DecisionDrawKind::Opening)
        .value("NORMAL", GYMJ::RL::DecisionDrawKind::Normal)
        .value("REPLACEMENT", GYMJ::RL::DecisionDrawKind::Replacement);

    py::class_<GYMJ::RL::ActionSemantics>(module, "ActionSemantics")
        .def_readonly("type", &GYMJ::RL::ActionSemantics::type)
        .def_readonly("tile", &GYMJ::RL::ActionSemantics::tile)
        .def_readonly(
            "relative_source",
            &GYMJ::RL::ActionSemantics::relative_source
        )
        .def_readonly(
            "relative_target",
            &GYMJ::RL::ActionSemantics::relative_target
        )
        .def_readonly("family", &GYMJ::RL::ActionSemantics::family);

    py::class_<GYMJ::RL::HuActionFeatures>(module, "HuActionFeatures")
        .def_readonly("available", &GYMJ::RL::HuActionFeatures::available)
        .def_readonly("is_tsumo", &GYMJ::RL::HuActionFeatures::is_tsumo)
        .def_readonly(
            "is_qianggang",
            &GYMJ::RL::HuActionFeatures::is_qianggang
        )
        .def_readonly(
            "pattern_mask",
            &GYMJ::RL::HuActionFeatures::pattern_mask
        )
        .def_readonly("shape_fan", &GYMJ::RL::HuActionFeatures::shape_fan)
        .def_readonly("ready_fan", &GYMJ::RL::HuActionFeatures::ready_fan)
        .def_readonly("timing_fan", &GYMJ::RL::HuActionFeatures::timing_fan)
        .def_readonly(
            "special_fan",
            &GYMJ::RL::HuActionFeatures::special_fan
        )
        .def_readonly("total_fan", &GYMJ::RL::HuActionFeatures::total_fan)
        .def_readonly("hu_points", &GYMJ::RL::HuActionFeatures::hu_points);

    py::class_<GYMJ::RL::ReadyActionFeatures>(
        module,
        "ReadyActionFeatures"
    )
        .def_readonly(
            "available",
            &GYMJ::RL::ReadyActionFeatures::available
        )
        .def_readonly("kind", &GYMJ::RL::ReadyActionFeatures::kind)
        .def_readonly("wait_mask", &GYMJ::RL::ReadyActionFeatures::wait_mask)
        .def_readonly(
            "wait_kinds",
            &GYMJ::RL::ReadyActionFeatures::wait_kinds
        )
        .def_readonly(
            "wait_copies",
            &GYMJ::RL::ReadyActionFeatures::wait_copies
        )
        .def_readonly(
            "max_base_hu_points",
            &GYMJ::RL::ReadyActionFeatures::max_base_hu_points
        );

    py::class_<GYMJ::RL::PassActionFeatures>(
        module,
        "PassActionFeatures"
    )
        .def_readonly(
            "available",
            &GYMJ::RL::PassActionFeatures::available
        )
        .def_readonly(
            "declines_pon",
            &GYMJ::RL::PassActionFeatures::declines_pon
        )
        .def_readonly(
            "declines_daiminkan",
            &GYMJ::RL::PassActionFeatures::declines_daiminkan
        )
        .def_readonly(
            "declines_ron",
            &GYMJ::RL::PassActionFeatures::declines_ron
        )
        .def_readonly(
            "declines_ready",
            &GYMJ::RL::PassActionFeatures::declines_ready
        );

    py::class_<GYMJ::RL::CandidateActionFeatures>(
        module,
        "CandidateActionFeatures"
    )
        .def_readonly("legal", &GYMJ::RL::CandidateActionFeatures::legal)
        .def_readonly(
            "supported_by_v4",
            &GYMJ::RL::CandidateActionFeatures::supported_by_v4
        )
        .def_readonly(
            "post_hand_known",
            &GYMJ::RL::CandidateActionFeatures::post_hand_known
        )
        .def_readonly(
            "semantics",
            &GYMJ::RL::CandidateActionFeatures::semantics
        )
        .def_readonly(
            "post_hand",
            &GYMJ::RL::CandidateActionFeatures::post_hand
        )
        .def_readonly("hu", &GYMJ::RL::CandidateActionFeatures::hu)
        .def_readonly("ready", &GYMJ::RL::CandidateActionFeatures::ready)
        .def_readonly("pass_features", &GYMJ::RL::CandidateActionFeatures::pass);

    py::class_<GYMJ::RL::PonFollowUpFeatures>(
        module,
        "PonFollowUpFeatures"
    )
        .def_readonly(
            "available",
            &GYMJ::RL::PonFollowUpFeatures::available
        )
        .def_readonly(
            "discard_mask",
            &GYMJ::RL::PonFollowUpFeatures::discard_mask
        )
        .def_readonly(
            "post_discard_hands",
            &GYMJ::RL::PonFollowUpFeatures::post_discard_hands
        );

    py::class_<GYMJ::RL::DecisionProgressFeatures>(
        module,
        "DecisionProgressFeatures"
    )
        .def_readonly(
            "decision_sequence",
            &GYMJ::RL::DecisionProgressFeatures::decision_sequence
        )
        .def_readonly(
            "has_normal_drawn",
            &GYMJ::RL::DecisionProgressFeatures::has_normal_drawn
        )
        .def_readonly(
            "has_discarded",
            &GYMJ::RL::DecisionProgressFeatures::has_discarded
        )
        .def_readonly(
            "opening_hand_changed",
            &GYMJ::RL::DecisionProgressFeatures::opening_hand_changed
        )
        .def_readonly(
            "ready_window_closed",
            &GYMJ::RL::DecisionProgressFeatures::ready_window_closed
        )
        .def_readonly(
            "last_draw_kind",
            &GYMJ::RL::DecisionProgressFeatures::last_draw_kind
        )
        .def_readonly(
            "must_discard_after_call",
            &GYMJ::RL::DecisionProgressFeatures::must_discard_after_call
        )
        .def_readonly(
            "hot_discard_player",
            &GYMJ::RL::DecisionProgressFeatures::hot_discard_player
        )
        .def_readonly(
            "pending_discard_is_hot",
            &GYMJ::RL::DecisionProgressFeatures::pending_discard_is_hot
        )
        .def_readonly(
            "pending_kakan",
            &GYMJ::RL::DecisionProgressFeatures::pending_kakan
        );

    py::class_<GYMJ::RL::DecisionFeatureSet>(module, "DecisionFeatureSet")
        .def_readonly(
            "observation",
            &GYMJ::RL::DecisionFeatureSet::observation
        )
        .def_readonly(
            "public_state",
            &GYMJ::RL::DecisionFeatureSet::public_state
        )
        .def_readonly("progress", &GYMJ::RL::DecisionFeatureSet::progress)
        .def_readonly("actions", &GYMJ::RL::DecisionFeatureSet::actions)
        .def_readonly(
            "pon_follow_up",
            &GYMJ::RL::DecisionFeatureSet::pon_follow_up
        );

    py::class_<GYMJ::RL::DecisionFeatureEncoder>(
        module,
        "DecisionFeatureEncoder"
    ).def_static("encode", &GYMJ::RL::DecisionFeatureEncoder::encode);

    py::class_<GYMJ::RL::PrivilegedEconomyFeatures>(
        module,
        "PrivilegedEconomyFeatures"
    )
        .def_readonly(
            "settlement_tile_counts",
            &GYMJ::RL::PrivilegedEconomyFeatures::settlement_tile_counts
        )
        .def_readonly(
            "bean_source_counts",
            &GYMJ::RL::PrivilegedEconomyFeatures::bean_source_counts
        )
        .def_readonly(
            "chongfeng_tiles",
            &GYMJ::RL::PrivilegedEconomyFeatures::chongfeng_tiles
        )
        .def_readonly(
            "chongfeng_tile_masks",
            &GYMJ::RL::PrivilegedEconomyFeatures::chongfeng_tile_masks
        )
        .def_readonly(
            "responsibility_chickens",
            &GYMJ::RL::PrivilegedEconomyFeatures::responsibility_chickens
        )
        .def_readonly(
            "player_tenpai",
            &GYMJ::RL::PrivilegedEconomyFeatures::player_tenpai
        )
        .def_readonly(
            "signed_fixed_chicken_counts",
            &GYMJ::RL::PrivilegedEconomyFeatures::signed_fixed_chicken_counts
        );

    py::class_<GYMJ::RL::PrivilegedPatternFeatures>(
        module,
        "PrivilegedPatternFeatures"
    )
        .def_readonly(
            "player_patterns",
            &GYMJ::RL::PrivilegedPatternFeatures::player_patterns
        )
        .def_readonly(
            "true_shape_wait_masks",
            &GYMJ::RL::PrivilegedPatternFeatures::true_shape_wait_masks
        )
        .def_readonly(
            "remaining_wall_counts",
            &GYMJ::RL::PrivilegedPatternFeatures::remaining_wall_counts
        )
        .def_readonly(
            "economy",
            &GYMJ::RL::PrivilegedPatternFeatures::economy
        )
        .def_readonly(
            "progress",
            &GYMJ::RL::PrivilegedPatternFeatures::progress
        );

    py::class_<GYMJ::RL::PrivilegedPatternEncoder>(
        module,
        "PrivilegedPatternEncoder"
    ).def_static("encode", &GYMJ::RL::PrivilegedPatternEncoder::encode);

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 7
namespace GymjPythonBindings {
void bind_observations(py::module_& module) {

    py::class_<GYMJ::RL::DiscardObservation>(
        module,
        "DiscardObservation"
    )
        .def_readonly("tile", &GYMJ::RL::DiscardObservation::tile)
        .def_readonly(
            "relative_actor",
            &GYMJ::RL::DiscardObservation::relative_actor
        )
        .def_readonly(
            "was_called",
            &GYMJ::RL::DiscardObservation::was_called
        );

    py::class_<GYMJ::RL::MeldObservation>(module, "MeldObservation")
        .def_readonly("tile", &GYMJ::RL::MeldObservation::tile)
        .def_readonly(
            "relative_owner",
            &GYMJ::RL::MeldObservation::relative_owner
        )
        .def_readonly(
            "relative_source",
            &GYMJ::RL::MeldObservation::relative_source
        )
        .def_readonly("type", &GYMJ::RL::MeldObservation::type)
        .def_readonly("valid", &GYMJ::RL::MeldObservation::valid);

    py::class_<GYMJ::RL::CompactObservation>(module, "CompactObservation")
        .def_readonly(
            "hand_counts",
            &GYMJ::RL::CompactObservation::hand_counts
        )
        .def_readonly("discards", &GYMJ::RL::CompactObservation::discards)
        .def_readonly(
            "discard_count",
            &GYMJ::RL::CompactObservation::discard_count
        )
        .def_readonly("melds", &GYMJ::RL::CompactObservation::melds)
        .def_readonly(
            "meld_count",
            &GYMJ::RL::CompactObservation::meld_count
        )
        .def_readonly(
            "visible_counts",
            &GYMJ::RL::CompactObservation::visible_counts
        )
        .def_readonly("waits", &GYMJ::RL::CompactObservation::waits)
        .def_readonly("shanten", &GYMJ::RL::CompactObservation::shanten)
        .def_readonly("phase", &GYMJ::RL::CompactObservation::phase)
        .def_readonly(
            "current_actor",
            &GYMJ::RL::CompactObservation::current_actor
        )
        .def_readonly(
            "current_discard",
            &GYMJ::RL::CompactObservation::current_discard
        )
        .def_readonly(
            "current_discard_actor",
            &GYMJ::RL::CompactObservation::current_discard_actor
        )
        .def_readonly("dealer", &GYMJ::RL::CompactObservation::dealer)
        .def_readonly(
            "wall_remaining",
            &GYMJ::RL::CompactObservation::wall_remaining
        )
        .def_readonly(
            "ready_kinds",
            &GYMJ::RL::CompactObservation::ready_kinds
        )
        .def_readonly(
            "locked_ready_waits",
            &GYMJ::RL::CompactObservation::locked_ready_waits
        )
        .def_readonly(
            "legal_mask",
            &GYMJ::RL::CompactObservation::legal_mask
        );

    py::class_<GYMJ::RL::ObservationEncoder>(module, "ObservationEncoder")
        .def_static("encode", &GYMJ::RL::ObservationEncoder::encode);

    py::class_<GYMJ::RL::PrivilegedState>(module, "PrivilegedState")
        .def_readonly(
            "player_tenpai",
            &GYMJ::RL::PrivilegedState::player_tenpai
        )
        .def_readonly(
            "opponent_hands",
            &GYMJ::RL::PrivilegedState::opponent_hands
        )
        .def_readonly("wall_tiles", &GYMJ::RL::PrivilegedState::wall_tiles)
        .def_readonly("wall_count", &GYMJ::RL::PrivilegedState::wall_count)
        .def_readonly(
            "future_draw_tiles",
            &GYMJ::RL::PrivilegedState::future_draw_tiles
        )
        .def_readonly(
            "replacement_draw_tiles",
            &GYMJ::RL::PrivilegedState::replacement_draw_tiles
        );

    py::class_<GYMJ::RL::PrivilegedStateEncoder>(
        module, "PrivilegedStateEncoder"
    ).def_static("encode", &GYMJ::RL::PrivilegedStateEncoder::encode);

}
} // namespace GymjPythonBindings
#elif GYMJ_BINDING_PART == 8
namespace GymjPythonBindings {
void bind_arena(py::module_& module) {

    py::class_<GYMJ::SearchConfig>(module, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("simulations", &GYMJ::SearchConfig::simulations)
        .def_readwrite("tree_depth", &GYMJ::SearchConfig::tree_depth)
        .def_readwrite("max_steps", &GYMJ::SearchConfig::max_steps)
        .def_readwrite("exploration", &GYMJ::SearchConfig::exploration)
        .def_readwrite("uniform_prior", &GYMJ::SearchConfig::uniform_prior);
    py::class_<GYMJ::SearchResult>(module, "SearchResult")
        .def_readonly("action_id", &GYMJ::SearchResult::action_id)
        .def_readonly("visits", &GYMJ::SearchResult::visits)
        .def_readonly("mean_scores", &GYMJ::SearchResult::mean_scores)
        .def_readonly("simulations", &GYMJ::SearchResult::simulations)
        .def_readonly("nodes", &GYMJ::SearchResult::nodes)
        .def_readonly("deepest_path", &GYMJ::SearchResult::deepest_path)
        .def_readonly("rollout_steps", &GYMJ::SearchResult::rollout_steps);
    py::class_<GYMJ::SearchRoot>(module, "SearchRoot")
        .def(py::init<const GYMJ::Arena&>())
        .def_static("unsupported_reason", &GYMJ::SearchRoot::unsupported_reason)
        .def_property_readonly("observer", &GYMJ::SearchRoot::observer)
        .def_property_readonly("remaining", &GYMJ::SearchRoot::remaining)
        .def("sample", &GYMJ::SearchRoot::sample)
        .def("fingerprint", [](const GYMJ::SearchRoot& root) { return py::bytes(root.fingerprint()); })
        .def("search", &GYMJ::SearchRoot::search, py::call_guard<py::gil_scoped_release>())
        .def("search_policy", &GYMJ::SearchRoot::search_policy, py::call_guard<py::gil_scoped_release>());

    py::class_<GYMJ::PlayerState>(module, "PlayerState")
        .def_readonly("player_id", &GYMJ::PlayerState::player_id)
        .def_readonly("shanten", &GYMJ::PlayerState::shanten)
        .def_property_readonly("hand", &hand_counts)
        .def_property_readonly("melds", [](const GYMJ::PlayerState& player) {
            return player.melds[0];
        });

    py::class_<GYMJ::Arena>(module, "Arena")
        .def(
            py::init([](
                uint64_t seed,
                uint8_t dealer,
                const std::string& log_path
            ) {
                return std::make_unique<GYMJ::Arena>(
                    GYMJ::ArenaConfig{seed, dealer, log_path}
                );
            }),
            py::arg("seed") = 0,
            py::arg("dealer") = 0,
            py::arg("log_path") = ""
        )
        .def_static(
            "from_tiles",
            [](
                const std::vector<uint8_t>& tile_ids,
                uint8_t dealer,
                uint64_t seed,
                const std::string& log_path
            ) {
                return std::make_unique<GYMJ::Arena>(
                    GYMJ::ArenaConfig{seed, dealer, log_path},
                    GYMJ::Wall::from_tiles(make_tiles(tile_ids), seed)
                );
            },
            py::arg("tile_ids"),
            py::arg("dealer") = 0,
            py::arg("seed") = 0,
            py::arg("log_path") = ""
        )
        .def("start_round", &GYMJ::Arena::start_round)
        .def("current_decision", &GYMJ::Arena::current_decision)
        .def("submit_action", &GYMJ::Arena::submit_action)
        .def_property_readonly("phase", &GYMJ::Arena::phase)
        .def_property_readonly("finished", &GYMJ::Arena::is_finished)
        .def_property_readonly("current_player", &GYMJ::Arena::current_player)
        .def_property_readonly("dealer", &GYMJ::Arena::dealer)
        .def_property_readonly("wall_remaining", &GYMJ::Arena::wall_remaining)
        .def(
            "ready_state",
            &GYMJ::Arena::ready_state,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly("result", [](const GYMJ::Arena& arena) {
            return arena.result();
        })
        .def_property_readonly(
            "chicken_result",
            &GYMJ::Arena::chicken_result,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly(
            "round_settlement",
            &GYMJ::Arena::round_settlement,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly("history", [](const GYMJ::Arena& arena) {
            return arena.history();
        })
        .def_property_readonly("discard_history", [](const GYMJ::Arena& arena) {
            return arena.discard_history();
        })
        .def("discard_history_for", &GYMJ::Arena::discard_history_for)
        .def(
            "player",
            &GYMJ::Arena::player,
            py::return_value_policy::reference_internal
        );

    py::class_<GYMJ::Game>(module, "Game")
        .def(py::init<const GYMJ::GameConfig&>())
        .def("start", &GYMJ::Game::start)
        .def("finish_hand", &GYMJ::Game::finish_hand)
        .def("start_next_hand", &GYMJ::Game::start_next_hand)
        .def("current_decision", &GYMJ::Game::current_decision)
        .def("submit_action", &GYMJ::Game::submit_action)
        .def_property_readonly("phase", &GYMJ::Game::phase)
        .def_property_readonly("finished", &GYMJ::Game::is_finished)
        .def_property_readonly("hand_index", &GYMJ::Game::hand_index)
        .def_property_readonly("dealer", &GYMJ::Game::dealer)
        .def_property_readonly(
            "dealer_continuation",
            &GYMJ::Game::dealer_continuation
        )
        .def_property_readonly(
            "scores",
            &GYMJ::Game::scores,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly(
            "history",
            &GYMJ::Game::history,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly(
            "arena",
            [](GYMJ::Game& game) -> GYMJ::Arena& {
                return game.arena();
            },
            py::return_value_policy::reference_internal
        );
}
} // namespace GymjPythonBindings
#endif
