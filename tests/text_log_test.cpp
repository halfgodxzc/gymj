#include "arena.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

GYMJ::Action choose_passive_action(const GYMJ::DecisionContext& decision) {
    const auto find = [&](GYMJ::ActionType type) {
        return std::find_if(
            decision.legal_actions.begin(),
            decision.legal_actions.end(),
            [&](const GYMJ::Action& action) {
                return action.type == type;
            }
        );
    };

    if (const auto tsumo = find(GYMJ::ActionType::Tsumo);
        tsumo != decision.legal_actions.end()) {
        return *tsumo;
    }
    if (const auto ron = find(GYMJ::ActionType::Ron);
        ron != decision.legal_actions.end()) {
        return *ron;
    }
    if (const auto pass = find(GYMJ::ActionType::Pass);
        pass != decision.legal_actions.end()) {
        return *pass;
    }

    const auto discard = find(GYMJ::ActionType::Discard);
    assert(discard != decision.legal_actions.end());
    return *discard;
}

void test_arena_writes_human_readable_text_log() {
    const std::filesystem::path log_path =
        std::filesystem::temp_directory_path() /
        "gymj_text_log_test.log";
    std::filesystem::remove(log_path);

    {
        GYMJ::Arena arena(GYMJ::ArenaConfig{
            123456,
            0,
            log_path.string(),
        });
        arena.start_round();
        while (!arena.is_finished()) {
            const GYMJ::DecisionContext decision = arena.current_decision();
            arena.submit_action(choose_passive_action(decision));
        }
    }

    std::ifstream input(log_path);
    assert(input);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();

    assert(text.find("=== Round Start ===") != std::string::npos);
    assert(text.find("P0 initial hand:") != std::string::npos);
    assert(text.find("[Decision 0]") != std::string::npos);
    assert(text.find("Normal split:") != std::string::npos);
    assert(text.find("Shanten: normal=") != std::string::npos);
    assert(text.find("Legal actions:") != std::string::npos);
    assert(text.find("Selected action:") != std::string::npos);
    assert(text.find("=== Round End ===") != std::string::npos);
    assert(text.find("Score deltas:") != std::string::npos);

    input.close();
    std::filesystem::remove(log_path);
}

} // namespace

int main() {
    test_arena_writes_human_readable_text_log();
    return 0;
}
