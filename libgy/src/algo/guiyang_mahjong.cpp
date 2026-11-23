#include "guiyang_mahjong.h"
#include "agari.h"
#include "shanten.h"

#include <algorithm>
#include <sstream>

namespace GYMJ {

    // ==========================================
    // Tile implementation
    // ==========================================
    uint8_t Tile::suit() const {
        return static_cast<uint8_t>(id / TILES_PER_SUIT);
    }
    uint8_t Tile::rank() const {
        return static_cast<uint8_t>((id % TILES_PER_SUIT) + 1);
    }
    bool Tile::operator==(const Tile& other) const { return id == other.id; }
    bool Tile::operator!=(const Tile& other) const { return id != other.id; }
    
    std::string Tile::to_string() const {
        if (id >= TILE_TYPE_COUNT) return "??";
        const char suits[] = {'m', 'p', 's'};
        return std::to_string(rank()) + suits[suit()];
    }

    // ==========================================
    // PlayerState implementation
    // ==========================================
    PlayerState::PlayerState(uint8_t id) : player_id(id), is_zeren_ji(false), shanten(14) {
        std::fill(std::begin(tehai), std::end(tehai), 0);
        std::fill(std::begin(waits), std::end(waits), false);
    }

    void PlayerState::apply_action(const Action& action) {
        uint8_t rel_actor = (action.actor - this->player_id + 4) % 4;

        switch (action.type) {
            case ActionType::Draw:
                if (rel_actor == 0) tehai[action.tile.id]++;
                break;
            case ActionType::Discard:
                if (rel_actor == 0) {
                    tehai[action.tile.id]--;
                }
                kawa[rel_actor].push_back(action.tile);
                break;
            case ActionType::Pon:
                if (rel_actor == 0) {
                    tehai[action.tile.id] -= 2; 
                }
                melds[rel_actor].push_back({MeldType::Pon, action.tile, action.target});
                remove_from_kawa(action.target, action.tile);
                break;
            case ActionType::AnKan:
                if (rel_actor == 0) {
                    tehai[action.tile.id] -= 4; 
                }
                melds[rel_actor].push_back({MeldType::AnKan, action.tile, action.actor}); 
                break;
            case ActionType::Kakan:
                if (rel_actor == 0) {
                    tehai[action.tile.id] -= 1;
                }
                for (auto& m : melds[rel_actor]) {
                    if (m.type == MeldType::Pon && m.tile == action.tile) {
                        m.type = MeldType::Kakan;
                        break;
                    }
                }
                break;
            case ActionType::Daiminkan:
                if (rel_actor == 0) {
                    tehai[action.tile.id] -= 3;
                }
                melds[rel_actor].push_back({
                    MeldType::Daiminkan,
                    action.tile,
                    action.target
                });
                remove_from_kawa(action.target, action.tile);
                break;
            case ActionType::Tsumo:
            case ActionType::Ron:
            case ActionType::DeclareReady:
            case ActionType::Pass:
                break;
        }
        update_cache(); 
    }

    std::vector<Action> PlayerState::get_legal_actions_on_draw(Tile drawn_tile) const {
        std::vector<Action> actions;
        for (uint8_t i = 0; i < TILE_TYPE_COUNT; ++i) {
            if (tehai[i] > 0) actions.push_back({ActionType::Discard, player_id, Tile{i}, player_id});
            if (tehai[i] == MAX_TILE_COPIES) {
                actions.push_back({ActionType::AnKan, player_id, Tile{i}, player_id});
            }
        }
        for (const auto& meld : melds[0]) {
            if (meld.type == MeldType::Pon &&
                meld.tile == drawn_tile &&
                tehai[meld.tile.id] > 0) {
                actions.push_back({ActionType::Kakan, player_id, meld.tile, player_id});
            }
        }
        return actions;
    }

    std::vector<Action> PlayerState::get_legal_actions_on_opponent_discard(uint8_t opponent_id, Tile discarded_tile) const {
        std::vector<Action> actions;
        actions.push_back({ActionType::Pass, player_id, discarded_tile, opponent_id});

        if (tehai[discarded_tile.id] >= 2) actions.push_back({ActionType::Pon, player_id, discarded_tile, opponent_id});
        if (tehai[discarded_tile.id] == 3) {
            actions.push_back({
                ActionType::Daiminkan,
                player_id,
                discarded_tile,
                opponent_id
            });
        }
        return actions;
    }

    std::string PlayerState::brief_info() const {
        std::stringstream ss;
        ss << "[P" << (int)player_id << "] Hand: ";
        for (uint8_t i = 0; i < TILE_TYPE_COUNT; ++i) {
            for (uint8_t c = 0; c < tehai[i]; ++c) ss << Tile{i}.to_string() << " ";
        }
        if (!melds[0].empty()) {
            ss << "| Melds: ";
            for (const auto& m : melds[0]) {
                if (m.type == MeldType::Pon) ss << "P(";
                else if (m.type == MeldType::AnKan) ss << "AK(";
                else if (m.type == MeldType::Kakan) ss << "KK(";
                else ss << "DK(";
                ss << m.tile.to_string() << ") ";
            }
        }
        ss << "| Waits: ";
        for (uint8_t i = 0; i < TILE_TYPE_COUNT; ++i) if(waits[i]) ss << Tile{i}.to_string() << " ";
        
        return ss.str();
    }

    void PlayerState::remove_from_kawa(uint8_t target_id, Tile t) {
        uint8_t rel_target = (target_id - this->player_id + 4) % 4;
        if (!kawa[rel_target].empty() && kawa[rel_target].back() == t) {
            kawa[rel_target].pop_back();
        }
    }

    void PlayerState::update_cache() {
        shanten = Algo::calculate_shanten(
            tehai,
            static_cast<int>(melds[0].size())
        ).best;

        auto visible_count = [this](uint8_t tile_id) {
            int count = tehai[tile_id];

            for (const auto& river : kawa) {
                for (const auto& tile : river) {
                    if (tile.id == tile_id) count++;
                }
            }

            for (const auto& player_melds : melds) {
                for (const auto& meld : player_melds) {
                    if (meld.tile.id != tile_id) continue;
                    count += (meld.type == MeldType::Pon) ? 3 : 4;
                }
            }

            return count;
        };

        std::fill(std::begin(waits), std::end(waits), false);
        if (shanten == 0) {
            uint8_t temp[TILE_TYPE_COUNT];
            std::copy(tehai, tehai + TILE_TYPE_COUNT, temp);
            
            for (uint8_t i = 0; i < TILE_TYPE_COUNT; ++i) {
                if (visible_count(i) >= MAX_TILE_COPIES) continue;
                temp[i]++; 
                
                if (Algo::is_standard_hu(temp, melds[0].size()) ||
                    (melds[0].empty() && Algo::is_seven_pairs(temp))) {
                    waits[i] = true;
                }
                
                temp[i]--; 
            }
        }
    }

} // namespace GYMJ
