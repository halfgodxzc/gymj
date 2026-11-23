#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace GYMJ {

    inline constexpr std::size_t TILE_TYPE_COUNT = 27;
    inline constexpr std::size_t TILES_PER_SUIT = 9;
    inline constexpr uint8_t MAX_TILE_COPIES = 4;
    inline constexpr uint8_t YAO_JI_TILE_ID = 18;   // 1s
    inline constexpr uint8_t BA_TONG_TILE_ID = 16;  // 8p
    inline constexpr int COMPLETE_HAND_TILE_COUNT = 14;
    inline constexpr int COMPLETE_HAND_MELD_COUNT = 4;
    inline constexpr int TILES_PER_MELD = 3;

    // Basic tile and action types.
    struct Tile {
        uint8_t id; // 0-26 (0-8: man, 9-17: pin, 18-26: sou)
        
        uint8_t suit() const;
        uint8_t rank() const;
        
        bool operator==(const Tile& other) const;
        bool operator!=(const Tile& other) const;
        
        std::string to_string() const;
    };

    enum class ActionType : uint8_t {
        Pass,
        Draw,
        Discard,
        Pon,
        AnKan,
        Kakan,
        Daiminkan,
        Tsumo,
        Ron,
        DeclareReady
    };

    struct Action {
        ActionType type;
        uint8_t actor;
        Tile tile;
        uint8_t target;
    };

    enum class MeldType : uint8_t { Pon, AnKan, Kakan, Daiminkan };

    struct Meld {
        MeldType type;
        Tile tile;
        uint8_t source;
    };

    // Player-local state and legal action generation.
    class PlayerState {
    public:
        uint8_t player_id;
        
        // Known state from this player's perspective.
        uint8_t tehai[TILE_TYPE_COUNT];
        std::vector<Tile> kawa[4];
        // Public melds by relative seat; index 0 is self.
        std::vector<Meld> melds[4];
        
        // Guiyang-specific markers.
        bool is_zeren_ji;
        std::vector<Tile> ji_tiles;

        // Cached hand analysis.
        int8_t shanten;
        bool waits[TILE_TYPE_COUNT];

        explicit PlayerState(uint8_t id);

        void apply_action(const Action& action);

        std::vector<Action> get_legal_actions_on_draw(Tile drawn_tile) const;
        std::vector<Action> get_legal_actions_on_opponent_discard(uint8_t opponent_id, Tile discarded_tile) const;

        std::string brief_info() const;

    private:
        void remove_from_kawa(uint8_t target_id, Tile t);
        void update_cache();
    };

} // namespace GYMJ
