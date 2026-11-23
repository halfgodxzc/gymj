#pragma once

#include <cstdint>
#include <vector>

namespace GYMJ {

constexpr int GUIYANG_HU_BASE_POINTS = 5;

enum class HuPattern : uint8_t {
    Standard,
    AllTriplets,
    SevenPairs,
    DragonSevenPairs,
    DiLong
};

enum class HuTiming : uint8_t {
    Normal,
    DiHu,
    TianHu
};

enum class SpecialHu : uint8_t {
    None,
    GangKai = 1 << 0,
    RePao = 1 << 1,
    QiangGang = 1 << 2,
    ShaBao = 1 << 3
};

enum class ReadyKind : uint8_t {
    None,
    TianTing,
    BaoTing
};

constexpr SpecialHu operator|(SpecialHu lhs, SpecialHu rhs) {
    return static_cast<SpecialHu>(
        static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs)
    );
}

constexpr bool has_special(SpecialHu value, SpecialHu flag) {
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(flag)) != 0;
}

enum class FanType : uint8_t {
    PingHu,
    DaDuiZi,
    SuGuaDou,
    QiDui,
    QingYiSe,
    QingDanDiao,
    LongQiDui,
    DiLong,
    BaoTing,
    TianTing,
    GangKai,
    RePao,
    QiangGang,
    ShaBao,
    DiHu,
    TianHu
};

struct HuFeatures {
    HuPattern pattern = HuPattern::Standard;
    bool is_flush = false;
    bool is_single_wait = false;
    ReadyKind ready_kind = ReadyKind::None;
    bool is_su_gua_dou = false;
    HuTiming timing = HuTiming::Normal;
    SpecialHu special = SpecialHu::None;
};

struct FanItem {
    FanType type;
    int fan;
};

struct HuScore {
    int shape_fan = 0;
    int ready_fan = 0;
    int timing_fan = 0;
    int special_fan = 0;
    int total_fan = 0;
    int hu_points = 0;
    std::vector<FanItem> items;
};

class GuiyangScoring {
public:
    static HuScore calculate(const HuFeatures& features);
};

} // namespace GYMJ
