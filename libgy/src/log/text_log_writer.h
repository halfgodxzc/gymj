#pragma once

#include <fstream>
#include <mutex>
#include <string>
#include <string_view>

namespace GYMJ {

class TextLogWriter {
public:
    explicit TextLogWriter(const std::string& path);

    void write(std::string_view text);

private:
    std::ofstream output_;
    std::mutex mutex_;
};

} // namespace GYMJ
