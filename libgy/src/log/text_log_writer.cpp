#include "text_log_writer.h"

#include <filesystem>
#include <stdexcept>

namespace GYMJ {

TextLogWriter::TextLogWriter(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("log path must not be empty");
    }

    const std::filesystem::path file_path(path);
    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(file_path.parent_path());
    }

    output_.open(file_path, std::ios::out | std::ios::trunc);
    if (!output_) {
        throw std::runtime_error("failed to open log file: " + path);
    }
}

void TextLogWriter::write(std::string_view text) {
    std::lock_guard<std::mutex> lock(mutex_);
    output_ << text;
    output_.flush();
    if (!output_) {
        throw std::runtime_error("failed to write text log");
    }
}

} // namespace GYMJ
