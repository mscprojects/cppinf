#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fmt/format.h>

class TestTempDir {
  public:
    explicit TestTempDir(std::string_view prefix) {
        const auto dir_template = std::filesystem::temp_directory_path() / fmt::format("{}-XXXXXX", prefix);
        std::string template_text = dir_template.string();

        std::vector<char> buffer(template_text.begin(), template_text.end());
        buffer.push_back('\0');

        const char* created_dir = ::mkdtemp(buffer.data());
        if (created_dir == nullptr) {
            throw std::system_error(errno, std::generic_category(),
                                    fmt::format("Failed to create temp directory for {}.", prefix));
        }

        path_ = created_dir;
    }

    TestTempDir(const TestTempDir&) = delete;
    TestTempDir& operator=(const TestTempDir&) = delete;

    TestTempDir(TestTempDir&&) = delete;
    TestTempDir& operator=(TestTempDir&&) = delete;

    ~TestTempDir() {
        std::error_code error_code;
        std::filesystem::remove_all(path_, error_code);
    }

    const std::filesystem::path& path() const {
        return path_;
    }

  private:
    std::filesystem::path path_;
};
