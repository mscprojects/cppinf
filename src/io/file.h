#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace cppinf::io {

// Reads a whole text file and throws with caller-provided context on open or read errors.
std::string read_text_file(const std::filesystem::path& path, std::string_view file_kind);

// Reads a whole binary file and throws with caller-provided context on open or read errors.
std::vector<std::byte> read_binary_file(const std::filesystem::path& path, std::string_view file_kind);

} // namespace cppinf::io
