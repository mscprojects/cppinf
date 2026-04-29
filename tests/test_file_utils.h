#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <fmt/format.h>

namespace cppinf::tests::file_test_utils {
namespace detail {

inline void append_u64_le(std::uint64_t value, std::vector<std::byte>& bytes) {
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
        bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 0xffU));
    }
}

} // namespace detail

// Writes text to the given path and fails the test setup immediately if the file cannot be written.
inline void write_text_file(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path);
    if (!output) {
        throw std::invalid_argument(fmt::format("Failed to open test file '{}' for writing.", path.string()));
    }

    output << text;
    if (!output) {
        throw std::invalid_argument(fmt::format("Failed to write test file '{}'.", path.string()));
    }
}

// Writes binary bytes to the given path and fails the test setup immediately if the file cannot be written.
inline void write_binary_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::invalid_argument(fmt::format("Failed to open test file '{}' for writing.", path.string()));
    }

    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::invalid_argument(fmt::format("Failed to write test file '{}'.", path.string()));
    }
}

// Builds a minimal safetensors file payload from header text and raw tensor bytes.
inline std::vector<std::byte> make_safetensors_file_bytes(std::string_view header_json,
                                                          std::span<const std::byte> tensor_data) {
    std::vector<std::byte> bytes;
    detail::append_u64_le(static_cast<std::uint64_t>(header_json.size()), bytes);
    for (const char character : header_json) {
        bytes.push_back(static_cast<std::byte>(character));
    }
    bytes.insert(bytes.end(), tensor_data.begin(), tensor_data.end());
    return bytes;
}

} // namespace cppinf::tests::file_test_utils
