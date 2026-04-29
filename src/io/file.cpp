#include "io/file.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>

namespace cppinf::io {

std::string read_text_file(const std::filesystem::path& path, std::string_view file_kind) {
    std::ifstream file(path);
    if (!file) {
        throw std::invalid_argument(fmt::format("Failed to open {} '{}'.", file_kind, path.string()));
    }

    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        throw std::invalid_argument(fmt::format("Failed to read {} '{}'.", file_kind, path.string()));
    }

    return text;
}

std::vector<std::byte> read_binary_file(const std::filesystem::path& path, std::string_view file_kind) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::invalid_argument(fmt::format("Failed to open {} '{}'.", file_kind, path.string()));
    }

    const auto file_size = file.tellg();
    if (file_size < 0) {
        throw std::invalid_argument(fmt::format("Failed to read {} '{}'.", file_kind, path.string()));
    }

    if (static_cast<std::uint64_t>(file_size) > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(fmt::format("{} '{}' is too large to load into memory.", file_kind, path.string()));
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    file.seekg(0, std::ios::beg);
    if (!file) {
        throw std::invalid_argument(fmt::format("Failed to read {} '{}'.", file_kind, path.string()));
    }

    if (!bytes.empty()) {
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!file) {
            throw std::invalid_argument(fmt::format("Failed to read {} '{}'.", file_kind, path.string()));
        }
    }

    return bytes;
}

} // namespace cppinf::io
