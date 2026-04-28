#pragma once

#include <span>
#include <string>
#include <string_view>

namespace cppinf::cli {

struct CliResult {
    int exit_code{};
    std::string output;

    bool operator==(const CliResult&) const = default;
};

CliResult run(std::span<const std::string_view> args);

} // namespace cppinf::cli
