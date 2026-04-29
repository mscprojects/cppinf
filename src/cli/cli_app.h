#pragma once

#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace cppinf::cli {

struct CliResult {
    int exit_code{};
    std::string output;

    bool operator==(const CliResult&) const = default;
};

using OutputWriter = std::function<void(std::string_view)>;

CliResult run(std::span<const std::string_view> args);
CliResult run_with_output_writer(std::span<const std::string_view> args, const OutputWriter& output_writer);

} // namespace cppinf::cli
