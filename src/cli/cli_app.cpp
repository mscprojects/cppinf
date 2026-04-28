#include "cli/cli_app.h"

#include <filesystem>

#include <fmt/format.h>

#include "loaders/hf/hf_model_summary.h"

namespace cppinf::cli {
namespace detail {

CliResult invalid_usage() {
    return CliResult{
        .exit_code = 1,
        .output = fmt::format("Usage:\n  cppinf\n  cppinf inspect hf <model-dir>\n"),
    };
}

} // namespace detail

CliResult run(std::span<const std::string_view> args) {
    if (args.empty()) {
        return CliResult{
            .exit_code = 0,
            .output = fmt::format("cppinf\n"),
        };
    }

    if (args.size() == 3 && args[0] == "inspect" && args[1] == "hf") {
        const auto summary = loaders::hf::load_model_summary(std::filesystem::path(args[2]));
        return CliResult{
            .exit_code = 0,
            .output = fmt::format("{}\n", loaders::hf::format_model_summary(summary)),
        };
    }

    return detail::invalid_usage();
}

} // namespace cppinf::cli
