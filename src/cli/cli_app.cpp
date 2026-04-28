#include "cli/cli_app.h"

#include <filesystem>
#include <limits>
#include <vector>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "loaders/hf/hf_model_summary.h"

namespace cppinf::cli {
namespace detail {

struct InspectHfOptions {
    std::filesystem::path model_dir;
    bool show_all_tensors{};
    std::size_t tensor_limit{8};
};

std::vector<std::string> to_owned_args(std::span<const std::string_view> args) {
    std::vector<std::string> owned_args;
    owned_args.reserve(args.size() + 1);
    owned_args.emplace_back("cppinf");
    for (const std::string_view arg : args) {
        owned_args.emplace_back(arg);
    }

    return owned_args;
}

std::string usage_text() {
    return fmt::format("Usage:\n  cppinf\n  cppinf inspect hf <model-dir> [--all] [--limit <count>]\n");
}

CliResult invalid_usage() {
    return CliResult{
        .exit_code = 1,
        .output = usage_text(),
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

    detail::InspectHfOptions inspect_hf_options;
    CLI::App app{"cppinf"};

    auto* inspect_subcommand = app.add_subcommand("inspect", "Inspect model artifacts.");
    inspect_subcommand->require_subcommand(1);

    auto* inspect_hf_subcommand = inspect_subcommand->add_subcommand("hf", "Inspect a Hugging Face model directory.");
    inspect_hf_subcommand->add_option("model_dir", inspect_hf_options.model_dir)->required();
    auto* all_tensors_option =
        inspect_hf_subcommand->add_flag("--all", inspect_hf_options.show_all_tensors, "Show all tensors.");
    auto* tensor_limit_option =
        inspect_hf_subcommand->add_option("--limit", inspect_hf_options.tensor_limit, "Show the first N tensors.");
    tensor_limit_option->check(CLI::PositiveNumber);
    all_tensors_option->excludes(tensor_limit_option);

    std::vector<std::string> owned_args = detail::to_owned_args(args);
    std::vector<char*> argv;
    argv.reserve(owned_args.size());
    for (std::string& arg : owned_args) {
        argv.push_back(arg.data());
    }

    try {
        app.parse(static_cast<int>(argv.size()), argv.data());
    } catch (const CLI::ParseError&) {
        return detail::invalid_usage();
    }

    if (inspect_hf_subcommand->parsed()) {
        const std::size_t tensor_limit = inspect_hf_options.show_all_tensors ? std::numeric_limits<std::size_t>::max()
                                                                             : inspect_hf_options.tensor_limit;
        const auto summary = loaders::hf::load_model_summary(inspect_hf_options.model_dir, tensor_limit);
        return CliResult{
            .exit_code = 0,
            .output = fmt::format("{}\n", loaders::hf::format_model_summary(summary)),
        };
    }

    return detail::invalid_usage();
}

} // namespace cppinf::cli
