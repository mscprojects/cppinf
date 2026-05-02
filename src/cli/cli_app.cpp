#include "cli/cli_app.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "loaders/hf/hf_model_summary.h"
#include "models/qwen3/qwen3_model.h"
#include "ops/op_utils.h"
#include "tokenizers/hf/hf_tokenizer.h"

namespace cppinf::cli {
namespace detail {

struct InspectHfOptions {
    std::filesystem::path model_dir;
    bool show_all_tensors{};
    std::size_t tensor_limit{8};
};

struct RunHfOptions {
    std::filesystem::path model_dir;
    std::string prompt;
    std::size_t max_new_tokens{32};
    float temperature{};
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
    return fmt::format("Usage:\n"
                       "  cppinf\n"
                       "  cppinf inspect hf <model-dir> [--all] [--limit <count>]\n"
                       "  cppinf run hf <model-dir> --prompt <text> [--max-new-tokens <count>] "
                       "[--temperature <value>]\n");
}

CliResult invalid_usage() {
    return CliResult{
        .exit_code = 1,
        .output = usage_text(),
    };
}

CliResult command_failure(std::string_view message) {
    return CliResult{
        .exit_code = 1,
        .output = fmt::format("{}\n", message),
    };
}

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view name) {
    if (dim <= 0) {
        throw std::invalid_argument(fmt::format("{} must be positive.", name));
    }
    return static_cast<std::size_t>(dim);
}

std::int64_t select_argmax_token_id(const tensors::Tensor& logits) {
    if (logits.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("CLI generation requires rank-2 logits.");
    }

    const auto& dims = logits.tensor_info().shape.dims();
    const auto sequence_length = checked_positive_dim_to_size(dims[0], "logits sequence length");
    const auto vocab_size = checked_positive_dim_to_size(dims[1], "logits vocab size");
    const auto last_row_offset = (sequence_length - 1) * vocab_size;

    std::int64_t best_token_id = 0;
    float best_logit = ops::detail::load_float_value(logits.tensor_info().dtype, logits.data(), last_row_offset);
    for (std::size_t token_index = 1; token_index < vocab_size; ++token_index) {
        const auto logit =
            ops::detail::load_float_value(logits.tensor_info().dtype, logits.data(), last_row_offset + token_index);
        if (logit > best_logit) {
            best_logit = logit;
            best_token_id = static_cast<std::int64_t>(token_index);
        }
    }

    return best_token_id;
}

std::int64_t sample_token_id(const tensors::Tensor& logits, float temperature, std::mt19937& random_engine) {
    if (logits.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("CLI generation requires rank-2 logits.");
    }

    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::invalid_argument("run hf requires a non-negative finite temperature.");
    }

    // Temperature 0 keeps generation deterministic by selecting the largest last-token logit instead of sampling.
    if (temperature == 0.0f) {
        return select_argmax_token_id(logits);
    }

    // Only the final sequence row matters for autoregressive decoding, it predicts the next token distribution.
    const auto& dims = logits.tensor_info().shape.dims();
    const auto sequence_length = checked_positive_dim_to_size(dims[0], "logits sequence length");
    const auto vocab_size = checked_positive_dim_to_size(dims[1], "logits vocab size");
    const auto last_row_offset = (sequence_length - 1) * vocab_size;

    // Divide logits by temperature before softmax: lower temperatures sharpen probabilities, higher temperatures
    // flatten them. Subtract the max scaled logit so exp() stays numerically stable.
    auto max_scaled_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t token_index = 0; token_index < vocab_size; ++token_index) {
        const auto logit =
            ops::detail::load_float_value(logits.tensor_info().dtype, logits.data(), last_row_offset + token_index);
        max_scaled_logit = std::max(max_scaled_logit, logit / temperature);
    }

    std::vector<double> weights;
    weights.reserve(vocab_size);
    for (std::size_t token_index = 0; token_index < vocab_size; ++token_index) {
        const auto logit =
            ops::detail::load_float_value(logits.tensor_info().dtype, logits.data(), last_row_offset + token_index);
        weights.push_back(std::exp(static_cast<double>(logit / temperature - max_scaled_logit)));
    }

    // discrete_distribution normalizes positive weights internally, so the exponentials can be used as softmax weights
    // without explicitly dividing by their sum.
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return static_cast<std::int64_t>(distribution(random_engine));
}

void stream_generated_text(const OutputWriter& output_writer, std::string_view previous_text,
                           std::string_view current_text) {
    if (!output_writer || current_text.size() <= previous_text.size()) {
        return;
    }

    if (!current_text.starts_with(previous_text)) {
        return;
    }

    const auto suffix = current_text.substr(previous_text.size());
    if (!suffix.empty()) {
        output_writer(suffix);
    }
}

std::string run_hf_generation(const RunHfOptions& options, const OutputWriter& output_writer) {
    const auto tokenizer = tokenizers::hf::HfTokenizer::from_dir(options.model_dir);
    const auto model = models::qwen3::Qwen3Model::from_dir(options.model_dir);

    auto token_ids = tokenizer.encode(options.prompt);
    if (token_ids.empty()) {
        throw std::invalid_argument("run hf requires a prompt that encodes to at least one token.");
    }

    auto decoded_text = tokenizer.decode(token_ids);
    if (output_writer && !decoded_text.empty()) {
        output_writer(decoded_text);
    }

    const auto eos_token_id = tokenizer.eos_token_id();
    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    auto cache = model.make_cache();
    auto logits = model.forward_cached(token_ids, cache);
    for (std::size_t step = 0; step < options.max_new_tokens; ++step) {
        const auto next_token_id = sample_token_id(logits, options.temperature, random_engine);
        if (eos_token_id.has_value() && next_token_id == *eos_token_id) {
            break;
        }
        token_ids.push_back(next_token_id);

        auto next_decoded_text = tokenizer.decode(token_ids);
        stream_generated_text(output_writer, decoded_text, next_decoded_text);
        decoded_text = std::move(next_decoded_text);

        if (step + 1 < options.max_new_tokens) {
            const std::array<std::int64_t, 1> current_token_ids = {next_token_id};
            logits = model.forward_cached(current_token_ids, cache);
        }
    }

    return decoded_text;
}

} // namespace detail

CliResult run(std::span<const std::string_view> args) {
    return run_with_output_writer(args, {});
}

CliResult run_with_output_writer(std::span<const std::string_view> args, const OutputWriter& output_writer) {
    if (args.empty()) {
        return CliResult{
            .exit_code = 0,
            .output = fmt::format("cppinf\n"),
        };
    }

    detail::InspectHfOptions inspect_hf_options;
    detail::RunHfOptions run_hf_options;
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

    auto* run_subcommand = app.add_subcommand("run", "Run model inference.");
    run_subcommand->require_subcommand(1);

    auto* run_hf_subcommand = run_subcommand->add_subcommand("hf", "Run greedy generation from a Hugging Face model.");
    run_hf_subcommand->add_option("model_dir", run_hf_options.model_dir)->required();
    run_hf_subcommand->add_option("--prompt", run_hf_options.prompt, "Prompt text.")->required();
    auto* max_new_tokens_option = run_hf_subcommand->add_option("--max-new-tokens", run_hf_options.max_new_tokens,
                                                                "Generate up to N new tokens.");
    max_new_tokens_option->check(CLI::PositiveNumber);
    run_hf_subcommand
        ->add_option("--temperature", run_hf_options.temperature,
                     "Sampling temperature. Use 0 for deterministic greedy generation.")
        ->check(CLI::NonNegativeNumber);

    auto owned_args = detail::to_owned_args(args);
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

    try {
        if (inspect_hf_subcommand->parsed()) {
            const auto tensor_limit = inspect_hf_options.show_all_tensors ? std::numeric_limits<std::size_t>::max()
                                                                          : inspect_hf_options.tensor_limit;
            const auto summary = loaders::hf::load_model_summary(inspect_hf_options.model_dir, tensor_limit);
            return CliResult{
                .exit_code = 0,
                .output = fmt::format("{}\n", loaders::hf::format_model_summary(summary)),
            };
        }

        if (run_hf_subcommand->parsed()) {
            const auto decoded_text = detail::run_hf_generation(run_hf_options, output_writer);
            return CliResult{
                .exit_code = 0,
                .output = output_writer ? std::string("\n") : fmt::format("{}\n", decoded_text),
            };
        }
    } catch (const std::exception& error) {
        return detail::command_failure(error.what());
    }

    return detail::invalid_usage();
}

} // namespace cppinf::cli
