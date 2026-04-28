#include "loaders/hf/hf_model_files.h"

#include <stdexcept>

#include <fmt/format.h>

namespace cppinf::loaders::hf {
namespace detail {

std::filesystem::path require_file(const std::filesystem::path& model_dir, std::string_view file_name) {
    const std::filesystem::path path = model_dir / file_name;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument(
            fmt::format("Required Hugging Face file '{}' is missing from '{}'.", file_name, model_dir.string()));
    }

    return path;
}

std::optional<std::filesystem::path> optional_file(const std::filesystem::path& model_dir, std::string_view file_name) {
    const std::filesystem::path path = model_dir / file_name;
    if (std::filesystem::is_regular_file(path)) {
        return path;
    }

    return std::nullopt;
}

} // namespace detail

HfModelFiles HfModelFiles::from_dir(const std::filesystem::path& model_dir) {
    if (!std::filesystem::is_directory(model_dir)) {
        throw std::invalid_argument(
            fmt::format("Hugging Face model directory '{}' does not exist.", model_dir.string()));
    }

    if (std::filesystem::is_regular_file(model_dir / "model.safetensors.index.json")) {
        throw std::invalid_argument("Sharded Hugging Face safetensors checkpoints are not supported yet.");
    }

    return HfModelFiles{
        .model_dir = model_dir,
        .config_path = detail::require_file(model_dir, "config.json"),
        .weights_path = detail::require_file(model_dir, "model.safetensors"),
        .tokenizer_json_path = detail::require_file(model_dir, "tokenizer.json"),
        .tokenizer_config_path = detail::require_file(model_dir, "tokenizer_config.json"),
        .generation_config_path = detail::optional_file(model_dir, "generation_config.json"),
        .merges_path = detail::optional_file(model_dir, "merges.txt"),
        .vocab_path = detail::optional_file(model_dir, "vocab.json"),
    };
}

HfConfig HfModelFiles::load_config() const {
    return HfConfig::from_file(config_path);
}

files::SafetensorsFile HfModelFiles::load_weights() const {
    return files::SafetensorsFile::from_file(weights_path);
}

} // namespace cppinf::loaders::hf
