#pragma once

#include <filesystem>
#include <optional>

#include "files/safetensors_file.h"
#include "loaders/hf/hf_config.h"

namespace cppinf::loaders::hf {

struct HfModelFiles {
    std::filesystem::path model_dir;
    std::filesystem::path config_path;
    std::filesystem::path weights_path;
    std::filesystem::path tokenizer_json_path;
    std::filesystem::path tokenizer_config_path;
    std::optional<std::filesystem::path> generation_config_path;
    std::optional<std::filesystem::path> merges_path;
    std::optional<std::filesystem::path> vocab_path;

    bool operator==(const HfModelFiles&) const = default;

    static HfModelFiles from_dir(const std::filesystem::path& model_dir);

    HfConfig load_config() const;
    files::SafetensorsFile load_weights() const;
};

} // namespace cppinf::loaders::hf
