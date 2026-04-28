#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "tensors/dtype.h"

namespace cppinf::loaders::hf {

struct HfConfig {
    std::vector<std::string> architectures;
    std::string model_type;
    std::size_t hidden_size{};
    std::size_t intermediate_size{};
    std::size_t max_position_embeddings{};
    std::size_t num_attention_heads{};
    std::size_t num_hidden_layers{};
    std::size_t num_key_value_heads{};
    std::size_t vocab_size{};
    std::int64_t bos_token_id{};
    std::int64_t eos_token_id{};
    tensors::DType tensor_dtype{};

    bool operator==(const HfConfig&) const = default;

    static HfConfig from_json_text(std::string_view json_text);
    static HfConfig from_file(const std::filesystem::path& path);
};

} // namespace cppinf::loaders::hf
