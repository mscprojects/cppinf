#include "loaders/hf/hf_config.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "io/file.h"
#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace cppinf::loaders::hf {
namespace detail {

using json = nlohmann::json;

const json& require_field(const json& object, std::string_view field_name) {
    if (!object.contains(field_name)) {
        throw std::invalid_argument(fmt::format("Missing required HF config field '{}'.", field_name));
    }

    return object.at(field_name);
}

std::size_t read_size(const json& value, std::string_view field_name) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(fmt::format("HF config field '{}' must be an integer.", field_name));
    }

    const auto parsed_value = value.get<std::int64_t>();
    if (parsed_value < 0) {
        throw std::invalid_argument(fmt::format("HF config field '{}' must be non-negative.", field_name));
    }
    if (static_cast<std::uint64_t>(parsed_value) > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(fmt::format("HF config field '{}' does not fit in size_t.", field_name));
    }

    return static_cast<std::size_t>(parsed_value);
}

std::int64_t read_i64(const json& value, std::string_view field_name) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(fmt::format("HF config field '{}' must be an integer.", field_name));
    }

    return value.get<std::int64_t>();
}

float read_float(const json& value, std::string_view field_name) {
    if (!value.is_number()) {
        throw std::invalid_argument(fmt::format("HF config field '{}' must be numeric.", field_name));
    }

    return value.get<float>();
}

bool read_bool(const json& value, std::string_view field_name) {
    if (!value.is_boolean()) {
        throw std::invalid_argument(fmt::format("HF config field '{}' must be a boolean.", field_name));
    }

    return value.get<bool>();
}

std::vector<std::string> read_architectures(const json& value) {
    if (!value.is_array()) {
        throw std::invalid_argument("HF config field 'architectures' must be an array.");
    }

    std::vector<std::string> architectures;
    architectures.reserve(value.size());
    for (const json& architecture : value) {
        if (!architecture.is_string()) {
            throw std::invalid_argument("HF config field 'architectures' must only contain strings.");
        }

        architectures.push_back(architecture.get<std::string>());
    }

    return architectures;
}

tensors::DType parse_torch_dtype(std::string_view torch_dtype) {
    if (torch_dtype == "float16") {
        return tensors::DType::F16;
    }
    if (torch_dtype == "bfloat16") {
        return tensors::DType::BF16;
    }
    if (torch_dtype == "float32") {
        return tensors::DType::F32;
    }
    if (torch_dtype == "int32") {
        return tensors::DType::I32;
    }
    if (torch_dtype == "int64") {
        return tensors::DType::I64;
    }
    if (torch_dtype == "uint8") {
        return tensors::DType::U8;
    }

    throw std::invalid_argument(fmt::format("Unsupported HF torch_dtype '{}'.", torch_dtype));
}

} // namespace detail

HfConfig HfConfig::from_json_text(std::string_view json_text) {
    detail::json json_config;
    try {
        json_config = detail::json::parse(json_text);
    } catch (const detail::json::parse_error& error) {
        throw std::invalid_argument(fmt::format("Failed to parse HF config JSON: {}", error.what()));
    }

    if (!json_config.is_object()) {
        throw std::invalid_argument("HF config must be a JSON object.");
    }

    return HfConfig{
        .architectures = detail::read_architectures(detail::require_field(json_config, "architectures")),
        .model_type = detail::require_field(json_config, "model_type").get<std::string>(),
        .head_dim = detail::read_size(detail::require_field(json_config, "head_dim"), "head_dim"),
        .hidden_size = detail::read_size(detail::require_field(json_config, "hidden_size"), "hidden_size"),
        .intermediate_size =
            detail::read_size(detail::require_field(json_config, "intermediate_size"), "intermediate_size"),
        .max_position_embeddings =
            detail::read_size(detail::require_field(json_config, "max_position_embeddings"), "max_position_embeddings"),
        .num_attention_heads =
            detail::read_size(detail::require_field(json_config, "num_attention_heads"), "num_attention_heads"),
        .num_hidden_layers =
            detail::read_size(detail::require_field(json_config, "num_hidden_layers"), "num_hidden_layers"),
        .num_key_value_heads =
            detail::read_size(detail::require_field(json_config, "num_key_value_heads"), "num_key_value_heads"),
        .vocab_size = detail::read_size(detail::require_field(json_config, "vocab_size"), "vocab_size"),
        .bos_token_id = detail::read_i64(detail::require_field(json_config, "bos_token_id"), "bos_token_id"),
        .eos_token_id = detail::read_i64(detail::require_field(json_config, "eos_token_id"), "eos_token_id"),
        .rms_norm_eps = detail::read_float(detail::require_field(json_config, "rms_norm_eps"), "rms_norm_eps"),
        .rope_theta = detail::read_float(detail::require_field(json_config, "rope_theta"), "rope_theta"),
        .tie_word_embeddings =
            detail::read_bool(detail::require_field(json_config, "tie_word_embeddings"), "tie_word_embeddings"),
        .tensor_dtype = detail::parse_torch_dtype(detail::require_field(json_config, "torch_dtype").get<std::string>()),
    };
}

HfConfig HfConfig::from_file(const std::filesystem::path& path) {
    return from_json_text(io::read_text_file(path, "HF config file"));
}

} // namespace cppinf::loaders::hf
