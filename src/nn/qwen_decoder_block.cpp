#include "nn/qwen_decoder_block.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/elementwise_ops.h"
#include "ops/nn_ops.h"
#include "ops/tensor_ops.h"
#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor_info.h"

namespace cppinf::nn {
namespace {

tensors::TensorInfo make_result_info(std::string_view name, tensors::DType dtype, const tensors::Shape& shape) {
    return tensors::TensorInfo{
        .name = std::string(name),
        .dtype = dtype,
        .shape = shape,
        .byte_offset = 0,
    };
}

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    const auto value = static_cast<std::size_t>(dim);
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

void validate_supported_float_dtype(tensors::DType dtype, std::string_view op_name) {
    switch (dtype) {
    case tensors::DType::BF16:
    case tensors::DType::F32:
        return;
    case tensors::DType::F16:
    case tensors::DType::I32:
    case tensors::DType::I64:
    case tensors::DType::U8:
        throw std::invalid_argument(fmt::format("{} currently supports only f32 and bf16 tensors.", op_name));
    }

    throw std::invalid_argument(fmt::format("{} received an unsupported dtype.", op_name));
}

tensors::TensorView maybe_cast_to_f32(const tensors::TensorView& input, std::optional<tensors::Tensor>& storage) {
    if (input.tensor_info().dtype == tensors::DType::F32) {
        return input;
    }

    storage.emplace(cppinf::ops::cast(input, tensors::DType::F32));
    return storage->view();
}

tensors::Tensor rename_tensor(std::string_view name, const tensors::Tensor& tensor) {
    return tensors::Tensor(make_result_info(name, tensor.tensor_info().dtype, tensor.tensor_info().shape),
                           std::vector<std::byte>(tensor.bytes().begin(), tensor.bytes().end()));
}

void validate_norm_weight(const tensors::TensorView& weight, std::string_view name, std::size_t hidden_size) {
    if (weight.tensor_info().shape.rank() != 1) {
        throw std::invalid_argument(fmt::format("{} must be rank-1.", name));
    }
    if (checked_positive_dim_to_size(weight.tensor_info().shape.dims()[0], fmt::format("{} size", name)) != hidden_size) {
        throw std::invalid_argument(fmt::format("{} must match hidden size.", name));
    }
}

void validate_qwen_decoder_block_inputs(const tensors::TensorView& hidden_states, const QwenDecoderBlockWeights& weights,
                                        std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                        std::size_t head_dim, float norm_epsilon, float rope_base) {
    validate_supported_float_dtype(hidden_states.tensor_info().dtype, "qwen_decoder_block");
    if (!std::isfinite(norm_epsilon) || norm_epsilon < 0.0f) {
        throw std::invalid_argument("qwen_decoder_block requires a non-negative finite norm epsilon.");
    }
    if (!std::isfinite(rope_base) || rope_base <= 0.0f) {
        throw std::invalid_argument("qwen_decoder_block requires a positive finite rope base.");
    }
    if (hidden_states.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("qwen_decoder_block requires rank-2 hidden states.");
    }

    const auto dtype = hidden_states.tensor_info().dtype;
    if (dtype != weights.input_layernorm_weight.tensor_info().dtype ||
        dtype != weights.post_attention_layernorm_weight.tensor_info().dtype) {
        throw std::invalid_argument("qwen_decoder_block requires matching tensor dtypes.");
    }

    const auto& hidden_dims = hidden_states.tensor_info().shape.dims();
    checked_positive_dim_to_size(hidden_dims[0], "qwen_decoder_block sequence length");
    const auto hidden_size = checked_positive_dim_to_size(hidden_dims[1], "qwen_decoder_block hidden size");
    validate_norm_weight(weights.input_layernorm_weight, "qwen_decoder_block input_layernorm_weight", hidden_size);
    validate_norm_weight(weights.post_attention_layernorm_weight, "qwen_decoder_block post_attention_layernorm_weight",
                         hidden_size);

    static_cast<void>(num_attention_heads);
    static_cast<void>(num_key_value_heads);
    static_cast<void>(head_dim);
}

} // namespace

tensors::Tensor qwen_decoder_block(const tensors::TensorView& hidden_states, const QwenDecoderBlockWeights& weights,
                                   std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                   std::size_t head_dim, float norm_epsilon, std::size_t sequence_position_offset,
                                   float rope_base) {
    validate_qwen_decoder_block_inputs(hidden_states, weights, num_attention_heads, num_key_value_heads, head_dim,
                                       norm_epsilon, rope_base);

    std::optional<tensors::Tensor> hidden_states_storage;
    std::optional<tensors::Tensor> input_layernorm_storage;
    std::optional<tensors::Tensor> post_attention_layernorm_storage;
    std::optional<tensors::Tensor> q_proj_storage;
    std::optional<tensors::Tensor> q_norm_storage;
    std::optional<tensors::Tensor> k_proj_storage;
    std::optional<tensors::Tensor> k_norm_storage;
    std::optional<tensors::Tensor> v_proj_storage;
    std::optional<tensors::Tensor> o_proj_storage;
    std::optional<tensors::Tensor> gate_proj_storage;
    std::optional<tensors::Tensor> up_proj_storage;
    std::optional<tensors::Tensor> down_proj_storage;

    const auto hidden_states_f32 = maybe_cast_to_f32(hidden_states, hidden_states_storage);
    const auto input_layernorm_weight_f32 = maybe_cast_to_f32(weights.input_layernorm_weight, input_layernorm_storage);
    const auto post_attention_layernorm_weight_f32 =
        maybe_cast_to_f32(weights.post_attention_layernorm_weight, post_attention_layernorm_storage);

    const auto q_proj_weight_f32 = maybe_cast_to_f32(weights.attention.q_proj_weight, q_proj_storage);
    const auto q_norm_weight_f32 = maybe_cast_to_f32(weights.attention.q_norm_weight, q_norm_storage);
    const auto k_proj_weight_f32 = maybe_cast_to_f32(weights.attention.k_proj_weight, k_proj_storage);
    const auto k_norm_weight_f32 = maybe_cast_to_f32(weights.attention.k_norm_weight, k_norm_storage);
    const auto v_proj_weight_f32 = maybe_cast_to_f32(weights.attention.v_proj_weight, v_proj_storage);
    const auto o_proj_weight_f32 = maybe_cast_to_f32(weights.attention.o_proj_weight, o_proj_storage);

    const auto gate_proj_weight_f32 = maybe_cast_to_f32(weights.mlp.gate_proj_weight, gate_proj_storage);
    const auto up_proj_weight_f32 = maybe_cast_to_f32(weights.mlp.up_proj_weight, up_proj_storage);
    const auto down_proj_weight_f32 = maybe_cast_to_f32(weights.mlp.down_proj_weight, down_proj_storage);

    const auto attention_weights_f32 = QwenAttentionWeights{
        .q_proj_weight = q_proj_weight_f32,
        .q_norm_weight = q_norm_weight_f32,
        .k_proj_weight = k_proj_weight_f32,
        .k_norm_weight = k_norm_weight_f32,
        .v_proj_weight = v_proj_weight_f32,
        .o_proj_weight = o_proj_weight_f32,
    };
    const auto mlp_weights_f32 = QwenMlpWeights{
        .gate_proj_weight = gate_proj_weight_f32,
        .up_proj_weight = up_proj_weight_f32,
        .down_proj_weight = down_proj_weight_f32,
    };

    const auto attention_input = cppinf::ops::rms_norm(hidden_states_f32, input_layernorm_weight_f32, norm_epsilon);
    const auto attention_output =
        qwen_attention(attention_input.view(), attention_weights_f32, num_attention_heads, num_key_value_heads,
                       head_dim, norm_epsilon, sequence_position_offset, rope_base);
    const auto attention_residual = cppinf::ops::add(hidden_states_f32, attention_output.view());
    const auto mlp_input =
        cppinf::ops::rms_norm(attention_residual.view(), post_attention_layernorm_weight_f32, norm_epsilon);
    const auto mlp_output = qwen_mlp(mlp_input.view(), mlp_weights_f32);
    auto result_f32 = cppinf::ops::add(attention_residual.view(), mlp_output.view());

    if (hidden_states.tensor_info().dtype == tensors::DType::F32) {
        return rename_tensor("qwen_decoder_block_result", result_f32);
    }

    return rename_tensor("qwen_decoder_block_result",
                         cppinf::ops::cast(result_f32.view(), hidden_states.tensor_info().dtype));
}

} // namespace cppinf::nn
