#include "nn/qwen_decoder_block.h"

#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/elementwise_ops.h"
#include "ops/nn_ops.h"
#include "ops/op_utils.h"
#include "ops/tensor_ops.h"
#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_utils.h"

namespace cppinf::nn {
namespace {

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

void validate_norm_weight(const tensors::TensorView& weight, std::string_view name, std::size_t hidden_size) {
    if (weight.tensor_info().shape.rank() != 1) {
        throw std::invalid_argument(fmt::format("{} must be rank-1.", name));
    }

    if (checked_positive_dim_to_size(weight.tensor_info().shape.dims()[0], fmt::format("{} size", name)) !=
        hidden_size) {
        throw std::invalid_argument(fmt::format("{} must match hidden size.", name));
    }
}

void validate_qwen_decoder_block_inputs(const tensors::TensorView& hidden_states,
                                        const QwenDecoderBlockWeights& weights, float norm_epsilon, float rope_base) {
    ops::detail::validate_supported_float_dtype(hidden_states.tensor_info().dtype, "qwen_decoder_block");
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
}

} // namespace

tensors::Tensor qwen_decoder_block(const tensors::TensorView& hidden_states, const QwenDecoderBlockWeights& weights,
                                   std::size_t num_attention_heads, std::size_t num_key_value_heads,
                                   std::size_t head_dim, float norm_epsilon, std::size_t sequence_position_offset,
                                   float rope_base) {
    validate_qwen_decoder_block_inputs(hidden_states, weights, norm_epsilon, rope_base);

    // RMSNorm prepares the residual stream [seq, hidden] for attention without changing its outer shape.
    const auto attention_input = ops::rms_norm(hidden_states, weights.input_layernorm_weight, norm_epsilon);

    // Self-attention mixes information across the visible prefix and returns another [seq, hidden] tensor, which we
    // add back to the residual stream.
    const auto attention_output =
        qwen_attention(attention_input.view(), weights.attention, num_attention_heads, num_key_value_heads, head_dim,
                       norm_epsilon, sequence_position_offset, rope_base);
    const auto attention_residual = ops::add(hidden_states, attention_output.view());

    // A second RMSNorm prepares the post-attention residual for the feed-forward branch.
    const auto mlp_input =
        ops::rms_norm(attention_residual.view(), weights.post_attention_layernorm_weight, norm_epsilon);

    // The Qwen MLP expands, gates, and projects features back to [seq, hidden], then the second residual edge closes
    // the block.
    const auto mlp_output = qwen_mlp(mlp_input.view(), weights.mlp);
    return tensors::rename_tensor("qwen_decoder_block_result", ops::add(attention_residual.view(), mlp_output.view()));
}

} // namespace cppinf::nn
