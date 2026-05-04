#include "nn/qwen_mlp.h"

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/elementwise_ops.h"
#include "ops/matmul.h"
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

tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight) {
    const auto transposed_weight = ops::transpose_2d(weight);
    if (input.tensor_info().shape.rank() == 2) {
        return ops::matmul(input, transposed_weight.view());
    }

    const auto& input_dims = input.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(input_dims[0], "qwen_mlp linear project batch size");
    const auto sequence_length = checked_positive_dim_to_size(input_dims[1], "qwen_mlp linear project sequence length");
    const auto hidden_size = checked_positive_dim_to_size(input_dims[2], "qwen_mlp linear project hidden size");
    const auto flat_input = ops::reshape(input, tensors::Shape({static_cast<std::int64_t>(batch_size * sequence_length),
                                                                static_cast<std::int64_t>(hidden_size)}));
    auto projected = ops::matmul(flat_input, transposed_weight.view());
    return tensors::materialize_tensor(
        "qwen_mlp_linear_project",
        ops::reshape(projected.view(),
                     tensors::Shape({static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(sequence_length),
                                     projected.tensor_info().shape.dims()[1]})));
}

void validate_projection_weight(const tensors::TensorView& weight, std::string_view name, std::size_t output_size,
                                std::size_t input_size) {
    if (weight.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument(fmt::format("{} must be rank-2.", name));
    }

    const auto& dims = weight.tensor_info().shape.dims();
    if (checked_positive_dim_to_size(dims[0], fmt::format("{} rows", name)) != output_size ||
        checked_positive_dim_to_size(dims[1], fmt::format("{} cols", name)) != input_size) {
        throw std::invalid_argument(fmt::format("{} has an unexpected shape.", name));
    }
}

void validate_qwen_mlp_inputs(const tensors::TensorView& hidden_states, const QwenMlpWeights& weights) {
    ops::detail::validate_supported_float_dtype(hidden_states.tensor_info().dtype, "qwen_mlp");
    const auto dtype = hidden_states.tensor_info().dtype;
    if (dtype != weights.gate_proj_weight.tensor_info().dtype || dtype != weights.up_proj_weight.tensor_info().dtype ||
        dtype != weights.down_proj_weight.tensor_info().dtype) {
        throw std::invalid_argument("qwen_mlp requires matching tensor dtypes.");
    }
    if (hidden_states.tensor_info().shape.rank() != 2 && hidden_states.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("qwen_mlp requires rank-2 or rank-3 hidden states.");
    }

    const auto& hidden_dims = hidden_states.tensor_info().shape.dims();
    for (std::size_t axis = 0; axis + 1 < hidden_dims.size(); ++axis) {
        checked_positive_dim_to_size(hidden_dims[axis], fmt::format("qwen_mlp dim {}", axis));
    }
    const auto hidden_size = checked_positive_dim_to_size(hidden_dims.back(), "qwen_mlp hidden size");
    const auto intermediate_size = checked_positive_dim_to_size(weights.gate_proj_weight.tensor_info().shape.dims()[0],
                                                                "qwen_mlp intermediate size");

    validate_projection_weight(weights.gate_proj_weight, "qwen_mlp gate_proj_weight", intermediate_size, hidden_size);
    validate_projection_weight(weights.up_proj_weight, "qwen_mlp up_proj_weight", intermediate_size, hidden_size);
    validate_projection_weight(weights.down_proj_weight, "qwen_mlp down_proj_weight", hidden_size, intermediate_size);
}

} // namespace

tensors::Tensor qwen_mlp(const tensors::TensorView& hidden_states, const QwenMlpWeights& weights) {
    validate_qwen_mlp_inputs(hidden_states, weights);

    // Two linear projections expand [..., hidden] into [..., intermediate]: one branch will become the gate, and the
    // other carries the candidate values.
    const auto gate_projection = linear_project(hidden_states, weights.gate_proj_weight);
    const auto up_projection = linear_project(hidden_states, weights.up_proj_weight);

    // SiLU turns the gate branch into smooth per-feature coefficients, and multiplying by the up branch keeps or
    // suppresses each intermediate feature.
    const auto activated_gate = ops::silu(gate_projection.view());
    const auto gated_projection = ops::mul(activated_gate.view(), up_projection.view());

    // The down projection maps the gated intermediate activations back to the model width [..., hidden].
    return tensors::rename_tensor("qwen_mlp_result", linear_project(gated_projection.view(), weights.down_proj_weight));
}

} // namespace cppinf::nn
