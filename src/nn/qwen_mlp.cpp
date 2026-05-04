#include "nn/qwen_mlp.h"

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/checked.h"
#include "nn/qwen_common.h"
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

void validate_qwen_mlp_inputs(const tensors::TensorView& hidden_states, const QwenMlpWeights& weights) {
    ops::detail::validate_supported_float_dtype(hidden_states.tensor_info().dtype, "qwen_mlp");
    const auto dtype = hidden_states.tensor_info().dtype;
    if (dtype != weights.gate_proj_weight.tensor_info().dtype || dtype != weights.up_proj_weight.tensor_info().dtype ||
        dtype != weights.down_proj_weight.tensor_info().dtype) {
        throw std::invalid_argument("qwen_mlp requires matching tensor dtypes.");
    }
    if (hidden_states.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("qwen_mlp requires rank-3 hidden states.");
    }

    const auto& hidden_dims = hidden_states.tensor_info().shape.dims();
    for (std::size_t axis = 0; axis + 1 < hidden_dims.size(); ++axis) {
        common::checked_positive_dim_to_size(hidden_dims[axis], fmt::format("qwen_mlp dim {}", axis));
    }
    const auto hidden_size = common::checked_positive_dim_to_size(hidden_dims.back(), "qwen_mlp hidden size");
    const auto intermediate_size = common::checked_positive_dim_to_size(
        weights.gate_proj_weight.tensor_info().shape.dims()[0], "qwen_mlp intermediate size");

    detail::validate_projection_weight(weights.gate_proj_weight, "qwen_mlp gate_proj_weight", intermediate_size,
                                       hidden_size);
    detail::validate_projection_weight(weights.up_proj_weight, "qwen_mlp up_proj_weight", intermediate_size,
                                       hidden_size);
    detail::validate_projection_weight(weights.down_proj_weight, "qwen_mlp down_proj_weight", hidden_size,
                                       intermediate_size);
}

} // namespace

tensors::Tensor qwen_mlp(const tensors::TensorView& hidden_states, const QwenMlpWeights& weights) {
    validate_qwen_mlp_inputs(hidden_states, weights);

    // Two linear projections expand [..., hidden] into [..., intermediate]: one branch will become the gate, and the
    // other carries the candidate values.
    const auto gate_projection = detail::linear_project(hidden_states, weights.gate_proj_weight, "qwen_mlp_gate_proj");
    const auto up_projection = detail::linear_project(hidden_states, weights.up_proj_weight, "qwen_mlp_up_proj");

    // SiLU turns the gate branch into smooth per-feature coefficients, and multiplying by the up branch keeps or
    // suppresses each intermediate feature.
    const auto activated_gate = ops::silu(gate_projection.view());
    const auto gated_projection = ops::mul(activated_gate.view(), up_projection.view());

    // The down projection maps the gated intermediate activations back to the model width [..., hidden].
    return tensors::rename_tensor(
        "qwen_mlp_result",
        detail::linear_project(gated_projection.view(), weights.down_proj_weight, "qwen_mlp_down_proj"));
}

} // namespace cppinf::nn
