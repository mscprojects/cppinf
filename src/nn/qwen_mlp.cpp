#include "nn/qwen_mlp.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/elementwise_ops.h"
#include "ops/matmul.h"
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

tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight) {
    const auto transposed_weight = cppinf::ops::transpose_2d(weight);
    return cppinf::ops::matmul(input, transposed_weight.view());
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
    validate_supported_float_dtype(hidden_states.tensor_info().dtype, "qwen_mlp");
    const auto dtype = hidden_states.tensor_info().dtype;
    if (dtype != weights.gate_proj_weight.tensor_info().dtype || dtype != weights.up_proj_weight.tensor_info().dtype ||
        dtype != weights.down_proj_weight.tensor_info().dtype) {
        throw std::invalid_argument("qwen_mlp requires matching tensor dtypes.");
    }
    if (hidden_states.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("qwen_mlp requires rank-2 hidden states.");
    }

    const auto& hidden_dims = hidden_states.tensor_info().shape.dims();
    checked_positive_dim_to_size(hidden_dims[0], "qwen_mlp sequence length");
    const auto hidden_size = checked_positive_dim_to_size(hidden_dims[1], "qwen_mlp hidden size");
    const auto intermediate_size = checked_positive_dim_to_size(weights.gate_proj_weight.tensor_info().shape.dims()[0],
                                                                "qwen_mlp intermediate size");

    validate_projection_weight(weights.gate_proj_weight, "qwen_mlp gate_proj_weight", intermediate_size, hidden_size);
    validate_projection_weight(weights.up_proj_weight, "qwen_mlp up_proj_weight", intermediate_size, hidden_size);
    validate_projection_weight(weights.down_proj_weight, "qwen_mlp down_proj_weight", hidden_size, intermediate_size);
}

} // namespace

tensors::Tensor qwen_mlp(const tensors::TensorView& hidden_states, const QwenMlpWeights& weights) {
    validate_qwen_mlp_inputs(hidden_states, weights);

    std::optional<tensors::Tensor> hidden_states_storage;
    std::optional<tensors::Tensor> gate_proj_weight_storage;
    std::optional<tensors::Tensor> up_proj_weight_storage;
    std::optional<tensors::Tensor> down_proj_weight_storage;

    const auto hidden_states_f32 = maybe_cast_to_f32(hidden_states, hidden_states_storage);
    const auto gate_proj_weight_f32 = maybe_cast_to_f32(weights.gate_proj_weight, gate_proj_weight_storage);
    const auto up_proj_weight_f32 = maybe_cast_to_f32(weights.up_proj_weight, up_proj_weight_storage);
    const auto down_proj_weight_f32 = maybe_cast_to_f32(weights.down_proj_weight, down_proj_weight_storage);

    const auto gate_projection = linear_project(hidden_states_f32, gate_proj_weight_f32);
    const auto up_projection = linear_project(hidden_states_f32, up_proj_weight_f32);
    const auto activated_gate = cppinf::ops::silu(gate_projection.view());
    const auto gated_projection = cppinf::ops::mul(activated_gate.view(), up_projection.view());
    auto result_f32 = linear_project(gated_projection.view(), down_proj_weight_f32);

    if (hidden_states.tensor_info().dtype == tensors::DType::F32) {
        return rename_tensor("qwen_mlp_result", result_f32);
    }

    return rename_tensor("qwen_mlp_result", cppinf::ops::cast(result_f32.view(), hidden_states.tensor_info().dtype));
}

} // namespace cppinf::nn
