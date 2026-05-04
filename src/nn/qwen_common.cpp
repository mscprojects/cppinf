#include "nn/qwen_common.h"

#include <stdexcept>

#include <fmt/format.h>

#include "common/checked.h"
#include "ops/matmul.h"
#include "ops/tensor_ops.h"
#include "tensors/shape.h"
#include "tensors/tensor_utils.h"

namespace cppinf::nn::detail {

tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight,
                               std::string_view result_name) {
    if (input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument(fmt::format("{} requires rank-3 input.", result_name));
    }

    // A transformer linear layer stores weights as [out_features, in_features], so transpose to multiply tokens
    // [..., in_features] by [in_features, out_features].
    const auto transposed_weight = ops::transpose_2d(weight);
    const auto& dims = input.tensor_info().shape.dims();
    const auto batch_size = common::checked_positive_dim_to_size(dims[0], fmt::format("{} batch size", result_name));
    const auto sequence_length =
        common::checked_positive_dim_to_size(dims[1], fmt::format("{} sequence length", result_name));
    const auto hidden_size = common::checked_positive_dim_to_size(dims[2], fmt::format("{} hidden size", result_name));
    const auto flat_input = ops::reshape(input, tensors::Shape({static_cast<std::int64_t>(batch_size * sequence_length),
                                                                static_cast<std::int64_t>(hidden_size)}));
    auto projected = ops::matmul(flat_input, transposed_weight.view());
    return tensors::materialize_tensor(
        result_name, ops::reshape(projected.view(), tensors::Shape({static_cast<std::int64_t>(batch_size),
                                                                    static_cast<std::int64_t>(sequence_length),
                                                                    projected.tensor_info().shape.dims()[1]})));
}

void validate_projection_weight(const tensors::TensorView& weight, std::string_view name, std::size_t output_size,
                                std::size_t input_size) {
    if (weight.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument(fmt::format("{} must be rank-2.", name));
    }

    const auto& dims = weight.tensor_info().shape.dims();
    if (common::checked_positive_dim_to_size(dims[0], fmt::format("{} rows", name)) != output_size ||
        common::checked_positive_dim_to_size(dims[1], fmt::format("{} cols", name)) != input_size) {
        throw std::invalid_argument(fmt::format("{} has an unexpected shape.", name));
    }
}

} // namespace cppinf::nn::detail
