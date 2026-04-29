#include "nn/rope.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/one_dnn_utils.h"
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

std::size_t flat_index_rank3(std::size_t dim0, std::size_t dim1, std::size_t dim2,
                             const std::vector<std::int64_t>& dims) {
    const auto size1 = static_cast<std::size_t>(dims[1]);
    const auto size2 = static_cast<std::size_t>(dims[2]);
    return ((dim0 * size1) + dim1) * size2 + dim2;
}

void validate_rope_input(const tensors::TensorView& input, float rope_base, std::size_t sequence_dimension) {
    ops::detail::validate_supported_float_dtype(input.tensor_info().dtype, "apply_rope");
    if (input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("apply_rope requires a rank-3 tensor.");
    }

    if (!std::isfinite(rope_base) || rope_base <= 0.0f) {
        throw std::invalid_argument("apply_rope requires a positive finite rope base.");
    }

    if (sequence_dimension > 1) {
        throw std::invalid_argument("apply_rope requires the sequence dimension to be 0 or 1.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    checked_positive_dim_to_size(dims[0], "rope dim 0");
    checked_positive_dim_to_size(dims[1], "rope dim 1");
    const auto head_size = checked_positive_dim_to_size(dims[2], "rope head size");
    if (head_size % 2 != 0) {
        throw std::invalid_argument("apply_rope requires an even head size.");
    }
}

} // namespace

tensors::Tensor apply_rope(const tensors::TensorView& input, std::size_t sequence_position_offset, float rope_base,
                           std::size_t sequence_dimension) {
    validate_rope_input(input, rope_base, sequence_dimension);

    const auto& dims = input.tensor_info().shape.dims();
    const auto outer_count = static_cast<std::size_t>(dims[1 - sequence_dimension]);
    const auto sequence_length = static_cast<std::size_t>(dims[sequence_dimension]);
    const auto head_size = static_cast<std::size_t>(dims[2]);
    const auto half_head_size = head_size / 2;

    // Build one inverse frequency per feature pair so lower-index pairs rotate slowly and higher-index pairs rotate
    // faster.
    std::vector<float> inverse_frequencies;
    inverse_frequencies.reserve(half_head_size);
    for (std::size_t pair_index = 0; pair_index < half_head_size; ++pair_index) {
        inverse_frequencies.push_back(
            std::pow(rope_base, -static_cast<float>(pair_index) / static_cast<float>(half_head_size)));
    }

    // RoPE computes angles in f32, rotates each paired feature channel by the token position angle, and returns to the
    // caller's dtype at the helper boundary. The outer shape stays [heads, seq, head_dim].
    auto result_f32 = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("rope_result", tensors::DType::F32, input.tensor_info().shape));

    for (std::size_t outer_index = 0; outer_index < outer_count; ++outer_index) {
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto position = static_cast<float>(sequence_position_offset + sequence_index);
            for (std::size_t pair_index = 0; pair_index < half_head_size; ++pair_index) {
                const auto angle = position * inverse_frequencies[pair_index];
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);

                const auto dim0 = sequence_dimension == 0 ? sequence_index : outer_index;
                const auto dim1 = sequence_dimension == 0 ? outer_index : sequence_index;
                const auto first_index = flat_index_rank3(dim0, dim1, pair_index, dims);
                const auto second_index = flat_index_rank3(dim0, dim1, pair_index + half_head_size, dims);

                const auto first_value = ops::detail::load_float_value(input, first_index);
                const auto second_value = ops::detail::load_float_value(input, second_index);

                ops::detail::store_float_value(tensors::DType::F32, result_f32.mutable_data(), first_index,
                                               first_value * cosine - second_value * sine);
                ops::detail::store_float_value(tensors::DType::F32, result_f32.mutable_data(), second_index,
                                               second_value * cosine + first_value * sine);
            }
        }
    }

    return ops::detail::maybe_cast_result(std::move(result_f32), input.tensor_info().dtype, "rope_result");
}

} // namespace cppinf::nn
