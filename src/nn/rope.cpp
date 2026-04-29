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

std::size_t flat_index(std::size_t head_index, std::size_t sequence_index, std::size_t feature_index,
                       std::size_t sequence_length, std::size_t feature_count) {
    return ((head_index * sequence_length) + sequence_index) * feature_count + feature_index;
}

void validate_rope_input(const tensors::TensorView& input, float rope_base) {
    ops::detail::validate_supported_float_dtype(input.tensor_info().dtype, "apply_rope");
    if (input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("apply_rope requires a rank-3 tensor.");
    }

    if (!std::isfinite(rope_base) || rope_base <= 0.0f) {
        throw std::invalid_argument("apply_rope requires a positive finite rope base.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    checked_positive_dim_to_size(dims[0], "rope heads");
    checked_positive_dim_to_size(dims[1], "rope sequence length");
    const auto head_size = checked_positive_dim_to_size(dims[2], "rope head size");
    if (head_size % 2 != 0) {
        throw std::invalid_argument("apply_rope requires an even head size.");
    }
}

} // namespace

tensors::Tensor apply_rope(const tensors::TensorView& input, std::size_t sequence_position_offset, float rope_base) {
    validate_rope_input(input, rope_base);

    const auto& dims = input.tensor_info().shape.dims();
    const auto head_count = static_cast<std::size_t>(dims[0]);
    const auto sequence_length = static_cast<std::size_t>(dims[1]);
    const auto head_size = static_cast<std::size_t>(dims[2]);
    const auto half_head_size = head_size / 2;

    std::optional<tensors::Tensor> input_storage;
    const auto input_f32 = ops::detail::maybe_cast_to_dtype(input, tensors::DType::F32, input_storage, "rope_result");

    std::vector<float> inverse_frequencies;
    inverse_frequencies.reserve(half_head_size);
    for (std::size_t pair_index = 0; pair_index < half_head_size; ++pair_index) {
        inverse_frequencies.push_back(
            std::pow(rope_base, -static_cast<float>(pair_index) / static_cast<float>(half_head_size)));
    }

    // RoPE computes angles in f32, then returns to the caller's dtype at the helper boundary.
    auto result_f32 = tensors::Tensor::zeros(
        tensors::make_result_tensor_info("rope_result", tensors::DType::F32, input.tensor_info().shape));

    for (std::size_t head_index = 0; head_index < head_count; ++head_index) {
        for (std::size_t sequence_index = 0; sequence_index < sequence_length; ++sequence_index) {
            const auto position = static_cast<float>(sequence_position_offset + sequence_index);
            for (std::size_t pair_index = 0; pair_index < half_head_size; ++pair_index) {
                const auto angle = position * inverse_frequencies[pair_index];
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);

                const auto first_index = flat_index(head_index, sequence_index, pair_index, sequence_length, head_size);
                const auto second_index =
                    flat_index(head_index, sequence_index, pair_index + half_head_size, sequence_length, head_size);

                const auto first_value =
                    ops::detail::load_float_value(input_f32.tensor_info().dtype, input_f32.data(), first_index);
                const auto second_value =
                    ops::detail::load_float_value(input_f32.tensor_info().dtype, input_f32.data(), second_index);

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
