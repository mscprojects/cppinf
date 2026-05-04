#pragma once

#include <cstddef>
#include <string_view>

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn::detail {

// Applies a transformer linear projection with weight layout [out_features, in_features] to rank-2 or rank-3 input.
tensors::Tensor linear_project(const tensors::TensorView& input, const tensors::TensorView& weight,
                               std::string_view result_name);

// Validates that a projection weight is rank-2 and matches the expected [output_size, input_size] layout.
void validate_projection_weight(const tensors::TensorView& weight, std::string_view name, std::size_t output_size,
                                std::size_t input_size);

} // namespace cppinf::nn::detail
