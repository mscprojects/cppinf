#pragma once

#include <optional>

#include "tensors/dtype.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

struct MatmulOptions {
    std::optional<tensors::DType> output_dtype = std::nullopt;
};

// Multiplies rank-2 [m, k] and [k, n] tensors or rank-3 [b, m, k] and [b, k, n] tensors.
// Supports f32 and bf16 inputs. Same-dtype inputs default to the same output dtype, while mixed f32/bf16 inputs
// default to f32 output. BF16 inputs may request an F32 result to keep low-precision storage while materializing F32
// scores.
tensors::Tensor matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs, MatmulOptions options = {});

} // namespace cppinf::ops
