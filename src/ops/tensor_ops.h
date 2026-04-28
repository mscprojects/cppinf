#pragma once

#include <cstddef>

#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Casts a tensor to f32 or bf16 without changing its shape.
tensors::Tensor cast(const tensors::TensorView& input, tensors::DType dtype);

// Returns a metadata-only view with the requested shape and the same underlying bytes.
tensors::TensorView reshape(const tensors::TensorView& input, tensors::Shape shape);

// Transposes a rank-2 tensor of shape [rows, cols] and returns an owning tensor of shape [cols, rows].
tensors::Tensor transpose_2d(const tensors::TensorView& input);

// Returns a contiguous dim=0 slice as a view with the same dtype and trailing dimensions.
tensors::TensorView narrow(const tensors::TensorView& input, std::size_t dim, std::size_t start, std::size_t length);

} // namespace cppinf::ops
