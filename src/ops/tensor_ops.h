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

// Removes a unit-length dimension without changing the underlying bytes.
tensors::TensorView squeeze(const tensors::TensorView& input, std::size_t dim);

// Transposes a rank-2 tensor of shape [rows, cols] and returns an owning tensor of shape [cols, rows].
tensors::Tensor transpose_2d(const tensors::TensorView& input);

// Transposes the last two dimensions of a rank-2 or rank-3 tensor and returns an owning tensor.
tensors::Tensor transpose_last_two_dims(const tensors::TensorView& input);

// Returns a metadata-only slice along any dimension.
tensors::TensorView narrow(const tensors::TensorView& input, std::size_t dim, std::size_t start, std::size_t length);

} // namespace cppinf::ops
