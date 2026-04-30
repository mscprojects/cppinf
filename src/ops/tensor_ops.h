#pragma once

#include <cstddef>
#include <span>

#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

// Casts a tensor to f32 or bf16.
// Shapes: [d0, ..., dn] -> [d0, ..., dn].
tensors::Tensor cast(const tensors::TensorView& input, tensors::DType dtype);

// Reinterprets a contiguous tensor with the same underlying bytes and element count.
// Shapes: [d0, ..., dn] -> [e0, ..., em].
tensors::TensorView reshape(const tensors::TensorView& input, tensors::Shape shape);

// Removes a unit-length dimension without changing the underlying bytes.
// Shapes: [d0, ..., 1, ..., dn] -> [d0, ..., dn].
tensors::TensorView squeeze(const tensors::TensorView& input, std::size_t dim);

// Transposes a rank-2 tensor and returns an owning tensor.
// Shapes: [rows, cols] -> [cols, rows].
tensors::Tensor transpose_2d(const tensors::TensorView& input);

// Transposes the last two dimensions of a rank-2 or rank-3 tensor and returns an owning tensor.
// Shapes: [rows, cols] -> [cols, rows], or [batch, rows, cols] -> [batch, cols, rows].
tensors::Tensor transpose_last_two_dims(const tensors::TensorView& input);

// Returns a metadata-only slice along any dimension.
// Shapes: [d0, ..., dim_size, ..., dn] -> [d0, ..., length, ..., dn].
tensors::TensorView narrow(const tensors::TensorView& input, std::size_t dim, std::size_t start, std::size_t length);

// Reorders dimensions by output-axis to input-axis mapping, where output_dim[i] = input_dim[axes[i]].
// Shapes: [d0, ..., dn] -> [d_axes[0], ..., d_axes[n]].
tensors::Tensor permute(const tensors::TensorView& input, std::span<const std::size_t> axes);

// Repeats each slice along one dimension consecutively.
// Shapes: [d0, ..., dim_size, ..., dn] -> [d0, ..., dim_size * repeats, ..., dn].
tensors::Tensor repeat_interleave(const tensors::TensorView& input, std::size_t dim, std::size_t repeats);

} // namespace cppinf::ops
