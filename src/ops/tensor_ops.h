#pragma once

#include <cstddef>

#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops {

tensors::Tensor cast(const tensors::TensorView& input, tensors::DType dtype);
tensors::TensorView reshape(const tensors::TensorView& input, tensors::Shape shape);
tensors::Tensor transpose_2d(const tensors::TensorView& input);
tensors::TensorView narrow(const tensors::TensorView& input, std::size_t dim, std::size_t start, std::size_t length);

} // namespace cppinf::ops
