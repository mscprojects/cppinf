#pragma once

#include "tensors/dtype.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops::detail {

tensors::Tensor one_dnn_add(const tensors::TensorView& lhs, const tensors::TensorView& rhs);
tensors::Tensor one_dnn_mul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);
tensors::Tensor one_dnn_matmul(const tensors::TensorView& lhs, const tensors::TensorView& rhs);
tensors::Tensor one_dnn_cast(const tensors::TensorView& input, tensors::DType dtype);
tensors::Tensor one_dnn_transpose_2d(const tensors::TensorView& input);
tensors::Tensor one_dnn_silu(const tensors::TensorView& input);
tensors::Tensor one_dnn_softmax_last_dim(const tensors::TensorView& input);
tensors::Tensor one_dnn_rms_norm(const tensors::TensorView& input, const tensors::TensorView& weight, float epsilon);

} // namespace cppinf::ops::detail
