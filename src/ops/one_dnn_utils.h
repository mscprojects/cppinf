#pragma once

#include <optional>
#include <span>
#include <string_view>

#include <oneapi/dnnl/dnnl.hpp>

#include "tensors/dtype.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::ops::detail {

const dnnl::engine& cpu_engine();
tensors::Tensor make_result_tensor(std::string_view name, tensors::DType dtype, const tensors::Shape& shape);
dnnl::memory::data_type to_dnnl_dtype(tensors::DType dtype);
dnnl::memory::desc make_dense_desc(const tensors::Shape& shape, tensors::DType dtype, std::string_view op_name);
dnnl::memory make_memory(const dnnl::memory::desc& desc, std::span<const std::byte> bytes);
dnnl::memory make_memory(const dnnl::memory::desc& desc, std::span<std::byte> bytes);
tensors::Tensor cast_with_one_dnn(const tensors::TensorView& input, tensors::DType dtype, std::string_view result_name);
tensors::TensorView maybe_cast_to_dtype(const tensors::TensorView& input, tensors::DType dtype,
                                        std::optional<tensors::Tensor>& storage, std::string_view result_name);
tensors::Tensor maybe_cast_result(tensors::Tensor tensor, tensors::DType dtype, std::string_view result_name);
tensors::Tensor binary_with_one_dnn(std::string_view result_name, const tensors::TensorView& lhs,
                                    const tensors::TensorView& rhs, dnnl::algorithm algorithm,
                                    tensors::DType output_dtype);
tensors::Tensor unary_with_one_dnn(std::string_view result_name, const tensors::TensorView& input,
                                   dnnl::algorithm algorithm, tensors::DType output_dtype, float alpha = 0.0f,
                                   float beta = 0.0f);

} // namespace cppinf::ops::detail
