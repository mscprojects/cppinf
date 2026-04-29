#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "tensors/shape.h"
#include "tensors/tensor.h"
#include "tensors/tensor_info.h"

namespace cppinf::tensors {

// Returns zero-offset metadata for a newly created output tensor.
inline TensorInfo make_result_tensor_info(std::string_view name, DType dtype, const Shape& shape) {
    return TensorInfo{
        .name = std::string(name),
        .dtype = dtype,
        .shape = shape,
        .byte_offset = 0,
    };
}

// Returns an owned tensor copy with the same bytes and a different tensor name.
inline Tensor rename_tensor(std::string_view name, const Tensor& tensor) {
    return Tensor(make_result_tensor_info(name, tensor.tensor_info().dtype, tensor.tensor_info().shape),
                  std::vector<std::byte>(tensor.bytes().begin(), tensor.bytes().end()));
}

} // namespace cppinf::tensors
