#pragma once

#include <cstddef>
#include <string_view>

namespace cppinf::tensors {

enum class DType {
    BF16,
    F32,
};

std::size_t element_size_bytes(DType dtype);
DType parse_dtype(std::string_view dtype_name);
std::string_view to_string(DType dtype);

} // namespace cppinf::tensors
