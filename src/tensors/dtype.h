#pragma once

#include <cstddef>
#include <string_view>

namespace cppinf::tensors {

enum class DType {
  F16,
  BF16,
  F32,
  I32,
  I64,
  U8,
};

std::size_t element_size_bytes(DType dtype);
std::string_view to_string(DType dtype);

} // namespace cppinf::tensors
