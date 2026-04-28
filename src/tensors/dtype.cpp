#include "dtype.h"

#include <stdexcept>

namespace cppinf::tensors {

std::size_t element_size_bytes(DType dtype) {
  switch (dtype) {
  case DType::F16:
  case DType::BF16:
    return 2;
  case DType::F32:
  case DType::I32:
    return 4;
  case DType::I64:
    return 8;
  case DType::U8:
    return 1;
  }

  throw std::invalid_argument("Unsupported DType.");
}

std::string_view to_string(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "f16";
  case DType::BF16:
    return "bf16";
  case DType::F32:
    return "f32";
  case DType::I32:
    return "i32";
  case DType::I64:
    return "i64";
  case DType::U8:
    return "u8";
  }

  throw std::invalid_argument("Unsupported DType.");
}

} // namespace cppinf::tensors
