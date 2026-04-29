#include "dtype.h"

#include <stdexcept>

namespace cppinf::tensors {

std::size_t element_size_bytes(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return 2;
    case DType::F32:
        return 4;
    }

    throw std::invalid_argument("Unsupported DType.");
}

DType parse_dtype(std::string_view dtype_name) {
    if (dtype_name == "BF16") {
        return DType::BF16;
    }

    if (dtype_name == "F32") {
        return DType::F32;
    }

    throw std::invalid_argument("Unsupported safetensors dtype. Only BF16 and F32 are supported.");
}

std::string_view to_string(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "bf16";
    case DType::F32:
        return "f32";
    }

    throw std::invalid_argument("Unsupported DType.");
}

} // namespace cppinf::tensors
