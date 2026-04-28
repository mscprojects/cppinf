#pragma once

#include <cstddef>
#include <string>

#include "dtype.h"
#include "shape.h"

namespace cppinf::tensors {

struct TensorInfo {
    std::string name;
    DType dtype;
    Shape shape;
    std::size_t byte_offset{};

    std::size_t byte_size() const;

    bool operator==(const TensorInfo&) const = default;
};

std::string to_string(const TensorInfo& tensor_info);

} // namespace cppinf::tensors
