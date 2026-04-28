#include "tensor_info.h"

#include <limits>
#include <stdexcept>

#include <fmt/format.h>

namespace cppinf::tensors {

std::size_t TensorInfo::byte_size() const {
    const std::size_t element_count = shape.num_elements();
    const std::size_t element_size = element_size_bytes(dtype);

    if (element_size != 0 && element_count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("Tensor byte size overflowed.");
    }

    return element_count * element_size;
}

std::string to_string(const TensorInfo& tensor_info) {
    return fmt::format("TensorInfo(name=\"{}\", dtype={}, shape={}, offset={}, bytes={})", tensor_info.name,
                       to_string(tensor_info.dtype), to_string(tensor_info.shape), tensor_info.byte_offset,
                       tensor_info.byte_size());
}

} // namespace cppinf::tensors
