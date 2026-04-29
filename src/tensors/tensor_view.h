#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "tensor_info.h"

namespace cppinf::tensors {

class TensorView {
  public:
    TensorView(TensorInfo tensor_info, std::span<const std::byte> data, std::vector<std::size_t> strides_bytes = {},
               std::size_t data_offset_bytes = 0);

    const TensorInfo& tensor_info() const;
    std::span<const std::byte> data() const;
    const std::vector<std::size_t>& strides_bytes() const;
    bool is_contiguous() const;
    std::size_t byte_size() const;

  private:
    TensorInfo tensor_info_;
    std::span<const std::byte> data_;
    std::vector<std::size_t> strides_bytes_;
    std::size_t data_offset_bytes_{};
};

std::string to_string(const TensorView& tensor_view);

} // namespace cppinf::tensors
