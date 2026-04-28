#pragma once

#include <cstddef>
#include <span>
#include <string>

#include "tensor_info.h"

namespace cppinf::tensors {

class TensorView {
public:
  TensorView(TensorInfo tensor_info, std::span<const std::byte> data);

  const TensorInfo &tensor_info() const;
  std::span<const std::byte> data() const;
  std::size_t byte_size() const;

private:
  TensorInfo tensor_info_;
  std::span<const std::byte> data_;
};

std::string to_string(const TensorView &tensor_view);

} // namespace cppinf::tensors
