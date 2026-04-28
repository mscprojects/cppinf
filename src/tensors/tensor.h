#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "tensor_info.h"
#include "tensor_view.h"

namespace cppinf::tensors {

class Tensor {
  public:
    Tensor(TensorInfo tensor_info, std::vector<std::byte> data);

    static Tensor zeros(TensorInfo tensor_info);

    const TensorInfo& tensor_info() const;
    const std::vector<std::byte>& bytes() const;
    std::span<const std::byte> data() const;
    std::span<std::byte> mutable_data();
    std::size_t byte_size() const;
    TensorView view() const;

    bool operator==(const Tensor&) const = default;

  private:
    TensorInfo tensor_info_;
    std::vector<std::byte> data_;
};

std::string to_string(const Tensor& tensor);

} // namespace cppinf::tensors
