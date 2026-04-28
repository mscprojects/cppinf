#include "tensor.h"

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

namespace cppinf::tensors {

Tensor::Tensor(TensorInfo tensor_info, std::vector<std::byte> data)
    : tensor_info_(std::move(tensor_info)), data_(std::move(data)) {
    if (tensor_info_.byte_offset != 0) {
        throw std::invalid_argument("Owning Tensor byte offset must be zero.");
    }
    if (data_.size() != tensor_info_.byte_size()) {
        throw std::invalid_argument("Tensor byte size does not match TensorInfo.");
    }
}

Tensor Tensor::zeros(TensorInfo tensor_info) {
    const std::size_t byte_size = tensor_info.byte_size();
    return Tensor(std::move(tensor_info), std::vector<std::byte>(byte_size));
}

const TensorInfo& Tensor::tensor_info() const {
    return tensor_info_;
}

const std::vector<std::byte>& Tensor::bytes() const {
    return data_;
}

std::span<const std::byte> Tensor::data() const {
    return data_;
}

std::span<std::byte> Tensor::mutable_data() {
    return data_;
}

std::size_t Tensor::byte_size() const {
    return data_.size();
}

TensorView Tensor::view() const {
    return TensorView(tensor_info_, data_);
}

std::string to_string(const Tensor& tensor) {
    return fmt::format("Tensor(name=\"{}\", dtype={}, shape={}, bytes={})", tensor.tensor_info().name,
                       to_string(tensor.tensor_info().dtype), to_string(tensor.tensor_info().shape),
                       tensor.byte_size());
}

} // namespace cppinf::tensors
