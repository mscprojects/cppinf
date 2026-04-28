#include "tensor_view.h"

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

namespace cppinf::tensors {

TensorView::TensorView(TensorInfo tensor_info, std::span<const std::byte> data)
    : tensor_info_(std::move(tensor_info)), data_(data) {
    if (data_.size() != tensor_info_.byte_size()) {
        throw std::invalid_argument("TensorView byte size does not match TensorInfo.");
    }
}

const TensorInfo& TensorView::tensor_info() const {
    return tensor_info_;
}

std::span<const std::byte> TensorView::data() const {
    return data_;
}

std::size_t TensorView::byte_size() const {
    return data_.size();
}

std::string to_string(const TensorView& tensor_view) {
    return fmt::format("TensorView(name=\"{}\", dtype={}, shape={}, bytes={})", tensor_view.tensor_info().name,
                       to_string(tensor_view.tensor_info().dtype), to_string(tensor_view.tensor_info().shape),
                       tensor_view.byte_size());
}

} // namespace cppinf::tensors
