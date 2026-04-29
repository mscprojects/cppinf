#include "tensor_view.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace cppinf::tensors {
namespace {

std::vector<std::size_t> make_dense_strides_bytes(const TensorInfo& tensor_info) {
    std::vector<std::size_t> strides(tensor_info.shape.rank());
    std::size_t stride = element_size_bytes(tensor_info.dtype);
    const auto& dims = tensor_info.shape.dims();
    for (std::size_t index = dims.size(); index-- > 0;) {
        strides[index] = stride;

        const auto dim = static_cast<std::size_t>(dims[index]);
        if (dim != 0 && stride > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("TensorView strides overflowed.");
        }
        stride *= dim;
    }

    return strides;
}

std::size_t covered_byte_size(const TensorInfo& tensor_info, const std::vector<std::size_t>& strides_bytes) {
    const auto element_size = element_size_bytes(tensor_info.dtype);
    if (tensor_info.shape.num_elements() == 0) {
        return 0;
    }

    std::size_t max_offset = 0;
    const auto& dims = tensor_info.shape.dims();
    for (std::size_t index = 0; index < dims.size(); ++index) {
        const auto dim = static_cast<std::size_t>(dims[index]);
        if (dim == 0) {
            return 0;
        }

        const auto span = (dim - 1) * strides_bytes[index];
        if (span > std::numeric_limits<std::size_t>::max() - max_offset) {
            throw std::overflow_error("TensorView covered byte size overflowed.");
        }
        max_offset += span;
    }

    if (element_size > std::numeric_limits<std::size_t>::max() - max_offset) {
        throw std::overflow_error("TensorView covered byte size overflowed.");
    }

    return max_offset + element_size;
}

} // namespace

TensorView::TensorView(TensorInfo tensor_info, std::span<const std::byte> data, std::vector<std::size_t> strides_bytes,
                       std::size_t data_offset_bytes)
    : tensor_info_(std::move(tensor_info)), data_(data), strides_bytes_(std::move(strides_bytes)),
      data_offset_bytes_(data_offset_bytes) {
    if (strides_bytes_.empty()) {
        strides_bytes_ = make_dense_strides_bytes(tensor_info_);
    }

    if (strides_bytes_.size() != tensor_info_.shape.rank()) {
        throw std::invalid_argument("TensorView strides must match tensor rank.");
    }

    if (data_offset_bytes_ > data_.size()) {
        throw std::invalid_argument("TensorView data offset is out of bounds.");
    }

    const auto required_size = covered_byte_size(tensor_info_, strides_bytes_);
    if (required_size > data_.size() - data_offset_bytes_) {
        throw std::invalid_argument("TensorView backing bytes do not cover the requested view.");
    }
}

const TensorInfo& TensorView::tensor_info() const {
    return tensor_info_;
}

std::span<const std::byte> TensorView::data() const {
    return data_.subspan(data_offset_bytes_);
}

const std::vector<std::size_t>& TensorView::strides_bytes() const {
    return strides_bytes_;
}

bool TensorView::is_contiguous() const {
    return strides_bytes_ == make_dense_strides_bytes(tensor_info_);
}

std::size_t TensorView::byte_size() const {
    return tensor_info_.byte_size();
}

std::string to_string(const TensorView& tensor_view) {
    return fmt::format("TensorView(name=\"{}\", dtype={}, shape={}, bytes={})", tensor_view.tensor_info().name,
                       to_string(tensor_view.tensor_info().dtype), to_string(tensor_view.tensor_info().shape),
                       tensor_view.byte_size());
}

} // namespace cppinf::tensors
