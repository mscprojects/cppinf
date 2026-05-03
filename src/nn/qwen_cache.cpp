#include "nn/qwen_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ops/tensor_ops.h"
#include "tensors/shape.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_utils.h"

namespace cppinf::nn {
namespace {

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    const auto value = static_cast<std::size_t>(dim);
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

std::int64_t checked_size_to_dim(std::size_t value, std::string_view field_name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(fmt::format("{} does not fit in int64_t.", field_name));
    }

    return static_cast<std::int64_t>(value);
}

void validate_current_cache_input(const tensors::TensorView& current, std::string_view result_name) {
    if (current.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument(fmt::format("{} requires a rank-3 current tensor.", result_name));
    }

    if (!current.is_contiguous()) {
        throw std::invalid_argument(fmt::format("{} requires a contiguous current tensor.", result_name));
    }
}

std::size_t validate_cache_tensor(const tensors::Tensor& cache_tensor, const tensors::TensorView& current,
                                  std::size_t cached_sequence_length, std::string_view result_name) {
    if (!cache_tensor.view().is_contiguous()) {
        throw std::invalid_argument(fmt::format("{} requires a contiguous cache tensor.", result_name));
    }

    const auto& cache_info = cache_tensor.tensor_info();
    const auto& cache_dims = cache_info.shape.dims();
    const auto& current_dims = current.tensor_info().shape.dims();
    if (cache_info.dtype != current.tensor_info().dtype || cache_info.shape.rank() != 3 ||
        cache_dims[1] != current_dims[1] || cache_dims[2] != current_dims[2]) {
        throw std::invalid_argument(fmt::format("{} cache tensor shape does not match current tensor.", result_name));
    }

    const auto capacity = checked_positive_dim_to_size(cache_dims[0], fmt::format("{} capacity", result_name));
    if (cached_sequence_length > capacity) {
        throw std::invalid_argument(fmt::format("{} cached sequence length exceeds cache capacity.", result_name));
    }

    return capacity;
}

tensors::Tensor make_cache_tensor(const tensors::TensorView& current, std::size_t capacity,
                                  std::string_view result_name) {
    std::vector<std::int64_t> cache_dims = current.tensor_info().shape.dims();
    cache_dims[0] = checked_size_to_dim(capacity, fmt::format("{} capacity", result_name));
    return tensors::Tensor::zeros(tensors::make_result_tensor_info(result_name, current.tensor_info().dtype,
                                                                   tensors::Shape(std::move(cache_dims))));
}

std::size_t grow_cache_capacity(std::size_t capacity, std::size_t required_capacity, std::string_view result_name) {
    if (required_capacity <= capacity) {
        return capacity;
    }

    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
        return required_capacity;
    }

    const auto doubled_capacity = capacity * 2;
    if (doubled_capacity < capacity) {
        throw std::overflow_error(fmt::format("{} capacity overflowed.", result_name));
    }

    return std::max(required_capacity, doubled_capacity);
}

void ensure_cache_tensor_capacity(std::optional<tensors::Tensor>& cache_tensor, const tensors::TensorView& current,
                                  std::size_t cached_sequence_length, std::size_t required_capacity,
                                  std::string_view result_name) {
    validate_current_cache_input(current, result_name);

    if (!cache_tensor.has_value()) {
        // First append establishes the backing shape and allocates exactly the requested sequence capacity.
        cache_tensor = make_cache_tensor(current, required_capacity, result_name);
        return;
    }

    const auto capacity = validate_cache_tensor(*cache_tensor, current, cached_sequence_length, result_name);
    if (required_capacity <= capacity) {
        return;
    }

    const auto new_capacity = grow_cache_capacity(capacity, required_capacity, result_name);
    auto grown_cache_tensor = make_cache_tensor(current, new_capacity, result_name);

    // Preserve the filled prefix [cached_sequence, heads, head_dim], unused capacity is intentionally left zeroed.
    const auto& current_dims = current.tensor_info().shape.dims();
    const auto row_byte_size = checked_positive_dim_to_size(current_dims[1], fmt::format("{} heads", result_name)) *
                               checked_positive_dim_to_size(current_dims[2], fmt::format("{} head dim", result_name)) *
                               tensors::element_size_bytes(current.tensor_info().dtype);
    const auto filled_byte_size = cached_sequence_length * row_byte_size;
    std::memcpy(grown_cache_tensor.mutable_data().data(), cache_tensor->data().data(), filled_byte_size);
    cache_tensor = std::move(grown_cache_tensor);
}

void append_sequence_to_cache(std::optional<tensors::Tensor>& cache_tensor, const tensors::TensorView& current,
                              std::size_t cached_sequence_length, std::string_view result_name) {
    const auto& current_dims = current.tensor_info().shape.dims();
    const auto current_sequence_length =
        checked_positive_dim_to_size(current_dims[0], fmt::format("{} sequence", result_name));
    const auto required_sequence_length = cached_sequence_length + current_sequence_length;
    if (required_sequence_length < cached_sequence_length) {
        throw std::overflow_error(fmt::format("{} sequence length overflowed.", result_name));
    }

    ensure_cache_tensor_capacity(cache_tensor, current, cached_sequence_length, required_sequence_length, result_name);

    // Cache tensors are dense [max_sequence, kv_heads, head_dim], so appending is one contiguous row-block copy.
    const auto destination_offset =
        cached_sequence_length * checked_positive_dim_to_size(current_dims[1], fmt::format("{} heads", result_name)) *
        checked_positive_dim_to_size(current_dims[2], fmt::format("{} head dim", result_name)) *
        tensors::element_size_bytes(current.tensor_info().dtype);
    std::memcpy(cache_tensor->mutable_data().data() + destination_offset, current.data().data(), current.byte_size());
}

const tensors::Tensor& require_cache_tensor(const std::optional<tensors::Tensor>& cache_tensor,
                                            std::string_view result_name) {
    if (!cache_tensor.has_value()) {
        throw std::invalid_argument(fmt::format("{} has not been allocated.", result_name));
    }

    return *cache_tensor;
}

} // namespace

QwenAttentionCache make_qwen_attention_cache(std::string_view key_name, std::string_view value_name,
                                             tensors::DType dtype, std::size_t max_sequence_length,
                                             std::size_t num_key_value_heads, std::size_t head_dim) {
    if (max_sequence_length == 0 || num_key_value_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("Qwen attention cache requires non-zero dimensions.");
    }

    const auto shape = tensors::Shape({checked_size_to_dim(max_sequence_length, "qwen_attention_cache sequence"),
                                       checked_size_to_dim(num_key_value_heads, "qwen_attention_cache kv heads"),
                                       checked_size_to_dim(head_dim, "qwen_attention_cache head dim")});
    return QwenAttentionCache{
        .key = tensors::Tensor::zeros(tensors::make_result_tensor_info(key_name, dtype, shape)),
        .value = tensors::Tensor::zeros(tensors::make_result_tensor_info(value_name, dtype, shape)),
        .sequence_length = 0,
    };
}

void append_to_qwen_attention_cache(QwenAttentionCache& cache, const tensors::TensorView& key,
                                    const tensors::TensorView& value) {
    const auto sequence_length =
        checked_positive_dim_to_size(key.tensor_info().shape.dims()[0], "qwen_attention cached append sequence");
    if (checked_positive_dim_to_size(value.tensor_info().shape.dims()[0], "qwen_attention cached value sequence") !=
        sequence_length) {
        throw std::invalid_argument("qwen_attention cache requires matching key/value append sequence lengths.");
    }

    append_sequence_to_cache(cache.key, key, cache.sequence_length, "qwen_attention_cached_key");
    append_sequence_to_cache(cache.value, value, cache.sequence_length, "qwen_attention_cached_value");
    cache.sequence_length += sequence_length;
}

tensors::TensorView qwen_attention_cache_key_view(const QwenAttentionCache& cache) {
    return ops::narrow(require_cache_tensor(cache.key, "qwen_attention_cached_key").view(), 0, 0,
                       cache.sequence_length);
}

tensors::TensorView qwen_attention_cache_value_view(const QwenAttentionCache& cache) {
    return ops::narrow(require_cache_tensor(cache.value, "qwen_attention_cached_value").view(), 0, 0,
                       cache.sequence_length);
}

} // namespace cppinf::nn
