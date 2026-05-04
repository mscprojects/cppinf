#include "nn/qwen_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "ops/tensor_ops.h"
#include "tensors/shape.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_utils.h"

namespace cppinf::nn {
namespace {

std::size_t checked_non_negative_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    return static_cast<std::size_t>(dim);
}

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    const auto value = checked_non_negative_dim_to_size(dim, field_name);
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

std::size_t cache_row_byte_size(const std::vector<std::int64_t>& dims, tensors::DType dtype,
                                std::string_view result_name) {
    return checked_positive_dim_to_size(dims[2], fmt::format("{} heads", result_name)) *
           checked_positive_dim_to_size(dims[3], fmt::format("{} head dim", result_name)) *
           tensors::element_size_bytes(dtype);
}

void validate_batch_lengths(std::span<const std::size_t> sequence_lengths, std::size_t batch_size,
                            std::string_view result_name) {
    if (sequence_lengths.size() != batch_size) {
        throw std::invalid_argument(fmt::format("{} sequence_lengths must match the batch size.", result_name));
    }
}

void validate_current_cache_input(const tensors::TensorView& current, std::span<const std::size_t> sequence_lengths,
                                  std::string_view result_name) {
    if (current.tensor_info().shape.rank() != 4) {
        throw std::invalid_argument(fmt::format("{} requires a rank-4 current tensor.", result_name));
    }

    if (!current.is_contiguous()) {
        throw std::invalid_argument(fmt::format("{} requires a contiguous current tensor.", result_name));
    }

    const auto& current_dims = current.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(current_dims[0], fmt::format("{} batch size", result_name));
    const auto current_sequence_capacity =
        checked_positive_dim_to_size(current_dims[1], fmt::format("{} sequence capacity", result_name));
    validate_batch_lengths(sequence_lengths, batch_size, result_name);
    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        if (sequence_lengths[batch_index] > current_sequence_capacity) {
            throw std::invalid_argument(
                fmt::format("{} sequence length exceeds the current tensor capacity.", result_name));
        }
    }
}

std::size_t validate_cache_tensor(const tensors::Tensor& cache_tensor, const tensors::TensorView& current,
                                  const std::vector<std::size_t>& cached_sequence_lengths,
                                  std::string_view result_name) {
    if (!cache_tensor.view().is_contiguous()) {
        throw std::invalid_argument(fmt::format("{} requires a contiguous cache tensor.", result_name));
    }

    const auto& cache_info = cache_tensor.tensor_info();
    const auto& cache_dims = cache_info.shape.dims();
    const auto& current_dims = current.tensor_info().shape.dims();
    if (cache_info.dtype != current.tensor_info().dtype || cache_info.shape.rank() != 4 ||
        cache_dims[0] != current_dims[0] || cache_dims[2] != current_dims[2] || cache_dims[3] != current_dims[3]) {
        throw std::invalid_argument(fmt::format("{} cache tensor shape does not match current tensor.", result_name));
    }

    const auto batch_size = checked_positive_dim_to_size(cache_dims[0], fmt::format("{} batch size", result_name));
    validate_batch_lengths(cached_sequence_lengths, batch_size, result_name);
    const auto capacity = checked_positive_dim_to_size(cache_dims[1], fmt::format("{} capacity", result_name));
    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        if (cached_sequence_lengths[batch_index] > capacity) {
            throw std::invalid_argument(fmt::format("{} cached sequence length exceeds cache capacity.", result_name));
        }
    }

    return capacity;
}

tensors::Tensor make_cache_tensor(const tensors::TensorView& current, std::size_t capacity,
                                  std::string_view result_name) {
    std::vector<std::int64_t> cache_dims = current.tensor_info().shape.dims();
    cache_dims[1] = checked_size_to_dim(capacity, fmt::format("{} capacity", result_name));
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

void ensure_cache_tensor_capacity(tensors::Tensor& cache_tensor, const tensors::TensorView& current,
                                  const std::vector<std::size_t>& cached_sequence_lengths,
                                  std::span<const std::size_t> current_sequence_lengths, std::size_t required_capacity,
                                  std::string_view result_name) {
    validate_current_cache_input(current, current_sequence_lengths, result_name);

    const auto capacity = validate_cache_tensor(cache_tensor, current, cached_sequence_lengths, result_name);
    if (required_capacity <= capacity) {
        return;
    }

    const auto new_capacity = grow_cache_capacity(capacity, required_capacity, result_name);
    auto grown_cache_tensor = make_cache_tensor(current, new_capacity, result_name);

    const auto& cache_dims = cache_tensor.tensor_info().shape.dims();
    const auto row_byte_size = cache_row_byte_size(cache_dims, current.tensor_info().dtype, result_name);
    const auto old_batch_byte_size = capacity * row_byte_size;
    const auto new_batch_byte_size = new_capacity * row_byte_size;
    for (std::size_t batch_index = 0; batch_index < cached_sequence_lengths.size(); ++batch_index) {
        const auto filled_byte_size = cached_sequence_lengths[batch_index] * row_byte_size;
        const auto source_offset = batch_index * old_batch_byte_size;
        const auto destination_offset = batch_index * new_batch_byte_size;
        std::memcpy(grown_cache_tensor.mutable_data().data() + destination_offset,
                    cache_tensor.data().data() + source_offset, filled_byte_size);
    }
    cache_tensor = std::move(grown_cache_tensor);
}

void append_sequence_to_cache(tensors::Tensor& cache_tensor, const tensors::TensorView& current,
                              std::vector<std::size_t>& cached_sequence_lengths,
                              std::span<const std::size_t> current_sequence_lengths, std::string_view result_name) {
    const auto& current_dims = current.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(current_dims[0], fmt::format("{} batch size", result_name));
    const auto current_sequence_capacity =
        checked_positive_dim_to_size(current_dims[1], fmt::format("{} sequence", result_name));
    validate_batch_lengths(current_sequence_lengths, batch_size, result_name);

    std::size_t required_sequence_length = 0;
    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        if (current_sequence_lengths[batch_index] > current_sequence_capacity) {
            throw std::invalid_argument(
                fmt::format("{} sequence length exceeds the current tensor capacity.", result_name));
        }

        const auto total_sequence_length = cached_sequence_lengths[batch_index] + current_sequence_lengths[batch_index];
        if (total_sequence_length < cached_sequence_lengths[batch_index]) {
            throw std::overflow_error(fmt::format("{} sequence length overflowed.", result_name));
        }
        required_sequence_length = std::max(required_sequence_length, total_sequence_length);
    }

    ensure_cache_tensor_capacity(cache_tensor, current, cached_sequence_lengths, current_sequence_lengths,
                                 required_sequence_length, result_name);

    const auto capacity = checked_positive_dim_to_size(cache_tensor.tensor_info().shape.dims()[1],
                                                       fmt::format("{} capacity", result_name));
    const auto row_byte_size = cache_row_byte_size(current_dims, current.tensor_info().dtype, result_name);
    const auto source_batch_byte_size = current_sequence_capacity * row_byte_size;
    const auto destination_batch_byte_size = capacity * row_byte_size;
    for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        const auto sequence_length = current_sequence_lengths[batch_index];
        if (sequence_length == 0) {
            continue;
        }

        const auto source_offset = batch_index * source_batch_byte_size;
        const auto destination_offset =
            batch_index * destination_batch_byte_size + cached_sequence_lengths[batch_index] * row_byte_size;
        std::memcpy(cache_tensor.mutable_data().data() + destination_offset, current.data().data() + source_offset,
                    sequence_length * row_byte_size);
        cached_sequence_lengths[batch_index] += sequence_length;
    }
}

void validate_batch_index(const QwenAttentionCache& cache, std::size_t batch_index) {
    if (batch_index >= cache.sequence_lengths.size()) {
        throw std::out_of_range("Qwen attention cache batch index is out of bounds.");
    }
}

} // namespace

QwenAttentionCache make_qwen_attention_cache(std::string_view key_name, std::string_view value_name,
                                             tensors::DType dtype, std::size_t batch_size,
                                             std::size_t max_sequence_length, std::size_t num_key_value_heads,
                                             std::size_t head_dim) {
    if (batch_size == 0 || max_sequence_length == 0 || num_key_value_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("Qwen attention cache requires non-zero dimensions.");
    }

    const auto shape = tensors::Shape({checked_size_to_dim(batch_size, "qwen_attention_cache batch size"),
                                       checked_size_to_dim(max_sequence_length, "qwen_attention_cache sequence"),
                                       checked_size_to_dim(num_key_value_heads, "qwen_attention_cache kv heads"),
                                       checked_size_to_dim(head_dim, "qwen_attention_cache head dim")});
    return QwenAttentionCache{
        .key = tensors::Tensor::zeros(tensors::make_result_tensor_info(key_name, dtype, shape)),
        .value = tensors::Tensor::zeros(tensors::make_result_tensor_info(value_name, dtype, shape)),
        .sequence_lengths = std::vector<std::size_t>(batch_size, 0),
    };
}

void append_to_qwen_attention_cache(QwenAttentionCache& cache, const tensors::TensorView& key,
                                    const tensors::TensorView& value, std::span<const std::size_t> sequence_lengths) {
    validate_current_cache_input(key, sequence_lengths, "qwen_attention_cached_key");
    validate_current_cache_input(value, sequence_lengths, "qwen_attention_cached_value");

    if (key.tensor_info().shape != value.tensor_info().shape) {
        throw std::invalid_argument("qwen_attention cache requires matching key/value append tensor shapes.");
    }

    if (cache.sequence_lengths.empty()) {
        throw std::invalid_argument("qwen_attention cache requires at least one batch row.");
    }

    auto updated_sequence_lengths = cache.sequence_lengths;
    append_sequence_to_cache(cache.key, key, updated_sequence_lengths, sequence_lengths, "qwen_attention_cached_key");
    auto value_sequence_lengths = cache.sequence_lengths;
    append_sequence_to_cache(cache.value, value, value_sequence_lengths, sequence_lengths,
                             "qwen_attention_cached_value");
    cache.sequence_lengths = std::move(updated_sequence_lengths);
}

tensors::TensorView qwen_attention_cache_key_view(const QwenAttentionCache& cache, std::size_t batch_index) {
    validate_batch_index(cache, batch_index);
    const auto batch_view = ops::squeeze(ops::narrow(cache.key.view(), 0, batch_index, 1), 0);
    return ops::narrow(batch_view, 0, 0, cache.sequence_lengths[batch_index]);
}

tensors::TensorView qwen_attention_cache_value_view(const QwenAttentionCache& cache, std::size_t batch_index) {
    validate_batch_index(cache, batch_index);
    const auto batch_view = ops::squeeze(ops::narrow(cache.value.view(), 0, batch_index, 1), 0);
    return ops::narrow(batch_view, 0, 0, cache.sequence_lengths[batch_index]);
}

} // namespace cppinf::nn
