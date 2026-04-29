#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "tensors/tensor_utils.h"
#include "tensors/tensor_view.h"

namespace cppinf::tests::safetensors_test_utils {

using ordered_json = nlohmann::ordered_json;

namespace detail {

inline std::string_view to_safetensors_dtype_name(tensors::DType dtype) {
    switch (dtype) {
    case tensors::DType::BF16:
        return "BF16";
    case tensors::DType::F32:
        return "F32";
    }

    throw std::invalid_argument("Unsupported dtype for safetensors test fixture.");
}

} // namespace detail

// Appends a tensor payload and matching header entry to an in-memory safetensors fixture.
inline void append_tensor(ordered_json& header, std::vector<std::byte>& tensor_data,
                          const tensors::TensorView& tensor) {
    const auto begin_offset = tensor_data.size();
    tensor_data.insert(tensor_data.end(), tensor.data().begin(), tensor.data().end());
    const auto end_offset = tensor_data.size();

    header[std::string(tensor.tensor_info().name)] = ordered_json{
        {"dtype", detail::to_safetensors_dtype_name(tensor.tensor_info().dtype)},
        {"shape", tensor.tensor_info().shape.dims()},
        {"data_offsets", {begin_offset, end_offset}},
    };
}

// Appends an f32 tensor fixture from literal float values.
inline void append_f32_tensor(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                              std::initializer_list<std::int64_t> dims, std::initializer_list<float> values) {
    const auto tensor = tensors::make_f32_tensor(name, tensors::Shape(std::vector<std::int64_t>(dims)),
                                                 std::span<const float>(values.begin(), values.size()));
    append_tensor(header, tensor_data, tensor.view());
}

// Appends a bf16 vector fixture from literal float values.
inline void append_bf16_vector(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                               std::initializer_list<float> values) {
    const auto tensor = tensors::make_bf16_tensor(name, tensors::Shape({static_cast<std::int64_t>(values.size())}),
                                                  std::span<const float>(values.begin(), values.size()));
    append_tensor(header, tensor_data, tensor.view());
}

// Appends a bf16 matrix fixture from row-major literal float values.
inline void append_bf16_matrix(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                               std::initializer_list<std::initializer_list<float>> rows) {
    if (rows.size() == 0) {
        throw std::invalid_argument("append_bf16_matrix requires at least one row.");
    }

    const auto column_count = rows.begin()->size();
    if (column_count == 0) {
        throw std::invalid_argument("append_bf16_matrix requires at least one column.");
    }

    std::vector<float> values;
    values.reserve(rows.size() * column_count);
    for (const auto& row : rows) {
        if (row.size() != column_count) {
            throw std::invalid_argument("append_bf16_matrix requires rows with equal width.");
        }

        values.insert(values.end(), row.begin(), row.end());
    }

    const auto tensor = tensors::make_bf16_tensor(
        name, tensors::Shape({static_cast<std::int64_t>(rows.size()), static_cast<std::int64_t>(column_count)}),
        std::span<const float>(values.data(), values.size()));
    append_tensor(header, tensor_data, tensor.view());
}

} // namespace cppinf::tests::safetensors_test_utils
