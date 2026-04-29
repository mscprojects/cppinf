#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "files/safetensors_file.h"
#include "models/qwen3/qwen3_model.h"
#include "tensors/bfloat16.h"
#include "tensors/dtype.h"
#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

using cppinf::files::SafetensorsFile;
using cppinf::models::qwen3::Qwen3Model;
using cppinf::tensors::bfloat16_bits_to_float;
using cppinf::tensors::DType;
using cppinf::tensors::Tensor;
using cppinf::tensors::TensorView;

class Qwen3RealModelTest : public ::testing::Test {
  protected:
    std::filesystem::path oracle_logits_path() const {
        return std::filesystem::path(__FILE__).parent_path() / "qwen3_real_last_token_logits.safetensors";
    }

    const char* real_model_dir_env() const {
        return std::getenv("CPPINF_QWEN3_REAL_MODEL_DIR");
    }

    std::vector<float> read_float_values(const TensorView& tensor_view) const {
        std::vector<float> values;
        values.reserve(tensor_view.tensor_info().shape.num_elements());

        switch (tensor_view.tensor_info().dtype) {
        case DType::BF16: {
            for (std::size_t index = 0; index < tensor_view.tensor_info().shape.num_elements(); ++index) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, tensor_view.data().data() + index * sizeof(std::uint16_t), sizeof(bits));
                values.push_back(bfloat16_bits_to_float(bits));
            }
            return values;
        }
        case DType::F32: {
            for (std::size_t index = 0; index < tensor_view.tensor_info().shape.num_elements(); ++index) {
                float value = 0.0f;
                std::memcpy(&value, tensor_view.data().data() + index * sizeof(float), sizeof(value));
                values.push_back(value);
            }
            return values;
        }
        case DType::F16:
        case DType::I32:
        case DType::I64:
        case DType::U8:
            throw std::invalid_argument("read_float_values supports only bf16 and f32 tensors.");
        }

        throw std::invalid_argument("read_float_values received an unsupported tensor dtype.");
    }

    std::vector<float> read_last_token_values(const Tensor& tensor) const {
        if (tensor.tensor_info().shape.rank() != 2) {
            throw std::invalid_argument("read_last_token_values requires a rank-2 tensor.");
        }

        const auto& dims = tensor.tensor_info().shape.dims();
        const auto sequence_length = static_cast<std::size_t>(dims[0]);
        const auto vocabulary_size = static_cast<std::size_t>(dims[1]);
        if (sequence_length == 0 || vocabulary_size == 0) {
            throw std::invalid_argument("read_last_token_values requires non-empty tensor dimensions.");
        }

        const auto row_byte_size = vocabulary_size * cppinf::tensors::element_size_bytes(tensor.tensor_info().dtype);
        const auto row_offset = (sequence_length - 1) * row_byte_size;
        return read_float_values(TensorView(
            cppinf::tensors::TensorInfo{
                .name = "last_token_logits",
                .dtype = tensor.tensor_info().dtype,
                .shape = cppinf::tensors::Shape({static_cast<std::int64_t>(vocabulary_size)}),
                .byte_offset = 0,
            },
            std::span<const std::byte>(tensor.bytes()).subspan(row_offset, row_byte_size)));
    }

    std::vector<float> load_oracle_last_token_values() const {
        const auto oracle_path = oracle_logits_path();
        if (!std::filesystem::exists(oracle_path)) {
            throw std::invalid_argument("Real-model oracle safetensors file is missing.");
        }

        const auto oracle_file = SafetensorsFile::from_file(oracle_path);
        return read_float_values(oracle_file.tensor_view("last_token_logits"));
    }

    std::size_t overlap_count(const std::vector<std::size_t>& lhs, const std::vector<std::size_t>& rhs) const {
        std::size_t count = 0;
        for (const auto value : lhs) {
            if (std::find(rhs.begin(), rhs.end(), value) != rhs.end()) {
                ++count;
            }
        }
        return count;
    }

    std::vector<std::pair<std::size_t, float>> top_k(const std::vector<float>& values, std::size_t k) const {
        std::vector<std::size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&values](std::size_t lhs, std::size_t rhs) {
                              if (values[lhs] == values[rhs]) {
                                  return lhs < rhs;
                              }
                              return values[lhs] > values[rhs];
                          });

        std::vector<std::pair<std::size_t, float>> result;
        result.reserve(k);
        for (std::size_t position = 0; position < k; ++position) {
            result.emplace_back(indices[position], values[indices[position]]);
        }
        return result;
    }

    void expect_slice_near(const std::vector<float>& values, std::size_t start_index,
                           std::initializer_list<float> expected, float tolerance) const {
        ASSERT_LE(start_index + expected.size(), values.size());

        std::size_t offset = 0;
        for (const auto expected_value : expected) {
            EXPECT_NEAR(expected_value, values[start_index + offset], tolerance) << "index=" << (start_index + offset);
            ++offset;
        }
    }
};

TEST_F(Qwen3RealModelTest, GivenRealCheckpoint_WhenRunningForward_ThenTopFiveLikelyTokensStayCloseToOracle) {
    const char* const model_dir_env = real_model_dir_env();
    if (model_dir_env == nullptr || *model_dir_env == '\0') {
        GTEST_SKIP() << "Set CPPINF_QWEN3_REAL_MODEL_DIR to run the real Qwen3 oracle test.";
    }

    const std::filesystem::path model_dir = model_dir_env;
    ASSERT_TRUE(std::filesystem::exists(model_dir)) << model_dir;

    const std::vector<std::int64_t> token_ids{151643, 42, 1024, 4096};
    const auto model = Qwen3Model::from_dir(model_dir);
    const auto logits = model.forward(token_ids);
    const auto oracle_last_token_values = load_oracle_last_token_values();
    const auto last_token_values = read_last_token_values(logits);

    EXPECT_EQ(std::size_t{151936}, model.config().vocab_size);
    EXPECT_EQ(DType::BF16, logits.tensor_info().dtype);
    ASSERT_EQ(oracle_last_token_values.size(), last_token_values.size());
    ASSERT_EQ(std::size_t{151936}, last_token_values.size());

    const auto oracle_top_k_values = top_k(oracle_last_token_values, 5);
    const auto actual_top_k_values = top_k(last_token_values, 5);
    ASSERT_EQ(std::size_t{5}, oracle_top_k_values.size());
    ASSERT_EQ(std::size_t{5}, actual_top_k_values.size());

    EXPECT_EQ(oracle_top_k_values.front().first, actual_top_k_values.front().first);
    EXPECT_NEAR(oracle_top_k_values.front().second, actual_top_k_values.front().second, 0.25f);

    std::vector<std::size_t> oracle_top_k_ids;
    std::vector<std::size_t> actual_top_k_ids;
    for (std::size_t index = 0; index < 5; ++index) {
        oracle_top_k_ids.push_back(oracle_top_k_values[index].first);
        actual_top_k_ids.push_back(actual_top_k_values[index].first);

        const auto oracle_id = oracle_top_k_values[index].first;
        EXPECT_NEAR(oracle_top_k_values[index].second, last_token_values[oracle_id], 0.5f)
            << "oracle top-5 token id=" << oracle_id;
    }

    EXPECT_GE(overlap_count(oracle_top_k_ids, actual_top_k_ids), std::size_t{4});
}

TEST_F(Qwen3RealModelTest, GivenRealCheckpoint_WhenRunningForward_ThenFullLastTokenLogitsStayCloseToOracle) {
    const char* const model_dir_env = real_model_dir_env();
    if (model_dir_env == nullptr || *model_dir_env == '\0') {
        GTEST_SKIP() << "Set CPPINF_QWEN3_REAL_MODEL_DIR to run the real Qwen3 oracle test.";
    }

    const std::filesystem::path model_dir = model_dir_env;
    ASSERT_TRUE(std::filesystem::exists(model_dir)) << model_dir;

    const std::vector<std::int64_t> token_ids{151643, 42, 1024, 4096};
    const auto model = Qwen3Model::from_dir(model_dir);
    const auto logits = model.forward(token_ids);
    const auto oracle_last_token_values = load_oracle_last_token_values();
    const auto last_token_values = read_last_token_values(logits);

    ASSERT_EQ(oracle_last_token_values.size(), last_token_values.size());

    float max_abs_diff = 0.0f;
    double sum_abs_diff = 0.0;
    for (std::size_t index = 0; index < last_token_values.size(); ++index) {
        const auto abs_diff = std::abs(last_token_values[index] - oracle_last_token_values[index]);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        sum_abs_diff += abs_diff;
    }

    const auto mean_abs_diff = static_cast<float>(sum_abs_diff / static_cast<double>(last_token_values.size()));

    EXPECT_LE(max_abs_diff, 0.5f);
    EXPECT_LE(mean_abs_diff, 0.2f);
}
