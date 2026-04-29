#include <algorithm>
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

#include "models/qwen3/qwen3_model.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"

using cppinf::models::qwen3::Qwen3Model;
using cppinf::tensors::bfloat16_bits_to_float;
using cppinf::tensors::DType;
using cppinf::tensors::Tensor;

class Qwen3RealModelTest : public ::testing::Test {
  protected:
    const char* real_model_dir_env() const {
        return std::getenv("CPPINF_QWEN3_REAL_MODEL_DIR");
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

        std::vector<float> values;
        values.reserve(vocabulary_size);

        switch (tensor.tensor_info().dtype) {
        case DType::BF16: {
            const auto row_offset = (sequence_length - 1) * vocabulary_size * sizeof(std::uint16_t);
            for (std::size_t index = 0; index < vocabulary_size; ++index) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, tensor.bytes().data() + row_offset + index * sizeof(std::uint16_t), sizeof(bits));
                values.push_back(bfloat16_bits_to_float(bits));
            }
            return values;
        }
        case DType::F32: {
            const auto row_offset = (sequence_length - 1) * vocabulary_size * sizeof(float);
            for (std::size_t index = 0; index < vocabulary_size; ++index) {
                float value = 0.0f;
                std::memcpy(&value, tensor.bytes().data() + row_offset + index * sizeof(float), sizeof(value));
                values.push_back(value);
            }
            return values;
        }
        case DType::F16:
        case DType::I32:
        case DType::I64:
        case DType::U8:
            throw std::invalid_argument("read_last_token_values supports only bf16 and f32 tensors.");
        }

        throw std::invalid_argument("read_last_token_values received an unsupported tensor dtype.");
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
            EXPECT_NEAR(expected_value, values[start_index + offset], tolerance)
                << "index=" << (start_index + offset);
            ++offset;
        }
    }
};

TEST_F(Qwen3RealModelTest, GivenRealCheckpoint_WhenRunningForward_ThenLastTokenMatchesOracle) {
    const char* const model_dir_env = real_model_dir_env();
    if (model_dir_env == nullptr || *model_dir_env == '\0') {
        GTEST_SKIP() << "Set CPPINF_QWEN3_REAL_MODEL_DIR to run the real Qwen3 oracle test.";
    }

    const std::filesystem::path model_dir = model_dir_env;
    ASSERT_TRUE(std::filesystem::exists(model_dir)) << model_dir;

    // Golden values generated with tests/models/qwen3/qwen3_real_model_oracle.py.
    // Case: Qwen3-0.6B-Base, token_ids=[151643, 42, 1024, 4096].
    const std::vector<std::int64_t> token_ids{151643, 42, 1024, 4096};
    const auto model = Qwen3Model::from_dir(model_dir);
    const auto logits = model.forward(token_ids);
    const auto last_token_values = read_last_token_values(logits);

    EXPECT_EQ(std::size_t{151936}, model.config().vocab_size);
    EXPECT_EQ(DType::BF16, logits.tensor_info().dtype);
    ASSERT_EQ(std::size_t{151936}, last_token_values.size());

    const auto top_k_values = top_k(last_token_values, 8);
    ASSERT_EQ(std::size_t{8}, top_k_values.size());

    // HF eager CPU BF16 and cppinf are close but not bit-exact today, so the real-model parity check
    // asserts on next-token agreement, stable top-4 membership, and bounded logit drift on sampled slices.
    EXPECT_EQ(std::size_t{284}, top_k_values.front().first);
    EXPECT_NEAR(18.25f, top_k_values.front().second, 0.25f);

    std::vector<std::size_t> actual_top_4_ids;
    for (std::size_t index = 0; index < 4; ++index) {
        actual_top_4_ids.push_back(top_k_values[index].first);
    }
    std::sort(actual_top_4_ids.begin(), actual_top_4_ids.end());
    const std::vector<std::size_t> expected_top_4_ids{25, 284, 445, 486};
    EXPECT_EQ(expected_top_4_ids, actual_top_4_ids);

    expect_slice_near(last_token_values, 0,
                      {11.9375f, 14.0f, 10.0f, 12.0f, 10.8125f, 14.4375f, 11.1875f, 15.1875f,
                       14.375f, 14.125f, 9.8125f, 15.25f, 9.3125f, 11.0f, 10.0f, 12.0625f},
                      0.5f);
    expect_slice_near(last_token_values, 1024,
                      {10.1875f, 3.578125f, 7.40625f, 9.1875f, 5.96875f, 8.6875f, 6.375f, 7.75f,
                       8.625f, 5.40625f, 10.1875f, 7.15625f, 10.25f, 3.671875f, 1.828125f, 7.90625f},
                      0.5f);
    expect_slice_near(last_token_values, last_token_values.size() - 16,
                      {1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f,
                       1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f, 1.046875f},
                      0.25f);
}
