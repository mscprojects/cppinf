#include <cstddef>

#include <gtest/gtest.h>

#include "nn/qwen_cache.h"
#include "test_tensor_utils.h"

namespace cppinf::tests {

using nn::append_to_qwen_attention_cache;
using nn::make_qwen_attention_cache;
using nn::qwen_attention_cache_key_view;
using nn::qwen_attention_cache_value_view;
using nn::QwenAttentionCache;
using tensor_test_utils::expect_float_values_near;
using tensor_test_utils::make_f32_tensor;
using tensors::DType;
using tensors::Shape;

class QwenCacheTest : public ::testing::Test {};

TEST_F(QwenCacheTest, GivenEmptyCache_WhenAppendingSequence_ThenStorageIsAllocatedOnDemand) {
    QwenAttentionCache cache;
    const auto key = make_f32_tensor("key", {2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto value = make_f32_tensor("value", {2, 1, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    append_to_qwen_attention_cache(cache, key.view(), value.view(), 4);

    ASSERT_TRUE(cache.key.has_value());
    ASSERT_TRUE(cache.value.has_value());
    EXPECT_EQ(std::size_t{2}, cache.sequence_length);
    EXPECT_EQ(std::size_t{4}, cache.sequence_position_offset);
    EXPECT_EQ(Shape({2, 1, 2}), cache.key->tensor_info().shape);
    EXPECT_EQ(Shape({2, 1, 2}), cache.value->tensor_info().shape);
    expect_float_values_near(qwen_attention_cache_key_view(cache), {1.0f, 2.0f, 3.0f, 4.0f}, 0.0f);
    expect_float_values_near(qwen_attention_cache_value_view(cache), {5.0f, 6.0f, 7.0f, 8.0f}, 0.0f);
}

TEST_F(QwenCacheTest, GivenFullFixedCache_WhenAppendingMoreTokens_ThenStorageGrowsAndPreservesPrefix) {
    auto cache = make_qwen_attention_cache("key_cache", "value_cache", DType::F32, 2, 1, 2);
    const auto first_key = make_f32_tensor("first_key", {2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto first_value = make_f32_tensor("first_value", {2, 1, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    const auto next_key = make_f32_tensor("next_key", {1, 1, 2}, {9.0f, 10.0f});
    const auto next_value = make_f32_tensor("next_value", {1, 1, 2}, {11.0f, 12.0f});

    append_to_qwen_attention_cache(cache, first_key.view(), first_value.view(), 0);
    append_to_qwen_attention_cache(cache, next_key.view(), next_value.view(), 2);

    ASSERT_TRUE(cache.key.has_value());
    ASSERT_TRUE(cache.value.has_value());
    EXPECT_EQ(std::size_t{3}, cache.sequence_length);
    EXPECT_EQ(Shape({4, 1, 2}), cache.key->tensor_info().shape);
    EXPECT_EQ(Shape({4, 1, 2}), cache.value->tensor_info().shape);
    expect_float_values_near(qwen_attention_cache_key_view(cache), {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f}, 0.0f);
    expect_float_values_near(qwen_attention_cache_value_view(cache), {5.0f, 6.0f, 7.0f, 8.0f, 11.0f, 12.0f}, 0.0f);
}

} // namespace cppinf::tests
