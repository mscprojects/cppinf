#include <string>

#include <gtest/gtest.h>

#include "loaders/hf/hf_config.h"

using cppinf::loaders::hf::HfConfig;
using cppinf::tensors::DType;

class HfConfigTest : public ::testing::Test {};

TEST_F(HfConfigTest, GivenValidJson_WhenParsing_ThenExpectedFieldsAreLoaded) {
    const std::string json_text = R"({
        "architectures": ["Qwen3ForCausalLM"],
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "head_dim": 128,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "max_position_embeddings": 32768,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "vocab_size": 151936
    })";

    const auto config = HfConfig::from_json_text(json_text);
    const HfConfig expected{
        .architectures = {"Qwen3ForCausalLM"},
        .model_type = "qwen3",
        .head_dim = 128,
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .max_position_embeddings = 32768,
        .num_attention_heads = 16,
        .num_hidden_layers = 28,
        .num_key_value_heads = 8,
        .vocab_size = 151936,
        .bos_token_id = 151643,
        .eos_token_id = 151643,
        .rms_norm_eps = 1e-6f,
        .rope_theta = 1000000.0f,
        .tie_word_embeddings = true,
        .tensor_dtype = DType::BF16,
    };

    EXPECT_EQ(expected, config);
}

TEST_F(HfConfigTest, GivenMissingField_WhenParsing_ThenItThrows) {
    const std::string json_text = R"({
        "architectures": ["Qwen3ForCausalLM"],
        "bos_token_id": 151643
    })";

    EXPECT_THROW(static_cast<void>(HfConfig::from_json_text(json_text)), std::invalid_argument);
}

TEST_F(HfConfigTest, GivenUnsupportedTorchDtype_WhenParsing_ThenItThrows) {
    const std::string json_text = R"({
        "architectures": ["Qwen3ForCausalLM"],
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "head_dim": 128,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "max_position_embeddings": 32768,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "tie_word_embeddings": true,
        "torch_dtype": "float64",
        "vocab_size": 151936
    })";

    EXPECT_THROW(static_cast<void>(HfConfig::from_json_text(json_text)), std::invalid_argument);
}
