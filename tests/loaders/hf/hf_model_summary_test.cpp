#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "loaders/hf/hf_model_summary.h"
#include "test_file_utils.h"
#include "test_temp_dir.h"

namespace cppinf::tests {

using loaders::hf::format_model_summary;
using loaders::hf::HfConfig;
using loaders::hf::HfModelSummary;
using loaders::hf::load_model_summary;
using tensors::DType;
using tensors::Shape;
using tensors::TensorInfo;

class HfModelSummaryTest : public ::testing::Test {
  protected:
    void write_text_file(std::string_view file_name, std::string_view text) {
        file_test_utils::write_text_file(temp_dir_.path() / file_name, text);
    }

    void write_binary_file(std::string_view file_name, std::span<const std::byte> bytes) {
        file_test_utils::write_binary_file(temp_dir_.path() / file_name, bytes);
    }

    void write_required_hf_files() {
        write_text_file("config.json", R"({
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
        })");
        write_text_file("tokenizer.json", R"({"version":"1.0"})");
        write_text_file("tokenizer_config.json", R"({"tokenizer_class":"Qwen2Tokenizer"})");

        const std::array<std::byte, 20> tensor_data{
            std::byte{0x10}, std::byte{0x11}, std::byte{0x12}, std::byte{0x13}, std::byte{0x14},
            std::byte{0x15}, std::byte{0x16}, std::byte{0x17}, std::byte{0x18}, std::byte{0x19},
            std::byte{0x1a}, std::byte{0x1b}, std::byte{0x1c}, std::byte{0x1d}, std::byte{0x1e},
            std::byte{0x1f}, std::byte{0x20}, std::byte{0x21}, std::byte{0x22}, std::byte{0x23},
        };
        const std::string header =
            R"({"__metadata__":{"format":"pt"},"embed":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"tail":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}})";
        write_binary_file("model.safetensors", file_test_utils::make_safetensors_file_bytes(header, tensor_data));
    }

    const std::filesystem::path& model_dir() const {
        return temp_dir_.path();
    }

  private:
    TestTempDir temp_dir_{"cppinf-hf-model-summary-test"};
};

TEST_F(HfModelSummaryTest, GivenValidDirectory_WhenLoadingSummary_ThenExpectedSummaryIsReturned) {
    write_required_hf_files();

    const auto summary = load_model_summary(model_dir());
    const HfModelSummary expected{
        .model_dir = model_dir(),
        .config =
            HfConfig{
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
            },
        .metadata_count = 1,
        .tensor_count = 2,
        .tensor_preview =
            {
                TensorInfo{
                    .name = "embed",
                    .dtype = DType::BF16,
                    .shape = Shape({2, 4}),
                    .byte_offset = 0,
                },
                TensorInfo{
                    .name = "tail",
                    .dtype = DType::F32,
                    .shape = Shape({1}),
                    .byte_offset = 16,
                },
            },
    };

    EXPECT_EQ(expected, summary);
}

TEST_F(HfModelSummaryTest, GivenTensorLimit_WhenLoadingSummary_ThenTensorPreviewIsTrimmed) {
    write_required_hf_files();

    const auto summary = load_model_summary(model_dir(), 1);
    const std::vector<TensorInfo> expected{
        TensorInfo{
            .name = "embed",
            .dtype = DType::BF16,
            .shape = Shape({2, 4}),
            .byte_offset = 0,
        },
    };

    EXPECT_EQ(expected, summary.tensor_preview);
}

TEST_F(HfModelSummaryTest, GivenSummary_WhenFormatting_ThenReadableTextIsProduced) {
    const HfModelSummary summary{
        .model_dir = "/tmp/model",
        .config =
            HfConfig{
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
            },
        .metadata_count = 1,
        .tensor_count = 2,
        .tensor_preview =
            {
                TensorInfo{
                    .name = "embed",
                    .dtype = DType::BF16,
                    .shape = Shape({2, 4}),
                    .byte_offset = 0,
                },
            },
    };

    EXPECT_EQ(std::string("HF model directory: /tmp/model\n"
                          "Model type: qwen3\n"
                          "Architectures: Qwen3ForCausalLM\n"
                          "Tensor dtype: bf16\n"
                          "Hidden size: 1024\n"
                          "Intermediate size: 3072\n"
                          "Max position embeddings: 32768\n"
                          "Hidden layers: 28\n"
                          "Attention heads: 16\n"
                          "Key/value heads: 8\n"
                          "Vocabulary size: 151936\n"
                          "BOS token id: 151643\n"
                          "EOS token id: 151643\n"
                          "Metadata entries: 1\n"
                          "Tensor count: 2\n"
                          "Tensor preview (showing 1 of 2):\n"
                          "  - embed | dtype=bf16 | shape=[2, 4] | offset=0 | bytes=16\n"),
              format_model_summary(summary));
}

} // namespace cppinf::tests
