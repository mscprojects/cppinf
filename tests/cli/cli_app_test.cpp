#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "cli/cli_app.h"
#include "test_temp_dir.h"

using cppinf::cli::CliResult;
using cppinf::cli::run;
using cppinf::cli::run_with_output_writer;

class CliAppTest : public ::testing::Test {
  protected:
    void write_text_file(std::string_view file_name, std::string_view text) {
        std::ofstream output(temp_dir_.path() / file_name);
        output << text;
    }

    void write_binary_file(std::string_view file_name, std::span<const std::byte> bytes) {
        std::ofstream output(temp_dir_.path() / file_name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    void write_inspect_fixture_files() {
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
            R"({"embed":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"token_ids":{"dtype":"U8","shape":[4],"data_offsets":[16,20]}})";
        write_binary_file("model.safetensors", make_safetensors_file_bytes(header, tensor_data));
    }

    void write_tiny_generation_model_dir() {
        write_generation_config_file();
        write_generation_tokenizer_files();
        write_generation_weights_file();
    }

    const std::filesystem::path& model_dir() const {
        return temp_dir_.path();
    }

  private:
    using ordered_json = nlohmann::ordered_json;

    void write_generation_config_file() {
        const ordered_json config = {
            {"architectures", {"Qwen3ForCausalLM"}},
            {"bos_token_id", 2},
            {"eos_token_id", 2},
            {"head_dim", 2},
            {"hidden_size", 2},
            {"intermediate_size", 2},
            {"max_position_embeddings", 32},
            {"model_type", "qwen3"},
            {"num_attention_heads", 1},
            {"num_hidden_layers", 1},
            {"num_key_value_heads", 1},
            {"rms_norm_eps", 1e-6},
            {"rope_theta", 1000000.0},
            {"tie_word_embeddings", true},
            {"torch_dtype", "float32"},
            {"vocab_size", 3},
        };
        write_text_file("config.json", config.dump(4));
    }

    void write_generation_tokenizer_files() {
        write_text_file(
            "tokenizer.json",
            R"({
                "version": "1.0",
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                        {
                            "type": "Split",
                            "pattern": {
                                "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                            },
                            "behavior": "Isolated",
                            "invert": false
                        },
                        {
                            "type": "ByteLevel",
                            "add_prefix_space": false,
                            "trim_offsets": false,
                            "use_regex": false
                        }
                    ]
                },
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                },
                "post_processor": {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                },
                "added_tokens": [
                    {
                        "id": 2,
                        "content": "<|endoftext|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    }
                ],
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "A": 0,
                        "B": 1
                    },
                    "merges": [],
                    "unk_token": null,
                    "continuing_subword_prefix": "",
                    "end_of_word_suffix": "",
                    "fuse_unk": false
                }
            })");

        write_text_file(
            "tokenizer_config.json",
            R"({
                "tokenizer_class": "Qwen2Tokenizer",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "added_tokens_decoder": {
                    "2": {"content": "<|endoftext|>", "special": true}
                }
            })");
    }

    void write_generation_weights_file() {
        auto header = ordered_json::object();
        std::vector<std::byte> tensor_data;

        append_f32_tensor(header, tensor_data, "model.embed_tokens.weight", {3, 2}, {1.0f, 0.0f, 2.0f, 1.0f, 1.5f, 5.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.input_layernorm.weight", {2}, {1.0f, 1.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.post_attention_layernorm.weight", {2}, {1.0f, 1.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.self_attn.q_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.self_attn.q_norm.weight", {2}, {1.0f, 1.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.self_attn.k_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.self_attn.k_norm.weight", {2}, {1.0f, 1.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.self_attn.v_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.self_attn.o_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.mlp.gate_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.mlp.up_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.layers.0.mlp.down_proj.weight", {2, 2},
                          {0.0f, 0.0f, 0.0f, 0.0f});
        append_f32_tensor(header, tensor_data, "model.norm.weight", {2}, {1.0f, 1.0f});

        write_binary_file("model.safetensors", make_safetensors_file_bytes(header.dump(), tensor_data));
    }

    std::vector<std::byte> make_safetensors_file_bytes(std::string_view header_json,
                                                       std::span<const std::byte> tensor_data) const {
        std::vector<std::byte> bytes;
        append_u64_le(static_cast<std::uint64_t>(header_json.size()), bytes);

        for (const char character : header_json) {
            bytes.push_back(static_cast<std::byte>(character));
        }

        bytes.insert(bytes.end(), tensor_data.begin(), tensor_data.end());
        return bytes;
    }

    std::size_t num_elements(std::span<const std::int64_t> shape) const {
        std::size_t elements = 1;
        for (const auto dim : shape) {
            if (dim < 0) {
                throw std::invalid_argument("Tensor shape dimensions must be non-negative.");
            }
            elements *= static_cast<std::size_t>(dim);
        }
        return elements;
    }

    void append_f32_tensor(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                           std::vector<std::int64_t> shape, std::initializer_list<float> values) {
        if (values.size() != num_elements(shape)) {
            throw std::invalid_argument("Tensor values do not match the requested shape.");
        }

        const auto begin = tensor_data.size();
        for (const float value : values) {
            const auto* raw = reinterpret_cast<const std::byte*>(&value);
            tensor_data.insert(tensor_data.end(), raw, raw + sizeof(float));
        }
        const auto end = tensor_data.size();

        header[std::string(name)] = ordered_json{
            {"dtype", "F32"},
            {"shape", std::move(shape)},
            {"data_offsets", {begin, end}},
        };
    }

    void append_u64_le(std::uint64_t value, std::vector<std::byte>& bytes) const {
        for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
            bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 0xffU));
        }
    }

    TestTempDir temp_dir_{"cppinf-cli-app-test"};
};

TEST_F(CliAppTest, GivenNoArguments_WhenRunning_ThenDefaultOutputIsReturned) {
    const CliResult expected{
        .exit_code = 0,
        .output = "cppinf\n",
    };

    EXPECT_EQ(expected, run({}));
}

TEST_F(CliAppTest, GivenInvalidArguments_WhenRunning_ThenUsageIsReturned) {
    const std::string_view args[] = {"inspect"};
    const CliResult expected{
        .exit_code = 1,
        .output = "Usage:\n"
                  "  cppinf\n"
                  "  cppinf inspect hf <model-dir> [--all] [--limit <count>]\n"
                  "  cppinf run hf <model-dir> --prompt <text> [--max-new-tokens <count>]\n",
    };

    EXPECT_EQ(expected, run(args));
}

TEST_F(CliAppTest, GivenInspectHfArguments_WhenRunning_ThenFormattedSummaryIsReturned) {
    write_inspect_fixture_files();
    const auto model_dir_path = model_dir().string();
    const std::string_view args[] = {"inspect", "hf", model_dir_path};

    const auto result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ(std::string::npos, result.output.find("Usage:"));
    EXPECT_NE(std::string::npos, result.output.find("HF model directory:"));
    EXPECT_NE(std::string::npos, result.output.find(model_dir_path));
}

TEST_F(CliAppTest, GivenInspectHfLimit_WhenRunning_ThenTensorPreviewIsLimited) {
    write_inspect_fixture_files();
    const auto model_dir_path = model_dir().string();
    const std::string_view args[] = {"inspect", "hf", model_dir_path, "--limit", "1"};

    const auto result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_NE(std::string::npos, result.output.find("Tensor preview (showing 1 of 2):"));
    EXPECT_NE(std::string::npos, result.output.find("embed"));
    EXPECT_EQ(std::string::npos, result.output.find("token_ids"));
}

TEST_F(CliAppTest, GivenInspectHfAll_WhenRunning_ThenAllTensorsAreShown) {
    write_inspect_fixture_files();
    const auto model_dir_path = model_dir().string();
    const std::string_view args[] = {"inspect", "hf", model_dir_path, "--all"};

    const auto result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_NE(std::string::npos, result.output.find("Tensors:\n"));
    EXPECT_NE(std::string::npos, result.output.find("embed"));
    EXPECT_NE(std::string::npos, result.output.find("token_ids"));
}

TEST_F(CliAppTest, GivenRunHfArguments_WhenGeneratingGreedyTokens_ThenPromptIsContinued) {
    write_tiny_generation_model_dir();
    const auto model_dir_path = model_dir().string();
    const std::string_view args[] = {"run", "hf", model_dir_path, "--prompt", "A", "--max-new-tokens", "1"};

    const auto result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ("AB\n", result.output);
}

TEST_F(CliAppTest, GivenRunHfArguments_WhenEosIsMostLikelyNextToken_ThenGenerationStopsBeforePrintingIt) {
    write_tiny_generation_model_dir();
    const auto model_dir_path = model_dir().string();
    const std::string_view args[] = {"run", "hf", model_dir_path, "--prompt", "B", "--max-new-tokens", "4"};

    const auto result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ("B\n", result.output);
    EXPECT_EQ(std::string::npos, result.output.find("<|endoftext|>"));
}

TEST_F(CliAppTest, GivenRunHfArgumentsAndWriter_WhenGenerating_ThenPromptAndNewTextAreStreamed) {
    write_tiny_generation_model_dir();
    const auto model_dir_path = model_dir().string();
    const std::string_view args[] = {"run", "hf", model_dir_path, "--prompt", "A", "--max-new-tokens", "1"};

    std::string streamed_output;
    const auto result = run_with_output_writer(args, [&](std::string_view chunk) { streamed_output += chunk; });

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ("\n", result.output);
    EXPECT_EQ("AB", streamed_output);
}
