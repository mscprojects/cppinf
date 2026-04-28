#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "cli/cli_app.h"
#include "test_temp_dir.h"

using cppinf::cli::CliResult;
using cppinf::cli::run;

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

    void write_required_hf_files() {
        write_text_file("config.json", R"({
            "architectures": ["Qwen3ForCausalLM"],
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
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

    const std::filesystem::path& model_dir() const {
        return temp_dir_.path();
    }

  private:
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
        .output = "Usage:\n  cppinf\n  cppinf inspect hf <model-dir> [--all] [--limit <count>]\n",
    };

    EXPECT_EQ(expected, run(args));
}

TEST_F(CliAppTest, GivenInspectHfArguments_WhenRunning_ThenFormattedSummaryIsReturned) {
    write_required_hf_files();
    const std::string model_dir_path = model_dir().string();
    const std::string_view args[] = {"inspect", "hf", model_dir_path};

    const CliResult result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ(std::string::npos, result.output.find("Usage:"));
    EXPECT_NE(std::string::npos, result.output.find("HF model directory:"));
    EXPECT_NE(std::string::npos, result.output.find(model_dir_path));
}

TEST_F(CliAppTest, GivenInspectHfLimit_WhenRunning_ThenTensorPreviewIsLimited) {
    write_required_hf_files();
    const std::string model_dir_path = model_dir().string();
    const std::string_view args[] = {"inspect", "hf", model_dir_path, "--limit", "1"};

    const CliResult result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_NE(std::string::npos, result.output.find("Tensor preview (showing 1 of 2):"));
    EXPECT_NE(std::string::npos, result.output.find("embed"));
    EXPECT_EQ(std::string::npos, result.output.find("token_ids"));
}

TEST_F(CliAppTest, GivenInspectHfAll_WhenRunning_ThenAllTensorsAreShown) {
    write_required_hf_files();
    const std::string model_dir_path = model_dir().string();
    const std::string_view args[] = {"inspect", "hf", model_dir_path, "--all"};

    const CliResult result = run(args);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_NE(std::string::npos, result.output.find("Tensors:\n"));
    EXPECT_NE(std::string::npos, result.output.find("embed"));
    EXPECT_NE(std::string::npos, result.output.find("token_ids"));
}
