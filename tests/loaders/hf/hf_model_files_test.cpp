#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "loaders/hf/hf_config.h"
#include "loaders/hf/hf_model_files.h"
#include "tensors/dtype.h"
#include "test_temp_dir.h"

using cppinf::loaders::hf::HfConfig;
using cppinf::loaders::hf::HfModelFiles;
using cppinf::tensors::DType;

class HfModelFilesTest : public ::testing::Test {
  protected:
    void write_text_file(std::string_view file_name, std::string_view text) {
        std::ofstream output(temp_dir_.path() / file_name);
        output << text;
    }

    void write_binary_file(std::string_view file_name, std::span<const std::byte> bytes) {
        std::ofstream output(temp_dir_.path() / file_name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
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

        const std::array<std::byte, 4> tensor_data{
            std::byte{0x00},
            std::byte{0x01},
            std::byte{0x02},
            std::byte{0x03},
        };
        const std::string header = R"({"weights":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
        write_binary_file("model.safetensors", make_safetensors_file_bytes(header, tensor_data));
    }

    const std::filesystem::path& model_dir() const {
        return temp_dir_.path();
    }

  private:
    void append_u64_le(std::uint64_t value, std::vector<std::byte>& bytes) const {
        for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
            bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 0xffU));
        }
    }

    TestTempDir temp_dir_{"cppinf-hf-model-files-test"};
};

TEST_F(HfModelFilesTest, GivenValidDirectory_WhenResolving_ThenExpectedFilesAreReturned) {
    write_required_hf_files();
    write_text_file("generation_config.json", R"({"max_new_tokens":2048})");
    write_text_file("merges.txt", "foo bar");
    write_text_file("vocab.json", R"({"hello": 0})");

    const HfModelFiles model_files = HfModelFiles::from_dir(model_dir());
    const HfModelFiles expected{
        .model_dir = model_dir(),
        .config_path = model_dir() / "config.json",
        .weights_path = model_dir() / "model.safetensors",
        .tokenizer_json_path = model_dir() / "tokenizer.json",
        .tokenizer_config_path = model_dir() / "tokenizer_config.json",
        .generation_config_path = model_dir() / "generation_config.json",
        .merges_path = model_dir() / "merges.txt",
        .vocab_path = model_dir() / "vocab.json",
    };

    EXPECT_EQ(expected, model_files);
}

TEST_F(HfModelFilesTest, GivenOptionalFilesMissing_WhenResolving_ThenOptionalsAreEmpty) {
    write_required_hf_files();

    const HfModelFiles model_files = HfModelFiles::from_dir(model_dir());

    EXPECT_FALSE(model_files.generation_config_path);
    EXPECT_FALSE(model_files.merges_path);
    EXPECT_FALSE(model_files.vocab_path);
}

TEST_F(HfModelFilesTest, GivenMissingRequiredFile_WhenResolving_ThenItThrows) {
    write_text_file("config.json", "{}");

    EXPECT_THROW(static_cast<void>(HfModelFiles::from_dir(model_dir())), std::invalid_argument);
}

TEST_F(HfModelFilesTest, GivenValidDirectory_WhenLoadingConfigAndWeights_ThenTheyAreAvailable) {
    write_required_hf_files();

    const HfModelFiles model_files = HfModelFiles::from_dir(model_dir());
    const HfConfig config = model_files.load_config();
    const auto weights = model_files.load_weights();
    const HfConfig expected_config{
        .architectures = {"Qwen3ForCausalLM"},
        .model_type = "qwen3",
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .max_position_embeddings = 32768,
        .num_attention_heads = 16,
        .num_hidden_layers = 28,
        .num_key_value_heads = 8,
        .vocab_size = 151936,
        .bos_token_id = 151643,
        .eos_token_id = 151643,
        .tensor_dtype = DType::BF16,
    };

    EXPECT_EQ(expected_config, config);
    EXPECT_TRUE(weights.contains_tensor("weights"));
}

TEST_F(HfModelFilesTest, GivenShardedIndex_WhenResolving_ThenItThrows) {
    write_required_hf_files();
    write_text_file("model.safetensors.index.json", R"({"metadata":{}})");

    EXPECT_THROW(static_cast<void>(HfModelFiles::from_dir(model_dir())), std::invalid_argument);
}
