#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "test_file_utils.h"
#include "test_temp_dir.h"
#include "tokenizers/hf/hf_tokenizer.h"

namespace cppinf::tests {

using tokenizers::hf::HfTokenizer;

class HfTokenizerTest : public ::testing::Test {
  protected:
    void write_text_file(std::string_view file_name, std::string_view contents) {
        file_test_utils::write_text_file(temp_dir_.path() / file_name, contents);
    }

    void write_tiny_tokenizer_files() {
        write_text_file("tokenizer.json",
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
                        "id": 100,
                        "content": "<|im_start|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    },
                    {
                        "id": 101,
                        "content": "<|im_end|>",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true
                    },
                    {
                        "id": 102,
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
                        "Hello": 0,
                        "user": 1,
                        "Ġworld": 2,
                        "Ġ": 3,
                        "Ċ": 4
                    },
                    "merges": [
                        "H e",
                        "He l",
                        "Hel l",
                        "Hell o",
                        "u s",
                        "us e",
                        "use r",
                        "Ġ w",
                        "Ġw o",
                        "Ġwo r",
                        "Ġwor l",
                        "Ġworl d"
                    ],
                    "unk_token": null,
                    "continuing_subword_prefix": "",
                    "end_of_word_suffix": "",
                    "fuse_unk": false
                }
            })");

        write_text_file("tokenizer_config.json",
                        R"({
                "tokenizer_class": "Qwen2Tokenizer",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "added_tokens_decoder": {
                    "100": {"content": "<|im_start|>", "special": true},
                    "101": {"content": "<|im_end|>", "special": true},
                    "102": {"content": "<|endoftext|>", "special": true}
                }
            })");
    }

    std::filesystem::path real_model_dir() const {
        const char* const model_dir_env = std::getenv("CPPINF_QWEN3_REAL_MODEL_DIR");
        if (model_dir_env != nullptr && *model_dir_env != '\0') {
            return model_dir_env;
        }

        throw std::invalid_argument("CPPINF_QWEN3_REAL_MODEL_DIR must be set for real tokenizer tests.");
    }

    const std::filesystem::path& temp_dir() const {
        return temp_dir_.path();
    }

  private:
    TestTempDir temp_dir_{"cppinf-hf-tokenizer-test"};
};

TEST_F(HfTokenizerTest, GivenTinyFixture_WhenEncodingAndDecoding_ThenExpectedIdsAndTextAreReturned) {
    write_tiny_tokenizer_files();

    const auto tokenizer = HfTokenizer::from_files(temp_dir() / "tokenizer.json", temp_dir() / "tokenizer_config.json");

    EXPECT_EQ((std::vector<std::int64_t>{0, 2}), tokenizer.encode("Hello world"));
    EXPECT_EQ((std::vector<std::int64_t>{0, 3, 2, 4}), tokenizer.encode("Hello  world\n"));
    EXPECT_EQ((std::vector<std::int64_t>{100, 1, 4, 0, 101, 4}),
              tokenizer.encode("<|im_start|>user\nHello<|im_end|>\n"));

    EXPECT_EQ(std::string("Hello world"), tokenizer.decode(std::vector<std::int64_t>{0, 2}));
    EXPECT_EQ(std::string("Hello  world\n"), tokenizer.decode(std::vector<std::int64_t>{0, 3, 2, 4}));
    EXPECT_EQ(std::string("<|im_start|>user\nHello<|im_end|>\n"),
              tokenizer.decode(std::vector<std::int64_t>{100, 1, 4, 0, 101, 4}));
    EXPECT_EQ(std::optional<std::int64_t>{102}, tokenizer.eos_token_id());
    EXPECT_EQ(std::optional<std::int64_t>{102}, tokenizer.pad_token_id());
}

TEST_F(HfTokenizerTest, GivenUnknownTokenId_WhenDecoding_ThenItThrows) {
    write_tiny_tokenizer_files();

    const auto tokenizer = HfTokenizer::from_files(temp_dir() / "tokenizer.json", temp_dir() / "tokenizer_config.json");

    EXPECT_THROW(tokenizer.decode(std::vector<std::int64_t>{999}), std::out_of_range);
}

TEST_F(HfTokenizerTest, GivenRealQwenTokenizer_WhenEncodingAndDecoding_ThenExpectedOracleValuesAreReturned) {
    const std::filesystem::path model_dir = real_model_dir();
    ASSERT_TRUE(std::filesystem::exists(model_dir)) << model_dir;

    const auto tokenizer = HfTokenizer::from_dir(model_dir);

    // Golden values generated with tests/tokenizers/hf/hf_tokenizer_oracle.py.
    EXPECT_EQ((std::vector<std::int64_t>{9707, 1879}), tokenizer.encode("Hello world"));
    EXPECT_EQ((std::vector<std::int64_t>{9707, 220, 1879, 198}), tokenizer.encode("Hello  world\n"));
    EXPECT_EQ((std::vector<std::int64_t>{785, 3974, 13876, 38835, 34208, 916, 220, 16, 18, 15678, 12590, 13}),
              tokenizer.encode("The quick brown fox jumps over 13 lazy dogs."));
    EXPECT_EQ((std::vector<std::int64_t>{151644, 872, 198, 9707, 151645, 198}),
              tokenizer.encode("<|im_start|>user\nHello<|im_end|>\n"));

    EXPECT_EQ(std::string("Hello world"), tokenizer.decode(std::vector<std::int64_t>{9707, 1879}));
    EXPECT_EQ(std::string("Hello  world\n"), tokenizer.decode(std::vector<std::int64_t>{9707, 220, 1879, 198}));
    EXPECT_EQ(std::string("The quick brown fox jumps over 13 lazy dogs."),
              tokenizer.decode(
                  std::vector<std::int64_t>{785, 3974, 13876, 38835, 34208, 916, 220, 16, 18, 15678, 12590, 13}));
    EXPECT_EQ(std::string("<|im_start|>user\nHello<|im_end|>\n"),
              tokenizer.decode(std::vector<std::int64_t>{151644, 872, 198, 9707, 151645, 198}));
    EXPECT_EQ(std::optional<std::int64_t>{151645}, tokenizer.eos_token_id());
    EXPECT_EQ(std::optional<std::int64_t>{151643}, tokenizer.pad_token_id());
}

} // namespace cppinf::tests
