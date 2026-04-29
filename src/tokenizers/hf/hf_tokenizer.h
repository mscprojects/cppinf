#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cppinf::tokenizers::hf {

class HfTokenizer {
  public:
    struct AddedToken {
        std::int64_t id{};
        std::string content;
        bool special{};
    };

    static HfTokenizer from_dir(const std::filesystem::path& model_dir);
    static HfTokenizer from_files(const std::filesystem::path& tokenizer_json_path,
                                  const std::filesystem::path& tokenizer_config_path);

    // Encodes text with the loaded HF byte-level BPE tokenizer and returns token ids.
    std::vector<std::int64_t> encode(std::string_view text) const;

    // Decodes token ids back to text with the loaded HF tokenizer vocabulary and added tokens.
    std::string decode(std::span<const std::int64_t> token_ids) const;

    std::optional<std::int64_t> eos_token_id() const;
    std::optional<std::int64_t> pad_token_id() const;

  private:
    HfTokenizer(std::unordered_map<std::string, std::int64_t> token_to_id,
                std::unordered_map<std::int64_t, std::string> id_to_token, std::vector<AddedToken> added_tokens,
                std::unordered_map<std::string, std::size_t> merge_ranks, std::vector<std::string> byte_encoder,
                std::unordered_map<std::string, unsigned char> byte_decoder, std::optional<std::int64_t> eos_token_id,
                std::optional<std::int64_t> pad_token_id);

    std::vector<std::int64_t> encode_non_added_segment(std::string_view text) const;
    std::vector<std::string> pretokenize(std::string_view text) const;
    std::vector<std::string> bpe_encode(std::string_view piece) const;
    std::optional<std::pair<std::size_t, std::int64_t>> added_token_match(std::string_view text,
                                                                           std::size_t offset) const;
    std::string decode_byte_level(std::string_view encoded) const;

    std::unordered_map<std::string, std::int64_t> token_to_id_;
    std::unordered_map<std::int64_t, std::string> id_to_token_;
    std::vector<AddedToken> added_tokens_;
    std::vector<std::pair<std::string, std::int64_t>> added_tokens_by_length_;
    std::unordered_set<std::int64_t> added_token_ids_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::vector<std::string> byte_encoder_;
    std::unordered_map<std::string, unsigned char> byte_decoder_;
    std::optional<std::int64_t> eos_token_id_;
    std::optional<std::int64_t> pad_token_id_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
};

} // namespace cppinf::tokenizers::hf
