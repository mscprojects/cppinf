#include "tokenizers/hf/hf_tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "io/file.h"
#include "loaders/hf/hf_model_files.h"

namespace cppinf::tokenizers::hf {
namespace {

using json = nlohmann::json;

constexpr std::string_view k_qwen_split_pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

json parse_json_file(const std::filesystem::path& path) {
    try {
        return json::parse(io::read_text_file(path, "tokenizer file"));
    } catch (const json::parse_error& error) {
        throw std::invalid_argument(
            fmt::format("Failed to parse tokenizer JSON '{}': {}", path.string(), error.what()));
    }
}

std::int64_t read_i64(const json& value, std::string_view field_name) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(fmt::format("{} must be an integer.", field_name));
    }
    return value.get<std::int64_t>();
}

std::string read_string(const json& value, std::string_view field_name) {
    if (!value.is_string()) {
        throw std::invalid_argument(fmt::format("{} must be a string.", field_name));
    }
    return value.get<std::string>();
}

std::optional<std::string> token_content_from_config(const json& config, std::string_view field_name) {
    const auto iterator = config.find(std::string(field_name));
    if (iterator == config.end()) {
        return std::nullopt;
    }

    if (iterator->is_null()) {
        return std::nullopt;
    }

    if (iterator->is_string()) {
        return iterator->get<std::string>();
    }

    if (iterator->is_object()) {
        const auto content_iterator = iterator->find("content");
        if (content_iterator != iterator->end() && content_iterator->is_string()) {
            return content_iterator->get<std::string>();
        }
    }

    throw std::invalid_argument(fmt::format("{} must be null, a string, or an object with content.", field_name));
}

char32_t codepoint_from_utf8(std::string_view text, std::size_t offset, std::size_t* length_out) {
    if (offset >= text.size()) {
        throw std::invalid_argument("UTF-8 decode offset is out of range.");
    }

    const auto lead = static_cast<unsigned char>(text[offset]);
    if ((lead & 0x80U) == 0U) {
        *length_out = 1;
        return lead;
    }

    if ((lead & 0xe0U) == 0xc0U) {
        *length_out = 2;
        if (offset + *length_out > text.size()) {
            throw std::invalid_argument("Invalid UTF-8 sequence.");
        }
        return (static_cast<char32_t>(lead & 0x1fU) << 6) |
               static_cast<char32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3fU);
    }

    if ((lead & 0xf0U) == 0xe0U) {
        *length_out = 3;
        if (offset + *length_out > text.size()) {
            throw std::invalid_argument("Invalid UTF-8 sequence.");
        }
        return (static_cast<char32_t>(lead & 0x0fU) << 12) |
               (static_cast<char32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3fU) << 6) |
               static_cast<char32_t>(static_cast<unsigned char>(text[offset + 2]) & 0x3fU);
    }

    if ((lead & 0xf8U) == 0xf0U) {
        *length_out = 4;
        if (offset + *length_out > text.size()) {
            throw std::invalid_argument("Invalid UTF-8 sequence.");
        }
        return (static_cast<char32_t>(lead & 0x07U) << 18) |
               (static_cast<char32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3fU) << 12) |
               (static_cast<char32_t>(static_cast<unsigned char>(text[offset + 2]) & 0x3fU) << 6) |
               static_cast<char32_t>(static_cast<unsigned char>(text[offset + 3]) & 0x3fU);
    }

    throw std::invalid_argument("Unsupported UTF-8 leading byte.");
}

std::string utf8_from_codepoint(char32_t codepoint) {
    std::string utf8;
    if (codepoint <= 0x7f) {
        utf8.push_back(static_cast<char>(codepoint));
        return utf8;
    }

    if (codepoint <= 0x7ff) {
        utf8.push_back(static_cast<char>(0xc0U | ((codepoint >> 6) & 0x1fU)));
        utf8.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        return utf8;
    }

    if (codepoint <= 0xffff) {
        utf8.push_back(static_cast<char>(0xe0U | ((codepoint >> 12) & 0x0fU)));
        utf8.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        utf8.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        return utf8;
    }

    if (codepoint <= 0x10ffff) {
        utf8.push_back(static_cast<char>(0xf0U | ((codepoint >> 18) & 0x07U)));
        utf8.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3fU)));
        utf8.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        utf8.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        return utf8;
    }

    throw std::invalid_argument("Unicode codepoint is out of range.");
}

std::vector<std::string> split_utf8_codepoints(std::string_view text) {
    std::vector<std::string> codepoints;
    for (std::size_t offset = 0; offset < text.size();) {
        std::size_t length = 0;
        codepoint_from_utf8(text, offset, &length);
        codepoints.emplace_back(text.substr(offset, length));
        offset += length;
    }
    return codepoints;
}

std::vector<std::string> build_byte_encoder() {
    std::vector<int> source_bytes;
    source_bytes.reserve(256);
    for (int value = 33; value <= 126; ++value) {
        source_bytes.push_back(value);
    }
    for (int value = 161; value <= 172; ++value) {
        source_bytes.push_back(value);
    }
    for (int value = 174; value <= 255; ++value) {
        source_bytes.push_back(value);
    }

    auto target_codepoints = source_bytes;
    int extra_index = 0;
    for (int byte_value = 0; byte_value <= 255; ++byte_value) {
        if (std::find(source_bytes.begin(), source_bytes.end(), byte_value) != source_bytes.end()) {
            continue;
        }
        source_bytes.push_back(byte_value);
        target_codepoints.push_back(256 + extra_index);
        ++extra_index;
    }

    std::vector<std::string> byte_encoder(256);
    for (std::size_t index = 0; index < source_bytes.size(); ++index) {
        byte_encoder[static_cast<std::size_t>(source_bytes[index])] =
            utf8_from_codepoint(static_cast<char32_t>(target_codepoints[index]));
    }
    return byte_encoder;
}

std::unordered_map<std::string, unsigned char> build_byte_decoder(const std::vector<std::string>& byte_encoder) {
    std::unordered_map<std::string, unsigned char> byte_decoder;
    for (std::size_t byte_value = 0; byte_value < byte_encoder.size(); ++byte_value) {
        byte_decoder.emplace(byte_encoder[byte_value], static_cast<unsigned char>(byte_value));
    }
    return byte_decoder;
}

bool is_ascii_letter(unsigned char byte_value) {
    return std::isalpha(byte_value) != 0;
}

bool is_letter_like(unsigned char byte_value) {
    return is_ascii_letter(byte_value) || byte_value >= 0x80U;
}

bool is_ascii_digit(unsigned char byte_value) {
    return std::isdigit(byte_value) != 0;
}

bool is_newline(unsigned char byte_value) {
    return byte_value == '\r' || byte_value == '\n';
}

bool is_space_not_newline(unsigned char byte_value) {
    return std::isspace(byte_value) != 0 && !is_newline(byte_value);
}

bool is_word_like(unsigned char byte_value) {
    return is_letter_like(byte_value) || is_ascii_digit(byte_value);
}

std::size_t match_contraction_length(std::string_view text, std::size_t offset) {
    static const std::array<std::string_view, 7> contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (const auto contraction : contractions) {
        if (offset + contraction.size() > text.size()) {
            continue;
        }

        bool matches = true;
        for (std::size_t index = 0; index < contraction.size(); ++index) {
            if (std::tolower(static_cast<unsigned char>(text[offset + index])) !=
                std::tolower(static_cast<unsigned char>(contraction[index]))) {
                matches = false;
                break;
            }
        }

        if (matches) {
            return contraction.size();
        }
    }

    return 0;
}

std::string merge_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

void validate_supported_tokenizer_json(const json& tokenizer_json) {
    if (!tokenizer_json.is_object()) {
        throw std::invalid_argument("tokenizer.json must be a JSON object.");
    }

    const auto& normalizer = tokenizer_json.at("normalizer");
    if (!normalizer.is_object() || read_string(normalizer.at("type"), "normalizer.type") != "NFC") {
        throw std::invalid_argument("Only HF tokenizer.json files with NFC normalizer are supported.");
    }

    const auto& pre_tokenizer = tokenizer_json.at("pre_tokenizer");
    if (!pre_tokenizer.is_object() || read_string(pre_tokenizer.at("type"), "pre_tokenizer.type") != "Sequence") {
        throw std::invalid_argument("Only HF tokenizer.json files with sequence pre-tokenizer are supported.");
    }
    const auto& pretokenizers = pre_tokenizer.at("pretokenizers");
    if (!pretokenizers.is_array() || pretokenizers.size() != 2) {
        throw std::invalid_argument(
            "Only HF tokenizer.json files with split + bytelevel pre-tokenizers are supported.");
    }

    if (read_string(pretokenizers[0].at("type"), "pre_tokenizer[0].type") != "Split" ||
        read_string(pretokenizers[1].at("type"), "pre_tokenizer[1].type") != "ByteLevel") {
        throw std::invalid_argument(
            "Only HF tokenizer.json files with split + bytelevel pre-tokenizers are supported.");
    }

    if (read_string(pretokenizers[0].at("pattern").at("Regex"), "pre_tokenizer[0].pattern.Regex") !=
        k_qwen_split_pattern) {
        throw std::invalid_argument("Only the Qwen/GPT byte-level split pattern is supported.");
    }

    const auto& decoder = tokenizer_json.at("decoder");
    if (!decoder.is_object() || read_string(decoder.at("type"), "decoder.type") != "ByteLevel") {
        throw std::invalid_argument("Only HF tokenizer.json files with bytelevel decoder are supported.");
    }

    const auto& post_processor = tokenizer_json.at("post_processor");
    if (!post_processor.is_object() || read_string(post_processor.at("type"), "post_processor.type") != "ByteLevel") {
        throw std::invalid_argument("Only HF tokenizer.json files with bytelevel post processor are supported.");
    }

    const auto& model = tokenizer_json.at("model");
    if (!model.is_object() || read_string(model.at("type"), "model.type") != "BPE") {
        throw std::invalid_argument("Only HF tokenizer.json files with BPE model are supported.");
    }
}

std::unordered_map<std::string, std::int64_t> parse_vocab(const json& tokenizer_json) {
    const auto& vocab_json = tokenizer_json.at("model").at("vocab");
    if (!vocab_json.is_object()) {
        throw std::invalid_argument("tokenizer.json model.vocab must be an object.");
    }

    std::unordered_map<std::string, std::int64_t> token_to_id;
    for (auto iterator = vocab_json.begin(); iterator != vocab_json.end(); ++iterator) {
        token_to_id.emplace(iterator.key(), read_i64(iterator.value(), "model.vocab id"));
    }
    return token_to_id;
}

std::unordered_map<std::string, std::size_t> parse_merges(const json& tokenizer_json) {
    const auto& merges_json = tokenizer_json.at("model").at("merges");
    if (!merges_json.is_array()) {
        throw std::invalid_argument("tokenizer.json model.merges must be an array.");
    }

    std::unordered_map<std::string, std::size_t> merge_ranks;
    for (std::size_t index = 0; index < merges_json.size(); ++index) {
        std::string left;
        std::string right;

        if (merges_json[index].is_string()) {
            const auto merge_text = read_string(merges_json[index], "model.merges entry");
            const auto separator = merge_text.find(' ');
            if (separator == std::string::npos) {
                throw std::invalid_argument("BPE merge string entries must contain a space separator.");
            }
            left = merge_text.substr(0, separator);
            right = merge_text.substr(separator + 1);
        } else if (merges_json[index].is_array() && merges_json[index].size() == 2) {
            left = read_string(merges_json[index][0], "model.merges[0]");
            right = read_string(merges_json[index][1], "model.merges[1]");
        } else {
            throw std::invalid_argument("BPE merge entries must be strings or 2-element string arrays.");
        }

        merge_ranks.emplace(merge_key(left, right), index);
    }
    return merge_ranks;
}

std::vector<HfTokenizer::AddedToken> parse_added_tokens(const json& tokenizer_json, const json& tokenizer_config) {
    std::unordered_map<std::int64_t, HfTokenizer::AddedToken> added_tokens_by_id;

    const auto& added_tokens_json = tokenizer_json.at("added_tokens");
    if (!added_tokens_json.is_array()) {
        throw std::invalid_argument("tokenizer.json added_tokens must be an array.");
    }
    for (const auto& added_token_json : added_tokens_json) {
        const auto id = read_i64(added_token_json.at("id"), "added_tokens.id");
        added_tokens_by_id.emplace(id,
                                   HfTokenizer::AddedToken{
                                       .id = id,
                                       .content = read_string(added_token_json.at("content"), "added_tokens.content"),
                                       .special = added_token_json.value("special", false),
                                   });
    }

    const auto config_iterator = tokenizer_config.find("added_tokens_decoder");
    if (config_iterator != tokenizer_config.end()) {
        if (!config_iterator->is_object()) {
            throw std::invalid_argument("tokenizer_config.json added_tokens_decoder must be an object.");
        }
        for (auto iterator = config_iterator->begin(); iterator != config_iterator->end(); ++iterator) {
            const auto id = std::stoll(iterator.key());
            if (added_tokens_by_id.contains(id)) {
                continue;
            }
            added_tokens_by_id.emplace(
                id, HfTokenizer::AddedToken{
                        .id = id,
                        .content = read_string(iterator.value().at("content"), "added_tokens_decoder.content"),
                        .special = iterator.value().value("special", false),
                    });
        }
    }

    std::vector<HfTokenizer::AddedToken> added_tokens;
    added_tokens.reserve(added_tokens_by_id.size());
    for (const auto& [_, added_token] : added_tokens_by_id) {
        added_tokens.push_back(added_token);
    }
    std::sort(added_tokens.begin(), added_tokens.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
    return added_tokens;
}

std::optional<std::int64_t> token_id_for_content(const std::unordered_map<std::string, std::int64_t>& token_to_id,
                                                 std::string_view content) {
    const auto iterator = token_to_id.find(std::string(content));
    if (iterator == token_to_id.end()) {
        return std::nullopt;
    }
    return iterator->second;
}

} // namespace

HfTokenizer::HfTokenizer(std::unordered_map<std::string, std::int64_t> token_to_id,
                         std::unordered_map<std::int64_t, std::string> id_to_token,
                         std::vector<AddedToken> added_tokens, std::unordered_map<std::string, std::size_t> merge_ranks,
                         std::vector<std::string> byte_encoder,
                         std::unordered_map<std::string, unsigned char> byte_decoder,
                         std::optional<std::int64_t> eos_token_id, std::optional<std::int64_t> pad_token_id)
    : token_to_id_(std::move(token_to_id)), id_to_token_(std::move(id_to_token)),
      added_tokens_(std::move(added_tokens)), merge_ranks_(std::move(merge_ranks)),
      byte_encoder_(std::move(byte_encoder)), byte_decoder_(std::move(byte_decoder)), eos_token_id_(eos_token_id),
      pad_token_id_(pad_token_id) {
    added_tokens_by_length_.reserve(added_tokens_.size());
    for (const auto& added_token : added_tokens_) {
        added_tokens_by_length_.emplace_back(added_token.content, added_token.id);
        added_token_ids_.insert(added_token.id);
    }
    std::sort(added_tokens_by_length_.begin(), added_tokens_by_length_.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first.size() == rhs.first.size()) {
            return lhs.first < rhs.first;
        }
        return lhs.first.size() > rhs.first.size();
    });
}

HfTokenizer HfTokenizer::from_dir(const std::filesystem::path& model_dir) {
    const auto model_files = loaders::hf::HfModelFiles::from_dir(model_dir);
    return from_files(model_files.tokenizer_json_path, model_files.tokenizer_config_path);
}

HfTokenizer HfTokenizer::from_files(const std::filesystem::path& tokenizer_json_path,
                                    const std::filesystem::path& tokenizer_config_path) {
    const auto tokenizer_json = parse_json_file(tokenizer_json_path);
    const auto tokenizer_config = parse_json_file(tokenizer_config_path);
    validate_supported_tokenizer_json(tokenizer_json);

    auto token_to_id = parse_vocab(tokenizer_json);
    auto id_to_token = std::unordered_map<std::int64_t, std::string>{};
    for (const auto& [token, id] : token_to_id) {
        id_to_token.emplace(id, token);
    }

    const auto added_tokens = parse_added_tokens(tokenizer_json, tokenizer_config);
    for (const auto& added_token : added_tokens) {
        token_to_id.emplace(added_token.content, added_token.id);
        id_to_token.emplace(added_token.id, added_token.content);
    }

    const auto eos_token_content = token_content_from_config(tokenizer_config, "eos_token");
    const auto pad_token_content = token_content_from_config(tokenizer_config, "pad_token");
    const auto eos_token_id = eos_token_content ? token_id_for_content(token_to_id, *eos_token_content) : std::nullopt;
    const auto pad_token_id = pad_token_content ? token_id_for_content(token_to_id, *pad_token_content) : std::nullopt;
    const auto byte_encoder = build_byte_encoder();
    const auto byte_decoder = build_byte_decoder(byte_encoder);

    return HfTokenizer(std::move(token_to_id), std::move(id_to_token), added_tokens, parse_merges(tokenizer_json),
                       byte_encoder, byte_decoder, eos_token_id, pad_token_id);
}

std::vector<std::int64_t> HfTokenizer::encode(std::string_view text) const {
    // The current Qwen tokenizer files require NFC normalization. For plain UTF-8 prompts we keep the input bytes
    // as-is.
    const std::string normalized_text(text);

    std::vector<std::int64_t> token_ids;
    std::size_t segment_start = 0;
    std::size_t offset = 0;
    while (offset < normalized_text.size()) {
        const auto match = added_token_match(normalized_text, offset);
        if (!match) {
            ++offset;
            continue;
        }

        if (segment_start < offset) {
            const auto segment_token_ids = encode_non_added_segment(
                std::string_view(normalized_text).substr(segment_start, offset - segment_start));
            token_ids.insert(token_ids.end(), segment_token_ids.begin(), segment_token_ids.end());
        }

        token_ids.push_back(match->second);
        offset += match->first;
        segment_start = offset;
    }

    if (segment_start < normalized_text.size()) {
        const auto segment_token_ids =
            encode_non_added_segment(std::string_view(normalized_text).substr(segment_start));
        token_ids.insert(token_ids.end(), segment_token_ids.begin(), segment_token_ids.end());
    }

    return token_ids;
}

std::string HfTokenizer::decode(std::span<const std::int64_t> token_ids) const {
    std::string decoded;
    std::string byte_level_buffer;

    const auto flush_byte_level_buffer = [&]() {
        if (byte_level_buffer.empty()) {
            return;
        }
        decoded += decode_byte_level(byte_level_buffer);
        byte_level_buffer.clear();
    };

    for (const auto token_id : token_ids) {
        const auto iterator = id_to_token_.find(token_id);
        if (iterator == id_to_token_.end()) {
            throw std::out_of_range(fmt::format("Tokenizer id {} is not in the vocabulary.", token_id));
        }

        if (added_token_ids_.contains(token_id)) {
            flush_byte_level_buffer();
            decoded += iterator->second;
            continue;
        }

        byte_level_buffer += iterator->second;
    }

    flush_byte_level_buffer();
    return decoded;
}

std::optional<std::int64_t> HfTokenizer::eos_token_id() const {
    return eos_token_id_;
}

std::optional<std::int64_t> HfTokenizer::pad_token_id() const {
    return pad_token_id_;
}

std::vector<std::int64_t> HfTokenizer::encode_non_added_segment(std::string_view text) const {
    std::vector<std::int64_t> token_ids;
    for (const auto& piece : pretokenize(text)) {
        std::string byte_level_piece;
        for (const auto byte_value : piece) {
            byte_level_piece += byte_encoder_[static_cast<unsigned char>(byte_value)];
        }

        for (const auto& bpe_token : bpe_encode(byte_level_piece)) {
            const auto iterator = token_to_id_.find(bpe_token);
            if (iterator == token_to_id_.end()) {
                throw std::invalid_argument(
                    fmt::format("Tokenizer piece '{}' is missing from the vocabulary.", bpe_token));
            }
            token_ids.push_back(iterator->second);
        }
    }
    return token_ids;
}

std::vector<std::string> HfTokenizer::pretokenize(std::string_view text) const {
    std::vector<std::string> pieces;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto current = static_cast<unsigned char>(text[offset]);

        if (const auto contraction_length = match_contraction_length(text, offset); contraction_length != 0) {
            pieces.emplace_back(text.substr(offset, contraction_length));
            offset += contraction_length;
            continue;
        }

        auto newline_scan = offset;
        while (newline_scan < text.size() && is_space_not_newline(static_cast<unsigned char>(text[newline_scan]))) {
            ++newline_scan;
        }

        if (newline_scan < text.size() && is_newline(static_cast<unsigned char>(text[newline_scan]))) {
            auto end = newline_scan;
            while (end < text.size() && is_newline(static_cast<unsigned char>(text[end]))) {
                ++end;
            }
            pieces.emplace_back(text.substr(offset, end - offset));
            offset = end;
            continue;
        }

        if (is_letter_like(current) ||
            ((is_space_not_newline(current) || (!is_word_like(current) && !is_newline(current))) &&
             offset + 1 < text.size() && is_letter_like(static_cast<unsigned char>(text[offset + 1])))) {
            auto end = offset;
            if (!is_letter_like(current)) {
                ++end;
            }
            while (end < text.size() && is_letter_like(static_cast<unsigned char>(text[end]))) {
                ++end;
            }
            pieces.emplace_back(text.substr(offset, end - offset));
            offset = end;
            continue;
        }

        if (is_ascii_digit(current)) {
            pieces.emplace_back(text.substr(offset, 1));
            ++offset;
            continue;
        }

        if ((is_space_not_newline(current) && offset + 1 < text.size() &&
             !is_space_not_newline(static_cast<unsigned char>(text[offset + 1])) &&
             !is_word_like(static_cast<unsigned char>(text[offset + 1])) &&
             !is_newline(static_cast<unsigned char>(text[offset + 1]))) ||
            (!is_space_not_newline(current) && !is_word_like(current) && !is_newline(current))) {
            auto end = offset;
            if (is_space_not_newline(current)) {
                ++end;
            }
            while (end < text.size() && !is_space_not_newline(static_cast<unsigned char>(text[end])) &&
                   !is_word_like(static_cast<unsigned char>(text[end])) &&
                   !is_newline(static_cast<unsigned char>(text[end]))) {
                ++end;
            }
            while (end < text.size() && is_newline(static_cast<unsigned char>(text[end]))) {
                ++end;
            }
            pieces.emplace_back(text.substr(offset, end - offset));
            offset = end;
            continue;
        }

        if (is_space_not_newline(current)) {
            auto end = offset;
            while (end < text.size() && is_space_not_newline(static_cast<unsigned char>(text[end]))) {
                ++end;
            }

            if (end < text.size() && !is_newline(static_cast<unsigned char>(text[end])) &&
                !is_space_not_newline(static_cast<unsigned char>(text[end])) && end - offset > 1) {
                pieces.emplace_back(text.substr(offset, end - offset - 1));
                offset = end - 1;
                continue;
            }
            pieces.emplace_back(text.substr(offset, end - offset));
            offset = end;
            continue;
        }

        std::size_t codepoint_length = 0;
        codepoint_from_utf8(text, offset, &codepoint_length);
        pieces.emplace_back(text.substr(offset, codepoint_length));
        offset += codepoint_length;
    }

    return pieces;
}

std::vector<std::string> HfTokenizer::bpe_encode(std::string_view piece) const {
    if (const auto iterator = bpe_cache_.find(std::string(piece)); iterator != bpe_cache_.end()) {
        return iterator->second;
    }

    auto symbols = split_utf8_codepoints(piece);
    if (symbols.empty()) {
        return {};
    }

    while (symbols.size() > 1) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::size_t best_index = symbols.size();
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto iterator = merge_ranks_.find(merge_key(symbols[index], symbols[index + 1]));
            if (iterator == merge_ranks_.end()) {
                continue;
            }

            if (iterator->second < best_rank) {
                best_rank = iterator->second;
                best_index = index;
            }
        }

        if (best_index == symbols.size()) {
            break;
        }

        std::vector<std::string> merged_symbols;
        merged_symbols.reserve(symbols.size());
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() && index == best_index) {
                merged_symbols.push_back(symbols[index] + symbols[index + 1]);
                index += 2;
                continue;
            }
            merged_symbols.push_back(symbols[index]);
            ++index;
        }
        symbols = std::move(merged_symbols);
    }

    bpe_cache_.emplace(std::string(piece), symbols);
    return symbols;
}

std::optional<std::pair<std::size_t, std::int64_t>> HfTokenizer::added_token_match(std::string_view text,
                                                                                   std::size_t offset) const {
    for (const auto& [content, token_id] : added_tokens_by_length_) {
        if (offset + content.size() > text.size()) {
            continue;
        }

        if (text.substr(offset, content.size()) == content) {
            return std::pair{content.size(), token_id};
        }
    }

    return std::nullopt;
}

std::string HfTokenizer::decode_byte_level(std::string_view encoded) const {
    std::string decoded;
    for (const auto& codepoint : split_utf8_codepoints(encoded)) {
        const auto iterator = byte_decoder_.find(codepoint);
        if (iterator == byte_decoder_.end()) {
            throw std::invalid_argument(fmt::format("Tokenizer decode byte '{}' is unknown.", codepoint));
        }
        decoded.push_back(static_cast<char>(iterator->second));
    }
    return decoded;
}

} // namespace cppinf::tokenizers::hf
