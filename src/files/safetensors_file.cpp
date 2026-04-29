#include "safetensors_file.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "io/file.h"
#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace cppinf::files {
namespace detail {

using json = nlohmann::json;

struct ParsedSafetensorsHeader {
    std::size_t tensor_data_offset{};
    std::vector<tensors::TensorInfo> tensor_infos;
    std::unordered_map<std::string, std::string> metadata;
};

std::size_t checked_to_size(std::uint64_t value, std::string_view field_name) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(fmt::format("{} does not fit in size_t.", field_name));
    }

    return static_cast<std::size_t>(value);
}

std::uint64_t read_u64_le(std::span<const std::byte> bytes) {
    if (bytes.size() < sizeof(std::uint64_t)) {
        throw std::invalid_argument("Safetensors files must start with an 8-byte header length.");
    }

    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[index])) << (index * 8);
    }

    return value;
}

std::int64_t read_i64(const json& value, std::string_view field_name) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(fmt::format("{} must be an integer.", field_name));
    }

    return value.get<std::int64_t>();
}

tensors::Shape parse_shape(const json& value) {
    if (!value.is_array()) {
        throw std::invalid_argument("shape must be an array.");
    }

    std::vector<std::int64_t> dims;
    dims.reserve(value.size());
    for (const json& dim : value) {
        dims.push_back(read_i64(dim, "shape dimension"));
    }

    return tensors::Shape(std::move(dims));
}

std::pair<std::size_t, std::size_t> parse_data_offsets(const json& value) {
    if (!value.is_array() || value.size() != 2) {
        throw std::invalid_argument("data_offsets must contain exactly two integers.");
    }

    const auto begin = read_i64(value[0], "data_offsets[0]");
    const auto end = read_i64(value[1], "data_offsets[1]");
    if (begin < 0 || end < 0 || end < begin) {
        throw std::invalid_argument("data_offsets must be non-negative and ordered.");
    }

    return {
        static_cast<std::size_t>(begin),
        static_cast<std::size_t>(end),
    };
}

std::unordered_map<std::string, std::string> parse_metadata(const json& value) {
    if (!value.is_object()) {
        throw std::invalid_argument("__metadata__ must be an object.");
    }

    std::unordered_map<std::string, std::string> metadata;
    for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
        if (!iterator.value().is_string()) {
            throw std::invalid_argument("__metadata__ values must be strings.");
        }

        metadata.emplace(iterator.key(), iterator.value().get<std::string>());
    }

    return metadata;
}

ParsedSafetensorsHeader parse_header(std::span<const std::byte> file_bytes) {
    const auto header_size = checked_to_size(read_u64_le(file_bytes), "header size");
    const auto tensor_data_offset = sizeof(std::uint64_t) + header_size;

    if (tensor_data_offset > file_bytes.size()) {
        throw std::invalid_argument("Safetensors header extends beyond the file size.");
    }

    const auto* header_begin = reinterpret_cast<const char*>(file_bytes.data() + sizeof(std::uint64_t));
    const std::string header_text(header_begin, header_size);

    json header;
    try {
        header = json::parse(header_text);
    } catch (const json::parse_error& error) {
        throw std::invalid_argument(fmt::format("Failed to parse safetensors header: {}", error.what()));
    }

    if (!header.is_object()) {
        throw std::invalid_argument("Safetensors header must be a JSON object.");
    }

    ParsedSafetensorsHeader parsed_header{
        .tensor_data_offset = tensor_data_offset,
    };

    for (auto iterator = header.begin(); iterator != header.end(); ++iterator) {
        if (iterator.key() == "__metadata__") {
            parsed_header.metadata = parse_metadata(iterator.value());
            continue;
        }

        if (!iterator.value().is_object()) {
            throw std::invalid_argument("Tensor entries must be JSON objects.");
        }

        const auto& tensor_json = iterator.value();
        const auto dtype = tensors::parse_dtype(tensor_json.at("dtype").get<std::string>());
        const auto shape = parse_shape(tensor_json.at("shape"));
        const auto [begin, end] = parse_data_offsets(tensor_json.at("data_offsets"));

        tensors::TensorInfo tensor_info{
            .name = iterator.key(),
            .dtype = dtype,
            .shape = shape,
            .byte_offset = begin,
        };

        if (tensor_info.byte_size() != end - begin) {
            throw std::invalid_argument("Tensor byte range does not match dtype and shape.");
        }
        if (begin > file_bytes.size() - tensor_data_offset ||
            tensor_info.byte_size() > file_bytes.size() - tensor_data_offset - begin) {
            throw std::invalid_argument("Tensor byte range extends beyond the file data.");
        }

        parsed_header.tensor_infos.push_back(std::move(tensor_info));
    }

    return parsed_header;
}

} // namespace detail

SafetensorsFile::SafetensorsFile(std::vector<std::byte> file_bytes, std::size_t tensor_data_offset,
                                 std::vector<tensors::TensorInfo> tensor_infos,
                                 std::unordered_map<std::string, std::string> metadata)
    : file_bytes_(std::move(file_bytes)), tensor_data_offset_(tensor_data_offset),
      tensor_infos_(std::move(tensor_infos)), metadata_(std::move(metadata)) {
    for (std::size_t index = 0; index < tensor_infos_.size(); ++index) {
        const auto [_, inserted] = tensor_indices_.emplace(tensor_infos_[index].name, index);
        if (!inserted) {
            throw std::invalid_argument("Duplicate tensor names are not allowed.");
        }
    }
}

SafetensorsFile SafetensorsFile::from_bytes(std::vector<std::byte> file_bytes) {
    const auto parsed_header = detail::parse_header(file_bytes);
    return SafetensorsFile(std::move(file_bytes), parsed_header.tensor_data_offset, parsed_header.tensor_infos,
                           parsed_header.metadata);
}

SafetensorsFile SafetensorsFile::from_file(const std::filesystem::path& path) {
    return from_bytes(io::read_binary_file(path, "safetensors file"));
}

bool SafetensorsFile::contains_tensor(std::string_view name) const {
    return tensor_indices_.contains(std::string(name));
}

const std::unordered_map<std::string, std::string>& SafetensorsFile::metadata() const {
    return metadata_;
}

const std::vector<tensors::TensorInfo>& SafetensorsFile::tensors() const {
    return tensor_infos_;
}

const tensors::TensorInfo& SafetensorsFile::tensor_info(std::string_view name) const {
    return tensor_infos_.at(tensor_index(name));
}

tensors::TensorView SafetensorsFile::tensor_view(std::string_view name) const {
    const tensors::TensorInfo& info = tensor_info(name);
    return tensors::TensorView(
        info,
        std::span<const std::byte>(file_bytes_).subspan(tensor_data_offset_ + info.byte_offset, info.byte_size()));
}

std::size_t SafetensorsFile::tensor_index(std::string_view name) const {
    const auto iterator = tensor_indices_.find(std::string(name));
    if (iterator == tensor_indices_.end()) {
        throw std::out_of_range("Tensor was not found in safetensors file.");
    }

    return iterator->second;
}

} // namespace cppinf::files
