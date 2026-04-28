#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tensors/tensor_info.h"
#include "tensors/tensor_view.h"

namespace cppinf::files {

class SafetensorsFile {
  public:
    static SafetensorsFile from_bytes(std::vector<std::byte> file_bytes);
    static SafetensorsFile from_file(const std::filesystem::path& path);

    bool contains_tensor(std::string_view name) const;
    const std::unordered_map<std::string, std::string>& metadata() const;
    const std::vector<tensors::TensorInfo>& tensors() const;
    const tensors::TensorInfo& tensor_info(std::string_view name) const;
    tensors::TensorView tensor_view(std::string_view name) const;

  private:
    SafetensorsFile(std::vector<std::byte> file_bytes, std::size_t tensor_data_offset,
                    std::vector<tensors::TensorInfo> tensor_infos,
                    std::unordered_map<std::string, std::string> metadata);

    std::size_t tensor_index(std::string_view name) const;

    std::vector<std::byte> file_bytes_;
    std::size_t tensor_data_offset_{};
    std::vector<tensors::TensorInfo> tensor_infos_;
    std::unordered_map<std::string, std::size_t> tensor_indices_;
    std::unordered_map<std::string, std::string> metadata_;
};

} // namespace cppinf::files
