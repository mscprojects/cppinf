#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "loaders/hf/hf_config.h"
#include "tensors/tensor_info.h"

namespace cppinf::loaders::hf {

struct HfModelSummary {
    std::filesystem::path model_dir;
    HfConfig config;
    std::size_t metadata_count{};
    std::size_t tensor_count{};
    std::vector<tensors::TensorInfo> tensor_preview;

    bool operator==(const HfModelSummary&) const = default;
};

HfModelSummary load_model_summary(const std::filesystem::path& model_dir, std::size_t tensor_preview_limit = 8);
std::string format_model_summary(const HfModelSummary& summary);

} // namespace cppinf::loaders::hf
