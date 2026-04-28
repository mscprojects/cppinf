#include "loaders/hf/hf_model_summary.h"

#include <algorithm>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "loaders/hf/hf_model_files.h"
#include "tensors/dtype.h"
#include "tensors/shape.h"

namespace cppinf::loaders::hf {

HfModelSummary load_model_summary(const std::filesystem::path& model_dir, std::size_t tensor_preview_limit) {
    const HfModelFiles model_files = HfModelFiles::from_dir(model_dir);
    const HfConfig config = model_files.load_config();
    const files::SafetensorsFile weights = model_files.load_weights();

    const auto& tensors = weights.tensors();
    const std::size_t preview_size = std::min(tensor_preview_limit, tensors.size());
    std::vector<tensors::TensorInfo> tensor_preview(tensors.begin(), tensors.begin() + preview_size);

    return HfModelSummary{
        .model_dir = model_dir,
        .config = config,
        .metadata_count = weights.metadata().size(),
        .tensor_count = tensors.size(),
        .tensor_preview = std::move(tensor_preview),
    };
}

std::string format_model_summary(const HfModelSummary& summary) {
    fmt::memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), "HF model directory: {}\n", summary.model_dir.string());
    fmt::format_to(std::back_inserter(buffer), "Model type: {}\n", summary.config.model_type);
    fmt::format_to(std::back_inserter(buffer), "Architectures: {}\n", fmt::join(summary.config.architectures, ", "));
    fmt::format_to(std::back_inserter(buffer), "Tensor dtype: {}\n", tensors::to_string(summary.config.tensor_dtype));
    fmt::format_to(std::back_inserter(buffer), "Hidden size: {}\n", summary.config.hidden_size);
    fmt::format_to(std::back_inserter(buffer), "Intermediate size: {}\n", summary.config.intermediate_size);
    fmt::format_to(std::back_inserter(buffer), "Max position embeddings: {}\n", summary.config.max_position_embeddings);
    fmt::format_to(std::back_inserter(buffer), "Hidden layers: {}\n", summary.config.num_hidden_layers);
    fmt::format_to(std::back_inserter(buffer), "Attention heads: {}\n", summary.config.num_attention_heads);
    fmt::format_to(std::back_inserter(buffer), "Key/value heads: {}\n", summary.config.num_key_value_heads);
    fmt::format_to(std::back_inserter(buffer), "Vocabulary size: {}\n", summary.config.vocab_size);
    fmt::format_to(std::back_inserter(buffer), "BOS token id: {}\n", summary.config.bos_token_id);
    fmt::format_to(std::back_inserter(buffer), "EOS token id: {}\n", summary.config.eos_token_id);
    fmt::format_to(std::back_inserter(buffer), "Metadata entries: {}\n", summary.metadata_count);
    fmt::format_to(std::back_inserter(buffer), "Tensor count: {}\n", summary.tensor_count);
    fmt::format_to(std::back_inserter(buffer), "Tensor preview:\n");

    for (const auto& tensor_info : summary.tensor_preview) {
        fmt::format_to(std::back_inserter(buffer), "  - {} | dtype={} | shape={} | offset={} | bytes={}\n",
                       tensor_info.name, tensors::to_string(tensor_info.dtype), tensors::to_string(tensor_info.shape),
                       tensor_info.byte_offset, tensor_info.byte_size());
    }

    return fmt::to_string(buffer);
}

} // namespace cppinf::loaders::hf
