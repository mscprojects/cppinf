#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include "files/safetensors_file.h"
#include "loaders/hf/hf_config.h"
#include "tensors/tensor.h"

namespace cppinf::models::qwen3 {

class Qwen3Model {
  public:
    static Qwen3Model from_dir(const std::filesystem::path& model_dir);

    // Runs token ids through the full Qwen3 model and returns rank-2 [sequence, vocab] logits.
    // Requires tied word embeddings and no cache. Output dtype matches the loaded checkpoint dtype.
    tensors::Tensor forward(std::span<const std::int64_t> token_ids) const;

    const loaders::hf::HfConfig& config() const;

  private:
    Qwen3Model(loaders::hf::HfConfig config, files::SafetensorsFile weights);

    loaders::hf::HfConfig config_;
    files::SafetensorsFile weights_;
};

} // namespace cppinf::models::qwen3
