#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "files/safetensors_file.h"
#include "loaders/hf/hf_config.h"
#include "nn/qwen_decoder_block.h"
#include "tensors/tensor.h"

namespace cppinf::models::qwen3 {

struct Qwen3ModelCache {
    std::vector<nn::QwenDecoderBlockCache> layers;
    std::size_t sequence_length{};
};

class Qwen3Model {
  public:
    static Qwen3Model from_dir(const std::filesystem::path& model_dir);

    // Runs token ids through the full Qwen3 model and returns rank-2 [sequence, vocab] logits.
    // Requires tied word embeddings and uses a temporary empty cache. Output dtype matches the loaded checkpoint dtype.
    tensors::Tensor forward(std::span<const std::int64_t> token_ids) const;

    // Creates an empty per-layer cache for incremental decoding.
    Qwen3ModelCache make_cache() const;

    // Runs token ids through the model while appending each layer's K/V tensors to cache.
    // The next call must pass tokens that immediately follow the cached prefix.
    tensors::Tensor forward_cached(std::span<const std::int64_t> token_ids, Qwen3ModelCache& cache) const;

    const loaders::hf::HfConfig& config() const;

  private:
    Qwen3Model(loaders::hf::HfConfig config, files::SafetensorsFile weights);

    loaders::hf::HfConfig config_;
    files::SafetensorsFile weights_;
};

} // namespace cppinf::models::qwen3
