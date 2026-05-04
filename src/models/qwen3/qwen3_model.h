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

using TokenIdBatch = std::vector<std::vector<std::int64_t>>;

struct Qwen3Cache {
    std::vector<nn::QwenDecoderBlockCache> layers;
    std::vector<std::size_t> sequence_lengths;
};

class Qwen3Session;

class Qwen3Model {
  public:
    static Qwen3Model from_dir(const std::filesystem::path& model_dir);

    // Runs token ids through the full Qwen3 model and returns rank-2 [sequence, vocab] logits.
    // Requires tied word embeddings and uses a temporary empty cache. Output dtype matches the loaded checkpoint dtype.
    tensors::Tensor forward(std::span<const std::int64_t> token_ids) const;

    // Runs a right-padded batch of token-id sequences through the full Qwen3 model and returns rank-3
    // [batch, sequence, vocab] logits. Only the prefix up to each input row's length is meaningful.
    tensors::Tensor forward(const TokenIdBatch& token_ids) const;

    // Creates an empty per-layer cache whose K/V tensors allocate and grow on demand.
    Qwen3Cache make_cache() const;

    // Creates an empty per-layer cache with initial storage for up to max_sequence_length tokens.
    Qwen3Cache make_cache(std::size_t max_sequence_length) const;

    // Creates an empty batched per-layer cache with initial storage for up to max_sequence_length tokens per batch row.
    Qwen3Cache make_cache(std::size_t batch_size, std::size_t max_sequence_length) const;

    // Runs only the uncached suffix of the session's complete growing token sequence.
    // The sequence must start with the session's previous tokens and include at least one new token.
    tensors::Tensor forward(std::span<const std::int64_t> token_ids, Qwen3Session& session) const;

    // Runs token ids through the model while appending each layer's K/V tensors to cache.
    // The next call must pass tokens that immediately follow the cached prefix.
    tensors::Tensor forward_cached(std::span<const std::int64_t> token_ids, Qwen3Cache& cache) const;

    // Runs a right-padded batch of token-id suffixes through the model while appending each layer's K/V tensors to
    // cache. The next call must pass only the tokens that immediately follow each cached prefix.
    tensors::Tensor forward_cached(const TokenIdBatch& token_ids, Qwen3Cache& cache) const;

    const loaders::hf::HfConfig& config() const;

  private:
    Qwen3Model(loaders::hf::HfConfig config, files::SafetensorsFile weights);

    loaders::hf::HfConfig config_;
    files::SafetensorsFile weights_;
};

class Qwen3Session {
  public:
    explicit Qwen3Session(Qwen3Cache cache);

    std::span<const std::int64_t> token_ids() const;
    std::size_t sequence_length() const;

  private:
    friend class Qwen3Model;

    Qwen3Cache cache_;
    std::vector<std::int64_t> token_ids_;
};

} // namespace cppinf::models::qwen3
