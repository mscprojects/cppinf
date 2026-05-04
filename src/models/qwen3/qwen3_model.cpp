#include "models/qwen3/qwen3_model.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/checked.h"
#include "loaders/hf/hf_model_files.h"
#include "nn/qwen_decoder_block.h"
#include "ops/matmul.h"
#include "ops/nn_ops.h"
#include "ops/tensor_ops.h"
#include "tensors/dtype.h"
#include "tensors/shape.h"
#include "tensors/tensor_info.h"
#include "tensors/tensor_utils.h"

namespace cppinf::models::qwen3 {
namespace {

struct PackedTokenBatch {
    std::vector<std::int64_t> token_ids;
    std::vector<std::size_t> sequence_lengths;
    std::size_t batch_size{};
    std::size_t max_sequence_length{};
};

using common::checked_positive_dim_to_size;
using common::checked_size_to_dim;

std::string layer_tensor_name(std::size_t layer_index, std::string_view suffix) {
    return fmt::format("model.layers.{}.{}", layer_index, suffix);
}

void validate_model_config(const loaders::hf::HfConfig& config) {
    if (config.model_type != "qwen3") {
        throw std::invalid_argument("Qwen3Model requires model_type='qwen3'.");
    }

    if (!config.tie_word_embeddings) {
        throw std::invalid_argument("Qwen3Model currently requires tied word embeddings.");
    }

    if (config.num_hidden_layers == 0 || config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
        config.head_dim == 0) {
        throw std::invalid_argument("Qwen3Model requires non-zero layer and head configuration.");
    }

    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::invalid_argument("Qwen3Model requires num_attention_heads to divide by num_key_value_heads.");
    }
}

void validate_required_tensors(const loaders::hf::HfConfig& config, const files::SafetensorsFile& weights) {
    weights.tensor_info("model.embed_tokens.weight");
    weights.tensor_info("model.norm.weight");
    for (std::size_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index) {
        weights.tensor_info(layer_tensor_name(layer_index, "input_layernorm.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "post_attention_layernorm.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "self_attn.q_proj.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "self_attn.q_norm.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "self_attn.k_proj.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "self_attn.k_norm.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "self_attn.v_proj.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "self_attn.o_proj.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "mlp.gate_proj.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "mlp.up_proj.weight"));
        weights.tensor_info(layer_tensor_name(layer_index, "mlp.down_proj.weight"));
    }
}

nn::QwenDecoderBlockWeights make_layer_weights(const files::SafetensorsFile& weights, std::size_t layer_index) {
    return nn::QwenDecoderBlockWeights{
        .input_layernorm_weight = weights.tensor_view(layer_tensor_name(layer_index, "input_layernorm.weight")),
        .post_attention_layernorm_weight =
            weights.tensor_view(layer_tensor_name(layer_index, "post_attention_layernorm.weight")),
        .attention =
            nn::QwenAttentionWeights{
                .q_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "self_attn.q_proj.weight")),
                .q_norm_weight = weights.tensor_view(layer_tensor_name(layer_index, "self_attn.q_norm.weight")),
                .k_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "self_attn.k_proj.weight")),
                .k_norm_weight = weights.tensor_view(layer_tensor_name(layer_index, "self_attn.k_norm.weight")),
                .v_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "self_attn.v_proj.weight")),
                .o_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "self_attn.o_proj.weight")),
            },
        .mlp =
            nn::QwenMlpWeights{
                .gate_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "mlp.gate_proj.weight")),
                .up_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "mlp.up_proj.weight")),
                .down_proj_weight = weights.tensor_view(layer_tensor_name(layer_index, "mlp.down_proj.weight")),
            },
    };
}

PackedTokenBatch pack_token_id_batch(const TokenIdBatch& batch_token_ids) {
    if (batch_token_ids.empty()) {
        throw std::invalid_argument("Qwen3Model batched forward requires at least one sequence.");
    }

    PackedTokenBatch packed_batch;
    packed_batch.batch_size = batch_token_ids.size();
    packed_batch.sequence_lengths.reserve(batch_token_ids.size());
    for (const auto& sequence : batch_token_ids) {
        if (sequence.empty()) {
            throw std::invalid_argument(
                "Qwen3Model batched forward requires each sequence to contain at least one token.");
        }
        packed_batch.max_sequence_length = std::max(packed_batch.max_sequence_length, sequence.size());
        packed_batch.sequence_lengths.push_back(sequence.size());
    }

    packed_batch.token_ids.assign(packed_batch.batch_size * packed_batch.max_sequence_length, 0);
    for (std::size_t batch_index = 0; batch_index < packed_batch.batch_size; ++batch_index) {
        const auto& sequence = batch_token_ids[batch_index];
        std::copy(sequence.begin(), sequence.end(),
                  std::next(packed_batch.token_ids.begin(),
                            static_cast<std::ptrdiff_t>(batch_index * packed_batch.max_sequence_length)));
    }

    return packed_batch;
}

tensors::Tensor embedding_lookup(const tensors::TensorView& embedding_table, const PackedTokenBatch& token_ids) {
    if (embedding_table.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("embedding_lookup requires a rank-2 embedding table.");
    }

    const auto& dims = embedding_table.tensor_info().shape.dims();
    const auto vocab_size = checked_positive_dim_to_size(dims[0], "embedding vocab size");
    const auto hidden_size = checked_positive_dim_to_size(dims[1], "embedding hidden size");
    const auto row_byte_size = hidden_size * tensors::element_size_bytes(embedding_table.tensor_info().dtype);

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen3_hidden_states", embedding_table.tensor_info().dtype,
        tensors::Shape({checked_size_to_dim(token_ids.batch_size, "qwen3_hidden_states batch size"),
                        checked_size_to_dim(token_ids.max_sequence_length, "qwen3_hidden_states sequence length"),
                        checked_size_to_dim(hidden_size, "qwen3_hidden_states hidden size")})));

    // Each valid token id selects one row from the embedding table [vocab, hidden], and copying those rows materializes
    // the batched prompt hidden states [batch, seq, hidden]. Right-padded positions intentionally stay zero.
    for (std::size_t batch_index = 0; batch_index < token_ids.batch_size; ++batch_index) {
        for (std::size_t token_index = 0; token_index < token_ids.sequence_lengths[batch_index]; ++token_index) {
            const auto token_id = token_ids.token_ids[batch_index * token_ids.max_sequence_length + token_index];
            if (token_id < 0 || static_cast<std::uint64_t>(token_id) >= vocab_size) {
                throw std::out_of_range("Qwen3Model token id is out of vocabulary range.");
            }

            const auto source_offset = static_cast<std::size_t>(token_id) * row_byte_size;
            const auto destination_offset = (batch_index * token_ids.max_sequence_length + token_index) * row_byte_size;
            std::memcpy(result.mutable_data().data() + destination_offset,
                        embedding_table.data().data() + source_offset, row_byte_size);
        }
    }

    return result;
}

tensors::Tensor project_last_dim(const tensors::TensorView& input, const tensors::TensorView& weight,
                                 std::string_view result_name) {
    const auto transposed_weight = ops::transpose_2d(weight);
    if (input.tensor_info().shape.rank() != 3) {
        throw std::invalid_argument("project_last_dim requires rank-3 hidden states.");
    }

    const auto& dims = input.tensor_info().shape.dims();
    const auto batch_size = checked_positive_dim_to_size(dims[0], fmt::format("{} batch size", result_name));
    const auto sequence_length = checked_positive_dim_to_size(dims[1], fmt::format("{} sequence length", result_name));
    const auto hidden_size = checked_positive_dim_to_size(dims[2], fmt::format("{} hidden size", result_name));
    const auto flat_input = ops::reshape(input, tensors::Shape({static_cast<std::int64_t>(batch_size * sequence_length),
                                                                static_cast<std::int64_t>(hidden_size)}));
    auto projected = ops::matmul(flat_input, transposed_weight.view());
    return tensors::materialize_tensor(
        result_name, ops::reshape(projected.view(), tensors::Shape({static_cast<std::int64_t>(batch_size),
                                                                    static_cast<std::int64_t>(sequence_length),
                                                                    projected.tensor_info().shape.dims()[1]})));
}

} // namespace

Qwen3Model::Qwen3Model(loaders::hf::HfConfig config, files::SafetensorsFile weights)
    : config_(std::move(config)), weights_(std::move(weights)) {}

Qwen3Model Qwen3Model::from_dir(const std::filesystem::path& model_dir) {
    const auto model_files = loaders::hf::HfModelFiles::from_dir(model_dir);
    const auto config = model_files.load_config();
    validate_model_config(config);

    const auto weights = model_files.load_weights();
    validate_required_tensors(config, weights);
    return Qwen3Model(config, weights);
}

tensors::Tensor Qwen3Model::forward(std::span<const std::int64_t> token_ids) const {
    const TokenIdBatch batch_token_ids{{token_ids.begin(), token_ids.end()}};
    const auto logits = forward(batch_token_ids);
    return tensors::materialize_tensor("qwen3_logits", ops::squeeze(logits.view(), 0));
}

tensors::Tensor Qwen3Model::forward(const TokenIdBatch& token_ids) const {
    const auto packed_batch = pack_token_id_batch(token_ids);
    auto cache = make_cache(packed_batch.batch_size, packed_batch.max_sequence_length);
    return forward_cached(token_ids, cache);
}

Qwen3Cache Qwen3Model::make_cache() const {
    return make_cache(1, 1);
}

Qwen3Cache Qwen3Model::make_cache(std::size_t max_sequence_length) const {
    return make_cache(1, max_sequence_length);
}

Qwen3Cache Qwen3Model::make_cache(std::size_t batch_size, std::size_t max_sequence_length) const {
    if (batch_size == 0 || max_sequence_length == 0) {
        throw std::invalid_argument("Qwen3Model cache requires non-zero batch and sequence dimensions.");
    }

    const auto dtype = weights_.tensor_view("model.embed_tokens.weight").tensor_info().dtype;
    std::vector<nn::QwenDecoderBlockCache> layers;
    layers.reserve(config_.num_hidden_layers);
    for (std::size_t layer_index = 0; layer_index < config_.num_hidden_layers; ++layer_index) {
        layers.push_back(nn::QwenDecoderBlockCache{
            .attention = nn::make_qwen_attention_cache(layer_tensor_name(layer_index, "self_attn.cache.key"),
                                                       layer_tensor_name(layer_index, "self_attn.cache.value"), dtype,
                                                       batch_size, max_sequence_length, config_.num_key_value_heads,
                                                       config_.head_dim),
        });
    }

    return Qwen3Cache{
        .layers = std::move(layers),
        .sequence_lengths = std::vector<std::size_t>(batch_size, 0),
    };
}

tensors::Tensor Qwen3Model::forward(std::span<const std::int64_t> token_ids, Qwen3Session& session) const {
    if (session.cache_.sequence_lengths.size() != 1 ||
        session.cache_.sequence_lengths[0] != session.token_ids_.size()) {
        throw std::invalid_argument("Qwen3Model session token history must match the cache length.");
    }

    if (token_ids.size() < session.token_ids_.size()) {
        throw std::invalid_argument("Qwen3Model session sequence cannot be shorter than the cached prefix.");
    }

    if (!std::equal(session.token_ids_.begin(), session.token_ids_.end(), token_ids.begin())) {
        throw std::invalid_argument("Qwen3Model session sequence must start with the cached token prefix.");
    }

    if (token_ids.size() == session.token_ids_.size()) {
        throw std::invalid_argument("Qwen3Model session forward requires at least one uncached token.");
    }

    const auto new_token_ids = token_ids.subspan(session.token_ids_.size());
    auto logits = forward_cached(new_token_ids, session.cache_);
    session.token_ids_.insert(session.token_ids_.end(), new_token_ids.begin(), new_token_ids.end());
    return logits;
}

tensors::Tensor Qwen3Model::forward_cached(std::span<const std::int64_t> token_ids, Qwen3Cache& cache) const {
    const TokenIdBatch batch_token_ids{{token_ids.begin(), token_ids.end()}};
    const auto logits = forward_cached(batch_token_ids, cache);
    return tensors::materialize_tensor("qwen3_logits", ops::squeeze(logits.view(), 0));
}

tensors::Tensor Qwen3Model::forward_cached(const TokenIdBatch& token_ids, Qwen3Cache& cache) const {
    const auto packed_batch = pack_token_id_batch(token_ids);
    if (cache.layers.size() != config_.num_hidden_layers) {
        throw std::invalid_argument("Qwen3Model cache must have one entry per decoder layer.");
    }
    if (cache.sequence_lengths.size() != packed_batch.batch_size) {
        throw std::invalid_argument("Qwen3Model cache batch size must match the input batch size.");
    }

    // Look up one learned embedding vector per valid token id, turning token ids [batch, seq] into hidden states
    // [batch, seq, hidden]. Right-padded positions stay zero so the batch-size-1 path and padded batch path share the
    // same batched execution contract.
    auto hidden_states = embedding_lookup(weights_.tensor_view("model.embed_tokens.weight"), packed_batch);

    // Run the transformer stack one decoder block at a time. Each block keeps the outer shape [batch, seq, hidden],
    // while causal attention mixes information across each row's visible prefix and the MLP refines each position
    // independently.
    for (std::size_t layer_index = 0; layer_index < config_.num_hidden_layers; ++layer_index) {
        const auto layer_weights = make_layer_weights(weights_, layer_index);
        hidden_states = nn::qwen_decoder_block_with_cache(hidden_states.view(), packed_batch.sequence_lengths,
                                                          layer_weights, cache.layers[layer_index],
                                                          config_.num_attention_heads, config_.num_key_value_heads,
                                                          config_.head_dim, config_.rms_norm_eps, config_.rope_theta);
    }
    for (std::size_t batch_index = 0; batch_index < packed_batch.batch_size; ++batch_index) {
        cache.sequence_lengths[batch_index] += packed_batch.sequence_lengths[batch_index];
    }

    // Apply the final RMSNorm before the language-model head, and keep the shape [batch, seq, hidden].
    const auto normalized_hidden_states =
        ops::rms_norm(hidden_states.view(), weights_.tensor_view("model.norm.weight"), config_.rms_norm_eps);

    // Reuse the tied token embedding table as the output projection. Transposing [vocab, hidden] to [hidden, vocab]
    // lets us map hidden states [batch, seq, hidden] into logits [batch, seq, vocab], one vocabulary score per token
    // position.
    auto logits = project_last_dim(normalized_hidden_states.view(), weights_.tensor_view("model.embed_tokens.weight"),
                                   "qwen3_logits");
    return tensors::rename_tensor("qwen3_logits", logits);
}

const loaders::hf::HfConfig& Qwen3Model::config() const {
    return config_;
}

Qwen3Session::Qwen3Session(Qwen3Cache cache) : cache_(std::move(cache)) {}

std::span<const std::int64_t> Qwen3Session::token_ids() const {
    return token_ids_;
}

std::size_t Qwen3Session::sequence_length() const {
    return token_ids_.size();
}

} // namespace cppinf::models::qwen3
