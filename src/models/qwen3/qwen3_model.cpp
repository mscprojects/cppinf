#include "models/qwen3/qwen3_model.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

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

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    const auto value = static_cast<std::size_t>(dim);
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

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

tensors::Tensor embedding_lookup(const tensors::TensorView& embedding_table, std::span<const std::int64_t> token_ids) {
    if (embedding_table.tensor_info().shape.rank() != 2) {
        throw std::invalid_argument("embedding_lookup requires a rank-2 embedding table.");
    }

    if (token_ids.empty()) {
        throw std::invalid_argument("Qwen3Model forward requires at least one token id.");
    }

    const auto& dims = embedding_table.tensor_info().shape.dims();
    const auto vocab_size = checked_positive_dim_to_size(dims[0], "embedding vocab size");
    const auto hidden_size = checked_positive_dim_to_size(dims[1], "embedding hidden size");
    const auto row_byte_size = hidden_size * tensors::element_size_bytes(embedding_table.tensor_info().dtype);

    auto result = tensors::Tensor::zeros(tensors::make_result_tensor_info(
        "qwen3_hidden_states", embedding_table.tensor_info().dtype,
        tensors::Shape({static_cast<std::int64_t>(token_ids.size()), static_cast<std::int64_t>(hidden_size)})));

    // Each token id selects one row from the embedding table [vocab, hidden], and copying those rows materializes the
    // prompt hidden states [seq, hidden].
    for (std::size_t token_index = 0; token_index < token_ids.size(); ++token_index) {
        const auto token_id = token_ids[token_index];
        if (token_id < 0 || static_cast<std::uint64_t>(token_id) >= vocab_size) {
            throw std::out_of_range("Qwen3Model token id is out of vocabulary range.");
        }

        const auto source_offset = static_cast<std::size_t>(token_id) * row_byte_size;
        const auto destination_offset = token_index * row_byte_size;
        std::memcpy(result.mutable_data().data() + destination_offset, embedding_table.data().data() + source_offset,
                    row_byte_size);
    }

    return result;
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
    // Look up one learned embedding vector per token id, turning token ids [seq] into hidden states [seq, hidden].
    auto hidden_states = embedding_lookup(weights_.tensor_view("model.embed_tokens.weight"), token_ids);

    // Run the transformer stack one decoder block at a time. Each block keeps the outer shape [seq, hidden], while
    // causal attention mixes information across earlier tokens and the MLP refines each position independently. This
    // path recomputes the full sequence from scratch, so positions always start at 0.
    for (std::size_t layer_index = 0; layer_index < config_.num_hidden_layers; ++layer_index) {
        const auto layer_weights = make_layer_weights(weights_, layer_index);
        hidden_states = nn::qwen_decoder_block(hidden_states.view(), layer_weights, config_.num_attention_heads,
                                               config_.num_key_value_heads, config_.head_dim, config_.rms_norm_eps, 0,
                                               config_.rope_theta);
    }

    // Apply the final RMSNorm before the language-model head, and keep the shape [seq, hidden].
    const auto normalized_hidden_states =
        ops::rms_norm(hidden_states.view(), weights_.tensor_view("model.norm.weight"), config_.rms_norm_eps);

    // Reuse the tied token embedding table as the output projection. Transposing [vocab, hidden] to [hidden, vocab]
    // lets us map hidden states [seq, hidden] into logits [seq, vocab], one vocabulary score per token position.
    const auto transposed_embedding = ops::transpose_2d(weights_.tensor_view("model.embed_tokens.weight"));
    auto logits = ops::matmul(normalized_hidden_states.view(), transposed_embedding.view());
    return tensors::rename_tensor("qwen3_logits", logits);
}

const loaders::hf::HfConfig& Qwen3Model::config() const {
    return config_;
}

} // namespace cppinf::models::qwen3
