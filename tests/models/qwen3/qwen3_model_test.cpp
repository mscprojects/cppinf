#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/qwen3/qwen3_model.h"
#include "ops/tensor_ops.h"
#include "tensors/tensor.h"
#include "test_file_utils.h"
#include "test_safetensors_utils.h"
#include "test_temp_dir.h"
#include "test_tensor_utils.h"

namespace cppinf::tests {

using models::qwen3::Qwen3Model;
using models::qwen3::Qwen3Session;
using models::qwen3::TokenIdBatch;
using ops::narrow;
using ops::squeeze;
using safetensors_test_utils::append_bf16_matrix;
using safetensors_test_utils::append_bf16_vector;
using tensor_test_utils::expect_float_values_near;
using tensors::DType;
using tensors::Shape;
using tensors::Tensor;

class Qwen3ModelTest : public ::testing::Test {
  protected:
    void write_tiny_model_dir(bool tie_word_embeddings = true) {
        write_config_file(tie_word_embeddings);
        write_text_file("tokenizer.json", R"({"version":"1.0"})");
        write_text_file("tokenizer_config.json", R"({"tokenizer_class":"Qwen2Tokenizer"})");
        write_weights_file();
    }

    const std::filesystem::path& model_dir() const {
        return temp_dir_.path();
    }

    tensors::TensorView logits_batch_row(const Tensor& logits, std::size_t batch_index) const {
        return squeeze(narrow(logits.view(), 0, batch_index, 1), 0);
    }

    tensors::TensorView logits_batch_row_prefix(const Tensor& logits, std::size_t batch_index,
                                                std::size_t sequence_length) const {
        return narrow(logits_batch_row(logits, batch_index), 0, 0, sequence_length);
    }

  private:
    using ordered_json = nlohmann::ordered_json;

    void write_config_file(bool tie_word_embeddings) {
        const ordered_json config = {
            {"architectures", {"Qwen3ForCausalLM"}},
            {"bos_token_id", 1},
            {"eos_token_id", 2},
            {"head_dim", 4},
            {"hidden_size", 6},
            {"intermediate_size", 10},
            {"max_position_embeddings", 64},
            {"model_type", "qwen3"},
            {"num_attention_heads", 2},
            {"num_hidden_layers", 2},
            {"num_key_value_heads", 1},
            {"rms_norm_eps", 1e-6},
            {"rope_theta", 1000000.0},
            {"tie_word_embeddings", tie_word_embeddings},
            {"torch_dtype", "bfloat16"},
            {"vocab_size", 13},
        };
        write_text_file("config.json", config.dump(4));
    }

    void write_weights_file() {
        auto header = ordered_json::object();
        std::vector<std::byte> tensor_data;

        append_bf16_matrix(header, tensor_data, "model.embed_tokens.weight",
                           {
                               {-0.73828125f, 0.369140625f, 0.51171875f, 0.921875f, 0.240234375f, -0.030029296875f},
                               {0.330078125f, 0.859375f, -0.69140625f, 0.3203125f, 0.5f, -0.62890625f},
                               {0.75f, 0.25f, 0.51953125f, -0.169921875f, -0.71875f, -0.30078125f},
                               {0.030029296875f, 0.859375f, 0.66015625f, 0.080078125f, 0.87109375f, -0.578125f},
                               {-0.010009765625f, -0.6796875f, 0.76171875f, 0.8203125f, 1.0234375f, -1.078125f},
                               {-0.859375f, 0.71875f, -0.240234375f, 0.4609375f, 0.5703125f, -0.5703125f},
                               {1.0390625f, 0.8203125f, -0.640625f, -0.2197265625f, -1.078125f, -0.349609375f},
                               {0.369140625f, 0.1396484375f, -0.73828125f, 0.06005859375f, -0.6015625f, 0.1298828125f},
                               {0.41015625f, 0.1396484375f, -0.7890625f, 0.310546875f, 0.8203125f, -0.4609375f},
                               {-1.0703125f, -0.578125f, 1.0625f, -1.0f, -0.2890625f, -0.78125f},
                               {-0.1904296875f, 0.921875f, 0.359375f, -0.10009765625f, -0.78125f, 0.69140625f},
                               {0.240234375f, -0.671875f, 0.55859375f, -0.8203125f, 0.62890625f, 0.7109375f},
                               {-0.310546875f, -0.1298828125f, -0.330078125f, -0.6796875f, -0.8984375f, 0.91015625f},
                           });
        append_bf16_vector(header, tensor_data, "model.layers.0.input_layernorm.weight",
                           {0.78125f, 0.7109375f, 1.0703125f, 0.8515625f, 1.03125f, 0.66015625f});
        append_bf16_vector(header, tensor_data, "model.layers.0.post_attention_layernorm.weight",
                           {1.2734375f, 1.4765625f, 0.94921875f, 1.046875f, 1.328125f, 1.2734375f});
        append_bf16_matrix(header, tensor_data, "model.layers.0.self_attn.q_proj.weight",
                           {
                               {0.10986328125f, 1.0390625f, -0.55078125f, 0.96875f, -0.1298828125f, -0.330078125f},
                               {-0.030029296875f, 0.94140625f, 0.310546875f, 0.5703125f, -0.30078125f, -0.58984375f},
                               {0.169921875f, 0.7890625f, -0.44921875f, -0.41015625f, -1.0390625f, -0.240234375f},
                               {-0.66015625f, 0.73046875f, 0.08984375f, 0.7109375f, -0.71875f, -0.80078125f},
                               {-0.010009765625f, -0.62890625f, -0.51953125f, -0.1796875f, -0.240234375f, 0.80078125f},
                               {-0.30078125f, -0.10986328125f, 0.8515625f, -0.9296875f, -0.06005859375f, -0.87890625f},
                               {0.41015625f, -1.0703125f, -0.53125f, -0.96875f, -1.0234375f, -0.91015625f},
                               {-0.10009765625f, 0.58984375f, -0.83984375f, 0.76953125f, 0.671875f, -0.58984375f},
                           });
        append_bf16_vector(header, tensor_data, "model.layers.0.self_attn.q_norm.weight",
                           {0.62109375f, 1.4609375f, 1.46875f, 1.0234375f});
        append_bf16_matrix(header, tensor_data, "model.layers.0.self_attn.k_proj.weight",
                           {
                               {-0.44921875f, 0.94921875f, 0.71875f, -0.030029296875f, 0.470703125f, 0.26953125f},
                               {-0.62890625f, 0.671875f, -1.0625f, -1.0625f, -0.69140625f, 0.578125f},
                               {-0.76953125f, 0.279296875f, -0.2890625f, -1.0078125f, -0.080078125f, 1.09375f},
                               {-0.0400390625f, 0.3203125f, -0.80078125f, 0.25f, 0.76171875f, 0.150390625f},
                           });
        append_bf16_vector(header, tensor_data, "model.layers.0.self_attn.k_norm.weight",
                           {0.78125f, 0.80859375f, 0.69140625f, 1.2265625f});
        append_bf16_matrix(header, tensor_data, "model.layers.0.self_attn.v_proj.weight",
                           {
                               {-0.828125f, -0.80859375f, 0.75f, -0.96875f, 0.33984375f, 0.470703125f},
                               {0.91015625f, -0.050048828125f, 0.490234375f, 1.0078125f, -1.1015625f, 0.87890625f},
                               {0.828125f, 0.050048828125f, -1.046875f, -0.380859375f, 0.6796875f, 1.046875f},
                               {0.53125f, 0.78125f, -0.310546875f, -0.78125f, -0.98046875f, -0.73828125f},
                           });
        append_bf16_matrix(header, tensor_data, "model.layers.0.self_attn.o_proj.weight",
                           {
                               {-0.98046875f, -0.1298828125f, 0.71875f, -0.02001953125f, 0.48046875f, 0.96875f,
                                -1.1015625f, -0.94921875f},
                               {0.51953125f, -0.2099609375f, -1.046875f, -0.41015625f, 0.6484375f, 0.33984375f,
                                0.890625f, -0.87109375f},
                               {-0.80078125f, 0.02001953125f, -0.51953125f, -0.150390625f, -0.640625f, -0.6484375f,
                                0.02001953125f, 1.0703125f},
                               {0.98828125f, -0.98046875f, 0.369140625f, 1.03125f, -0.1201171875f, -0.41015625f,
                                -0.33984375f, 0.828125f},
                               {-0.26953125f, 0.1396484375f, 0.33984375f, -0.419921875f, -0.5f, -0.1298828125f,
                                0.859375f, -0.87109375f},
                               {0.2890625f, 1.0625f, -0.26953125f, -1.0078125f, 0.48046875f, -0.1396484375f,
                                -0.150390625f, -0.06982421875f},
                           });
        append_bf16_matrix(
            header, tensor_data, "model.layers.0.mlp.gate_proj.weight",
            {
                {0.890625f, 0.921875f, -1.0234375f, -0.9609375f, -0.050048828125f, 0.5390625f},
                {0.02001953125f, 0.71875f, 0.279296875f, -0.330078125f, 0.010009765625f, 0.390625f},
                {0.87890625f, 0.1396484375f, -0.98828125f, 1.046875f, -0.390625f, 0.400390625f},
                {1.0f, -0.62890625f, 0.7109375f, 1.0390625f, 0.48046875f, 1.0703125f},
                {-0.1904296875f, -0.53125f, 0.10009765625f, -0.080078125f, -1.0078125f, -0.8984375f},
                {0.51953125f, -0.69140625f, 0.55078125f, -0.96875f, 0.02001953125f, 0.73828125f},
                {0.609375f, -0.4296875f, 0.050048828125f, -0.1904296875f, -0.349609375f, 0.10009765625f},
                {-0.23046875f, 0.8515625f, -0.62890625f, -0.8515625f, -0.66015625f, 0.16015625f},
                {0.80859375f, -0.98828125f, 0.279296875f, -0.75f, -0.259765625f, 1.09375f},
                {-0.400390625f, -0.71875f, 0.050048828125f, -0.41015625f, 0.02001953125f, -0.1298828125f},
            });
        append_bf16_matrix(header, tensor_data, "model.layers.0.mlp.up_proj.weight",
                           {
                               {0.150390625f, 0.5390625f, 0.419921875f, 0.578125f, 1.078125f, 0.73046875f},
                               {-0.921875f, -0.87109375f, -0.26953125f, 0.921875f, -0.96875f, 0.8203125f},
                               {-0.91015625f, 0.48046875f, 0.51171875f, 0.73046875f, 0.83984375f, 0.279296875f},
                               {1.0625f, -0.359375f, 0.48046875f, -0.169921875f, -0.91015625f, -0.10009765625f},
                               {-0.83984375f, 0.9609375f, 0.98046875f, -0.62109375f, 0.94140625f, -1.0f},
                               {0.55859375f, 0.390625f, -1.0078125f, 0.30078125f, -0.76171875f, -0.2197265625f},
                               {-0.6484375f, -0.279296875f, 0.640625f, 0.51171875f, 0.030029296875f, 0.7890625f},
                               {-0.1201171875f, 0.859375f, -0.06982421875f, 0.6015625f, 0.578125f, -0.33984375f},
                               {-0.80859375f, -0.8984375f, -0.23046875f, 0.030029296875f, -1.0f, 0.330078125f},
                               {-0.69140625f, 0.83984375f, -0.94921875f, 0.490234375f, 0.578125f, -0.4609375f},
                           });
        append_bf16_matrix(header, tensor_data, "model.layers.0.mlp.down_proj.weight",
                           {
                               {-0.859375f, 0.1796875f, 0.71875f, -0.2197265625f, 0.44921875f, -0.44921875f,
                                0.66015625f, 0.859375f, -0.5390625f, -0.76953125f},
                               {-0.87109375f, 0.921875f, 0.69921875f, -0.9609375f, -0.030029296875f, -0.671875f,
                                -0.010009765625f, -0.41015625f, 0.51953125f, 0.98828125f},
                               {-0.87890625f, 0.380859375f, -0.98828125f, -1.03125f, 0.51953125f, -0.80078125f,
                                -0.640625f, -0.83984375f, -0.8515625f, -1.078125f},
                               {0.2001953125f, -0.73828125f, 0.26953125f, -0.69921875f, 0.10986328125f, 0.41015625f,
                                -0.4609375f, 1.0f, -0.30078125f, 0.26953125f},
                               {-0.1201171875f, 0.10986328125f, 0.439453125f, -0.6015625f, -0.470703125f,
                                -0.06005859375f, 1.03125f, 0.96875f, 0.1904296875f, 0.2890625f},
                               {1.0703125f, 0.62109375f, 1.0625f, 0.33984375f, 1.0703125f, 0.330078125f, -1.1015625f,
                                0.41015625f, -0.98046875f, 0.76171875f},
                           });
        append_bf16_vector(header, tensor_data, "model.layers.1.input_layernorm.weight",
                           {0.7890625f, 1.3828125f, 1.4296875f, 1.1328125f, 1.40625f, 0.5703125f});
        append_bf16_vector(header, tensor_data, "model.layers.1.post_attention_layernorm.weight",
                           {0.80078125f, 0.55078125f, 1.1015625f, 1.109375f, 0.51953125f, 0.73046875f});
        append_bf16_matrix(header, tensor_data, "model.layers.1.self_attn.q_proj.weight",
                           {
                               {-0.400390625f, 1.0f, -0.2001953125f, 1.078125f, -0.98046875f, -0.2197265625f},
                               {-0.2890625f, -0.78125f, -0.62109375f, 0.51171875f, -0.310546875f, -0.8984375f},
                               {-0.369140625f, -0.71875f, -0.06982421875f, 0.2099609375f, -0.33984375f, 0.859375f},
                               {0.439453125f, -0.80859375f, 0.380859375f, -0.30078125f, 0.279296875f, 0.69921875f},
                               {-0.5f, -0.921875f, 0.80859375f, -0.380859375f, -0.5f, 0.10986328125f},
                               {-0.2197265625f, -0.48046875f, 1.1015625f, 0.2099609375f, -0.470703125f, 0.51953125f},
                               {0.33984375f, -0.439453125f, 0.859375f, -0.44921875f, -0.030029296875f, 0.359375f},
                               {-0.8515625f, -0.390625f, -0.4296875f, -0.2099609375f, 0.96875f, 0.8515625f},
                           });
        append_bf16_vector(header, tensor_data, "model.layers.1.self_attn.q_norm.weight",
                           {1.4921875f, 0.80859375f, 0.6484375f, 1.2890625f});
        append_bf16_matrix(header, tensor_data, "model.layers.1.self_attn.k_proj.weight",
                           {
                               {0.69921875f, -0.30078125f, -0.98828125f, 0.80078125f, -0.890625f, 0.169921875f},
                               {1.1015625f, -0.8984375f, 0.1201171875f, -0.1796875f, -0.94140625f, 0.349609375f},
                               {0.91015625f, 0.55078125f, 0.010009765625f, -0.2001953125f, 0.9609375f, -0.330078125f},
                               {-0.7109375f, 0.94921875f, 0.87890625f, 0.8203125f, 0.62109375f, -0.7890625f},
                           });
        append_bf16_vector(header, tensor_data, "model.layers.1.self_attn.k_norm.weight",
                           {1.1171875f, 1.140625f, 1.078125f, 1.3671875f});
        append_bf16_matrix(header, tensor_data, "model.layers.1.self_attn.v_proj.weight",
                           {
                               {-0.87109375f, 0.671875f, -0.0400390625f, 0.050048828125f, 0.9609375f, -0.439453125f},
                               {-0.330078125f, 0.279296875f, 0.578125f, -0.259765625f, -0.6015625f, -0.2099609375f},
                               {-0.310546875f, 0.6484375f, 0.10009765625f, 0.0f, 0.400390625f, 0.30078125f},
                               {-0.4609375f, 0.06982421875f, -0.890625f, 0.78125f, 0.7109375f, -0.73828125f},
                           });
        append_bf16_matrix(
            header, tensor_data, "model.layers.1.self_attn.o_proj.weight",
            {
                {0.3203125f, -0.69921875f, -0.55859375f, -0.2099609375f, -0.94140625f, -0.7890625f, 0.828125f,
                 -1.0703125f},
                {0.16015625f, -0.5f, -0.4296875f, 0.16015625f, 0.91015625f, 0.259765625f, -0.5703125f, -1.1015625f},
                {0.25f, 0.87109375f, -0.490234375f, 0.240234375f, -0.150390625f, -0.55078125f, 1.0f, 0.69140625f},
                {-0.4296875f, -0.55078125f, 0.98046875f, -0.91015625f, -1.0390625f, 0.890625f, 1.09375f, 0.859375f},
                {-0.8203125f, 0.6015625f, 0.859375f, 0.30078125f, 1.0234375f, 0.380859375f, 1.1015625f, -1.078125f},
                {0.71875f, -0.87109375f, -0.609375f, 0.921875f, -0.419921875f, 0.150390625f, 0.73046875f,
                 -0.02001953125f},
            });
        append_bf16_matrix(header, tensor_data, "model.layers.1.mlp.gate_proj.weight",
                           {
                               {1.03125f, 0.578125f, -0.30078125f, -1.09375f, -0.030029296875f, 0.08984375f},
                               {-1.078125f, 0.080078125f, 0.80078125f, 0.71875f, -0.490234375f, -0.490234375f},
                               {-1.078125f, 0.02001953125f, -0.33984375f, -0.1796875f, -0.3203125f, 0.3203125f},
                               {-0.33984375f, 0.279296875f, -0.6484375f, 0.87890625f, 0.6484375f, -0.51953125f},
                               {-0.51171875f, 0.390625f, -0.73828125f, -0.51953125f, 0.369140625f, -0.921875f},
                               {0.6015625f, 0.73046875f, -0.8515625f, -1.0390625f, 0.921875f, 0.83984375f},
                               {-0.490234375f, -0.1904296875f, 0.94140625f, 0.330078125f, -0.390625f, -0.2197265625f},
                               {-0.6015625f, 0.1396484375f, 1.078125f, -0.259765625f, 0.80859375f, -0.330078125f},
                               {-0.10986328125f, -0.80859375f, 0.640625f, -0.75f, 0.98828125f, 0.6015625f},
                               {0.53125f, 0.5703125f, 0.80859375f, -0.62109375f, 0.6484375f, -0.470703125f},
                           });
        append_bf16_matrix(header, tensor_data, "model.layers.1.mlp.up_proj.weight",
                           {
                               {0.94140625f, -0.6015625f, 0.5f, -0.419921875f, 0.78125f, 0.53125f},
                               {0.58984375f, 0.87890625f, -0.050048828125f, 0.41015625f, 0.640625f, -0.26953125f},
                               {0.48046875f, 0.890625f, -0.1396484375f, -0.73828125f, 0.66015625f, 0.73046875f},
                               {-0.87109375f, -1.0703125f, -0.98046875f, -0.240234375f, 0.1904296875f, 0.2001953125f},
                               {-1.0234375f, 0.06982421875f, -0.310546875f, 0.5f, -1.046875f, 0.240234375f},
                               {0.828125f, -0.94140625f, -0.030029296875f, 0.75f, 0.3203125f, 0.08984375f},
                               {-0.369140625f, 0.4296875f, -0.240234375f, 0.8515625f, 0.73046875f, 0.1904296875f},
                               {-0.69140625f, -0.6796875f, 0.33984375f, 0.51953125f, 0.51171875f, -0.51953125f},
                               {-0.51953125f, 0.2001953125f, -0.44921875f, 0.921875f, 0.359375f, -0.55078125f},
                               {-0.23046875f, 0.71875f, 0.240234375f, 0.4296875f, -0.279296875f, -0.87109375f},
                           });
        append_bf16_matrix(header, tensor_data, "model.layers.1.mlp.down_proj.weight",
                           {
                               {-0.369140625f, -0.380859375f, 0.87109375f, -0.53125f, 0.76171875f, -0.050048828125f,
                                -0.51171875f, 0.419921875f, -0.66015625f, -0.390625f},
                               {0.7890625f, 0.640625f, -0.76171875f, -1.0703125f, -0.16015625f, -0.73046875f,
                                -0.51953125f, -0.5f, -0.030029296875f, 0.279296875f},
                               {-0.06982421875f, 0.53125f, -0.671875f, -0.0f, 0.51171875f, 0.30078125f, -0.1796875f,
                                0.2890625f, -0.2197265625f, -0.73828125f},
                               {-1.0234375f, -0.1796875f, 0.2890625f, 0.259765625f, -0.3203125f, 0.6015625f, 0.75f,
                                0.0400390625f, -0.4296875f, 0.419921875f},
                               {-0.96875f, 0.921875f, -1.0078125f, 0.5703125f, -0.8984375f, 0.349609375f, -0.44921875f,
                                0.30078125f, 0.55078125f, -0.10986328125f},
                               {0.51953125f, 0.330078125f, -0.5703125f, 0.44921875f, 0.169921875f, 0.859375f,
                                0.06005859375f, -0.369140625f, -0.1201171875f, 0.1201171875f},
                           });
        append_bf16_vector(header, tensor_data, "model.norm.weight",
                           {0.80078125f, 1.0625f, 0.7890625f, 1.0390625f, 1.4296875f, 0.8984375f});

        const auto header_text = header.dump();
        write_binary_file("model.safetensors", file_test_utils::make_safetensors_file_bytes(header_text, tensor_data));
    }

    void write_text_file(std::string_view file_name, std::string_view contents) const {
        file_test_utils::write_text_file(temp_dir_.path() / file_name, contents);
    }

    void write_binary_file(std::string_view file_name, std::span<const std::byte> bytes) const {
        file_test_utils::write_binary_file(temp_dir_.path() / file_name, bytes);
    }

    TestTempDir temp_dir_{"cppinf-qwen3-model-test"};
};

TEST_F(Qwen3ModelTest, GivenTinyBf16Checkpoint_WhenRunningForward_ThenExpectedLogitsAreReturned) {
    // Golden values generated with tests/models/qwen3/qwen3_model_oracle.py.
    // Case: bf16_tiny_model.
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    const std::vector<std::int64_t> token_ids{1, 5, 3, 2};

    const auto logits = model.forward(token_ids);

    EXPECT_EQ(std::size_t{4}, model.config().head_dim);
    EXPECT_EQ(std::size_t{2}, model.config().num_hidden_layers);
    EXPECT_EQ(DType::BF16, logits.tensor_info().dtype);
    EXPECT_EQ(Shape({4, 13}), logits.tensor_info().shape);
    expect_float_values_near(
        logits.view(),
        {0.306640625f,   0.1044921875f, 0.28515625f,     1.484375f,    4.28125f,    -0.033935546875f, -1.625f,
         -1.4296875f,    0.94921875f,   1.21875f,        -2.515625f,   0.859375f,   -2.8125f,         1.46875f,
         -0.1826171875f, -0.58984375f,  1.2421875f,      4.71875f,     0.84765625f, -2.8125f,         -1.671875f,
         0.71875f,       1.484375f,     -2.453125f,      0.275390625f, -2.796875f,  1.65625f,         1.5703125f,
         -0.8125f,       3.125f,        4.59375f,        2.34375f,     -2.34375f,   -1.9453125f,      1.765625f,
         1.109375f,      -2.21875f,     -0.08740234375f, -3.671875f,   -1.640625f,  -1.046875f,       0.66015625f,
         -2.0625f,       0.65234375f,   -1.328125f,      0.73046875f,  0.60546875f, -0.32421875f,     2.296875f,
         -1.75f,         -0.29296875f,  0.53515625f},
        0.05f);
}

TEST_F(Qwen3ModelTest, GivenTinyBf16Checkpoint_WhenRunningCachedForward_ThenLastTokenMatchesFullForward) {
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    auto cache = model.make_cache(4);
    const std::vector<std::int64_t> prompt_token_ids{1, 5, 3};
    const std::vector<std::int64_t> next_token_id{2};

    const auto prompt_logits = model.forward_cached(prompt_token_ids, cache);
    const auto next_logits = model.forward_cached(next_token_id, cache);

    EXPECT_EQ(DType::BF16, prompt_logits.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 13}), prompt_logits.tensor_info().shape);
    EXPECT_EQ(DType::BF16, next_logits.tensor_info().dtype);
    EXPECT_EQ(Shape({1, 13}), next_logits.tensor_info().shape);
    EXPECT_EQ(std::vector<std::size_t>({4}), cache.sequence_lengths);
    ASSERT_EQ(std::size_t{2}, cache.layers.size());
    EXPECT_EQ(std::vector<std::size_t>({4}), cache.layers[0].attention.sequence_lengths);
    EXPECT_EQ(std::vector<std::size_t>({4}), cache.layers[1].attention.sequence_lengths);
    expect_float_values_near(next_logits.view(),
                             {-1.640625f, -1.046875f, 0.66015625f, -2.0625f, 0.65234375f, -1.328125f, 0.73046875f,
                              0.60546875f, -0.32421875f, 2.296875f, -1.75f, -0.29296875f, 0.53515625f},
                             0.05f);
}

TEST_F(Qwen3ModelTest, GivenGrowingTokenSequence_WhenUsingSession_ThenCacheStateIsOwnedBySession) {
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    auto session = Qwen3Session(model.make_cache());
    std::vector<std::int64_t> token_ids{1, 5, 3};

    const auto prompt_logits = model.forward(token_ids, session);
    token_ids.push_back(2);
    const auto next_logits = model.forward(token_ids, session);

    EXPECT_EQ(DType::BF16, prompt_logits.tensor_info().dtype);
    EXPECT_EQ(Shape({3, 13}), prompt_logits.tensor_info().shape);
    EXPECT_EQ(DType::BF16, next_logits.tensor_info().dtype);
    EXPECT_EQ(Shape({1, 13}), next_logits.tensor_info().shape);
    EXPECT_EQ(std::size_t{4}, session.sequence_length());
    EXPECT_EQ(token_ids, std::vector<std::int64_t>(session.token_ids().begin(), session.token_ids().end()));
    expect_float_values_near(next_logits.view(),
                             {-1.640625f, -1.046875f, 0.66015625f, -2.0625f, 0.65234375f, -1.328125f, 0.73046875f,
                              0.60546875f, -0.32421875f, 2.296875f, -1.75f, -0.29296875f, 0.53515625f},
                             0.05f);
}

TEST_F(Qwen3ModelTest, GivenChangedPrefix_WhenUsingSession_ThenItThrows) {
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    auto session = Qwen3Session(model.make_cache());
    const std::vector<std::int64_t> token_ids{1, 5, 3};
    const std::vector<std::int64_t> changed_prefix_token_ids{1, 4, 3, 2};

    model.forward(token_ids, session);

    EXPECT_THROW(model.forward(changed_prefix_token_ids, session), std::invalid_argument);
}

TEST_F(Qwen3ModelTest, GivenUntiedEmbeddings_WhenLoadingModel_ThenItThrows) {
    write_tiny_model_dir(false);

    EXPECT_THROW(Qwen3Model::from_dir(model_dir()), std::invalid_argument);
}

TEST_F(Qwen3ModelTest, GivenTinyBf16Checkpoint_WhenRunningEqualLengthBatchedForward_ThenExpectedLogitsAreReturned) {
    // Golden values generated with tests/models/qwen3/qwen3_model_oracle.py.
    // Case: bf16_tiny_model_batched_equal.
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    const TokenIdBatch token_ids{{1, 5, 3, 2}, {1, 5, 3, 1}};

    const auto logits = model.forward(token_ids);

    EXPECT_EQ(DType::BF16, logits.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 4, 13}), logits.tensor_info().shape);
    expect_float_values_near(
        logits_batch_row(logits, 0),
        {0.306640625f,   0.1044921875f, 0.28515625f,     1.484375f,    4.28125f,    -0.033935546875f, -1.625f,
         -1.4296875f,    0.94921875f,   1.21875f,        -2.515625f,   0.859375f,   -2.8125f,         1.46875f,
         -0.1826171875f, -0.58984375f,  1.2421875f,      4.71875f,     0.84765625f, -2.8125f,         -1.671875f,
         0.71875f,       1.484375f,     -2.453125f,      0.275390625f, -2.796875f,  1.65625f,         1.5703125f,
         -0.8125f,       3.125f,        4.59375f,        2.34375f,     -2.34375f,   -1.9453125f,      1.765625f,
         1.109375f,      -2.21875f,     -0.08740234375f, -3.671875f,   -1.640625f,  -1.046875f,       0.66015625f,
         -2.0625f,       0.65234375f,   -1.328125f,      0.73046875f,  0.60546875f, -0.32421875f,     2.296875f,
         -1.75f,         -0.29296875f,  0.53515625f},
        0.05f);
    expect_float_values_near(
        logits_batch_row(logits, 1),
        {0.306640625f,   0.1044921875f, 0.28515625f,     1.484375f,    4.28125f,    -0.033935546875f, -1.625f,
         -1.4296875f,    0.94921875f,   1.21875f,        -2.515625f,   0.859375f,   -2.8125f,         1.46875f,
         -0.1826171875f, -0.58984375f,  1.2421875f,      4.71875f,     0.84765625f, -2.8125f,         -1.671875f,
         0.71875f,       1.484375f,     -2.453125f,      0.275390625f, -2.796875f,  1.65625f,         1.5703125f,
         -0.8125f,       3.125f,        4.59375f,        2.34375f,     -2.34375f,   -1.9453125f,      1.765625f,
         1.109375f,      -2.21875f,     -0.08740234375f, -3.671875f,   0.68359375f, 0.984375f,        0.031005859375f,
         2.25f,          4.5625f,       0.73828125f,     -1.5703125f,  -1.4921875f, 1.609375f,        0.51953125f,
         -2.5625f,       0.46875f,      -3.453125f},
        0.1f);
}

TEST_F(Qwen3ModelTest, GivenTinyBf16Checkpoint_WhenRunningMixedLengthBatchedForward_ThenValidLogitsAreReturned) {
    // Golden values generated with tests/models/qwen3/qwen3_model_oracle.py.
    // Case: bf16_tiny_model_batched_mixed.
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    const TokenIdBatch token_ids{{1, 5, 3, 2}, {1, 5, 3}};

    const auto logits = model.forward(token_ids);

    EXPECT_EQ(DType::BF16, logits.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 4, 13}), logits.tensor_info().shape);
    expect_float_values_near(
        logits_batch_row(logits, 0),
        {0.306640625f,   0.1044921875f, 0.28515625f,     1.484375f,    4.28125f,    -0.033935546875f, -1.625f,
         -1.4296875f,    0.94921875f,   1.21875f,        -2.515625f,   0.859375f,   -2.8125f,         1.46875f,
         -0.1826171875f, -0.58984375f,  1.2421875f,      4.71875f,     0.84765625f, -2.8125f,         -1.671875f,
         0.71875f,       1.484375f,     -2.453125f,      0.275390625f, -2.796875f,  1.65625f,         1.5703125f,
         -0.8125f,       3.125f,        4.59375f,        2.34375f,     -2.34375f,   -1.9453125f,      1.765625f,
         1.109375f,      -2.21875f,     -0.08740234375f, -3.671875f,   -1.640625f,  -1.046875f,       0.66015625f,
         -2.0625f,       0.65234375f,   -1.328125f,      0.73046875f,  0.60546875f, -0.32421875f,     2.296875f,
         -1.75f,         -0.29296875f,  0.53515625f},
        0.05f);
    expect_float_values_near(logits_batch_row_prefix(logits, 1, 3),
                             {0.306640625f, 0.1044921875f,   0.28515625f,    1.484375f,    4.28125f,   -0.033935546875f,
                              -1.625f,      -1.4296875f,     0.94921875f,    1.21875f,     -2.515625f, 0.859375f,
                              -2.8125f,     1.46875f,        -0.1826171875f, -0.58984375f, 1.2421875f, 4.71875f,
                              0.84765625f,  -2.8125f,        -1.671875f,     0.71875f,     1.484375f,  -2.453125f,
                              0.275390625f, -2.796875f,      1.65625f,       1.5703125f,   -0.8125f,   3.125f,
                              4.59375f,     2.34375f,        -2.34375f,      -1.9453125f,  1.765625f,  1.109375f,
                              -2.21875f,    -0.08740234375f, -3.671875f},
                             0.05f);
}

TEST_F(Qwen3ModelTest, GivenMixedLengthBatch_WhenRunningCachedForward_ThenEachRowMatchesStandaloneForward) {
    write_tiny_model_dir();

    const auto model = Qwen3Model::from_dir(model_dir());
    auto cache = model.make_cache(2, 4);
    const TokenIdBatch prompt_token_ids{{1, 5, 3}, {1, 5}};
    const TokenIdBatch next_token_ids{{2}, {3}};

    const auto prompt_logits = model.forward_cached(prompt_token_ids, cache);
    const auto next_logits = model.forward_cached(next_token_ids, cache);
    const auto first_expected = model.forward(std::vector<std::int64_t>{1, 5, 3, 2});
    const auto second_expected = model.forward(std::vector<std::int64_t>{1, 5, 3});

    EXPECT_EQ(DType::BF16, prompt_logits.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 3, 13}), prompt_logits.tensor_info().shape);
    EXPECT_EQ(DType::BF16, next_logits.tensor_info().dtype);
    EXPECT_EQ(Shape({2, 1, 13}), next_logits.tensor_info().shape);
    EXPECT_EQ(std::vector<std::size_t>({4, 3}), cache.sequence_lengths);
    ASSERT_EQ(std::size_t{2}, cache.layers.size());
    EXPECT_EQ(std::vector<std::size_t>({4, 3}), cache.layers[0].attention.sequence_lengths);
    EXPECT_EQ(std::vector<std::size_t>({4, 3}), cache.layers[1].attention.sequence_lengths);
    EXPECT_EQ(tensor_test_utils::read_float_values(logits_batch_row(next_logits, 0)),
              tensor_test_utils::read_float_values(narrow(first_expected.view(), 0, 3, 1)));
    EXPECT_EQ(tensor_test_utils::read_float_values(logits_batch_row(next_logits, 1)),
              tensor_test_utils::read_float_values(narrow(second_expected.view(), 0, 2, 1)));
}

} // namespace cppinf::tests
