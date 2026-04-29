#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/qwen3/qwen3_model.h"
#include "tensors/bfloat16.h"
#include "tensors/tensor.h"
#include "test_temp_dir.h"

using cppinf::models::qwen3::Qwen3Model;
using cppinf::tensors::bfloat16_bits_to_float;
using cppinf::tensors::DType;
using cppinf::tensors::float_to_bfloat16_bits;
using cppinf::tensors::Shape;

class Qwen3ModelTest : public ::testing::Test {
  protected:
    void write_tiny_model_dir(bool tie_word_embeddings = true) {
        write_config_file(tie_word_embeddings);
        write_text_file("tokenizer.json", R"({"version":"1.0"})");
        write_text_file("tokenizer_config.json", R"({"tokenizer_class":"Qwen2Tokenizer"})");
        write_weights_file();
    }

    std::vector<float> read_bf16_values(const cppinf::tensors::Tensor& tensor) const {
        std::vector<float> values;
        values.reserve(tensor.tensor_info().shape.num_elements());
        for (std::size_t index = 0; index < tensor.tensor_info().shape.num_elements(); ++index) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, tensor.bytes().data() + index * sizeof(std::uint16_t), sizeof(std::uint16_t));
            values.push_back(bfloat16_bits_to_float(bits));
        }
        return values;
    }

    void expect_bf16_values_near(const cppinf::tensors::Tensor& tensor, std::initializer_list<float> expected,
                                 float tolerance) const {
        const auto actual_values = read_bf16_values(tensor);
        ASSERT_EQ(expected.size(), actual_values.size());

        std::size_t index = 0;
        for (const auto expected_value : expected) {
            EXPECT_NEAR(expected_value, actual_values[index], tolerance) << "index=" << index;
            ++index;
        }
    }

    const std::filesystem::path& model_dir() const {
        return temp_dir_.path();
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
        write_binary_file("model.safetensors", make_safetensors_file_bytes(header_text, tensor_data));
    }

    void append_bf16_vector(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                            std::initializer_list<float> values) {
        append_bf16_tensor(header, tensor_data, name, {static_cast<std::int64_t>(values.size())},
                           std::vector<float>(values));
    }

    void append_bf16_matrix(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                            std::initializer_list<std::initializer_list<float>> rows) {
        if (rows.size() == 0) {
            throw std::invalid_argument("append_bf16_matrix requires at least one row.");
        }

        const auto column_count = rows.begin()->size();
        if (column_count == 0) {
            throw std::invalid_argument("append_bf16_matrix requires at least one column.");
        }

        std::vector<float> values;
        values.reserve(rows.size() * column_count);
        for (const auto& row : rows) {
            if (row.size() != column_count) {
                throw std::invalid_argument("append_bf16_matrix requires rows with equal width.");
            }
            values.insert(values.end(), row.begin(), row.end());
        }

        append_bf16_tensor(header, tensor_data, name,
                           {static_cast<std::int64_t>(rows.size()), static_cast<std::int64_t>(column_count)}, values);
    }

    void append_bf16_tensor(ordered_json& header, std::vector<std::byte>& tensor_data, std::string_view name,
                            const std::vector<std::int64_t>& dims, const std::vector<float>& values) {
        const auto begin_offset = tensor_data.size();
        for (const auto value : values) {
            const auto bits = float_to_bfloat16_bits(value);
            const auto* bytes = reinterpret_cast<const std::byte*>(&bits);
            tensor_data.insert(tensor_data.end(), bytes, bytes + sizeof(bits));
        }
        const auto end_offset = tensor_data.size();

        header[std::string(name)] = ordered_json{
            {"dtype", "BF16"},
            {"shape", dims},
            {"data_offsets", {begin_offset, end_offset}},
        };
    }

    std::vector<std::byte> make_safetensors_file_bytes(std::string_view header_json,
                                                       std::span<const std::byte> tensor_data) const {
        std::vector<std::byte> bytes;
        append_u64_le(static_cast<std::uint64_t>(header_json.size()), bytes);
        for (const auto character : header_json) {
            bytes.push_back(static_cast<std::byte>(character));
        }
        bytes.insert(bytes.end(), tensor_data.begin(), tensor_data.end());
        return bytes;
    }

    void append_u64_le(std::uint64_t value, std::vector<std::byte>& bytes) const {
        for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
            bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 0xffU));
        }
    }

    void write_text_file(std::string_view file_name, std::string_view contents) const {
        std::ofstream output(temp_dir_.path() / file_name);
        output << contents;
    }

    void write_binary_file(std::string_view file_name, std::span<const std::byte> bytes) const {
        std::ofstream output(temp_dir_.path() / file_name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
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
    expect_bf16_values_near(logits,
                            {0.3046875f,  0.12353515625f,  0.287109375f,  1.4921875f,  4.28125f,    -0.0245361328125f,
                             -1.609375f,  -1.421875f,      0.96484375f,   1.203125f,   -2.515625f,  0.84375f,
                             -2.828125f,  1.484375f,       -0.205078125f, -0.578125f,  1.203125f,   4.71875f,
                             0.83203125f, -2.796875f,      -1.65625f,     0.69140625f, 1.453125f,   -2.4375f,
                             0.24609375f, -2.765625f,      1.65625f,      1.6171875f,  -0.84375f,   3.171875f,
                             4.5625f,     2.390625f,       -2.34375f,     -1.9453125f, 1.796875f,   1.078125f,
                             -2.203125f,  -0.08837890625f, -3.671875f,    -1.6328125f, -1.03125f,   0.65234375f,
                             -2.046875f,  0.65625f,        -1.3046875f,   0.73046875f, 0.60546875f, -0.3125f,
                             2.296875f,   -1.7578125f,     -0.30078125f,  0.5234375f},
                            1e-6f);
}

TEST_F(Qwen3ModelTest, GivenUntiedEmbeddings_WhenLoadingModel_ThenItThrows) {
    write_tiny_model_dir(false);

    EXPECT_THROW(static_cast<void>(Qwen3Model::from_dir(model_dir())), std::invalid_argument);
}
