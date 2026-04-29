#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_view.h"

namespace cppinf::nn {

struct QwenMlpWeights {
    tensors::TensorView gate_proj_weight;
    tensors::TensorView up_proj_weight;
    tensors::TensorView down_proj_weight;
};

// Applies the bias-free Qwen MLP to rank-2 [sequence, hidden] hidden states.
// gate and up project to the intermediate width, down projects back to hidden, BF16 inputs compute in f32.
tensors::Tensor qwen_mlp(const tensors::TensorView& hidden_states, const QwenMlpWeights& weights);

} // namespace cppinf::nn
