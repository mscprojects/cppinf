"""Offline oracle generator for qwen_decoder_block tests against Hugging Face Qwen3DecoderLayer.

Run with:
    uv run --no-project --with transformers==4.52.3 --with torch==2.7.1 \
        --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match python tests/nn/qwen_decoder_block_oracle.py
"""

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding


def round_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 100) / 100


def quantize_to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return x.clone()
    if dtype == torch.bfloat16:
        return x.to(torch.bfloat16).float()
    raise ValueError(f"Unsupported dtype: {dtype}")


def make_config(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    norm_epsilon: float,
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=13,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=64,
        rms_norm_eps=norm_epsilon,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        use_sliding_window=False,
        sliding_window=None,
    )


def make_causal_mask(sequence_length: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, sequence_length, sequence_length), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def copy_weight(parameter: torch.nn.Parameter, weight: torch.Tensor, dtype: torch.dtype) -> None:
    parameter.copy_(weight.to(dtype))


def run_case(
    label: str,
    hidden_states: torch.Tensor,
    input_ln: torch.Tensor,
    post_ln: torch.Tensor,
    q_proj: torch.Tensor,
    q_norm: torch.Tensor,
    k_proj: torch.Tensor,
    k_norm: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    norm_epsilon: float,
    output_dtype: torch.dtype,
) -> None:
    hidden_size = hidden_states.shape[-1]
    intermediate_size = gate_proj.shape[0]
    config = make_config(
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        norm_epsilon,
    )
    decoder_layer = Qwen3DecoderLayer(config, layer_idx=0).eval().to(output_dtype)
    rotary_embedding = Qwen3RotaryEmbedding(config)

    with torch.no_grad():
        copy_weight(decoder_layer.input_layernorm.weight, input_ln, output_dtype)
        copy_weight(decoder_layer.post_attention_layernorm.weight, post_ln, output_dtype)
        copy_weight(decoder_layer.self_attn.q_proj.weight, q_proj, output_dtype)
        copy_weight(decoder_layer.self_attn.q_norm.weight, q_norm, output_dtype)
        copy_weight(decoder_layer.self_attn.k_proj.weight, k_proj, output_dtype)
        copy_weight(decoder_layer.self_attn.k_norm.weight, k_norm, output_dtype)
        copy_weight(decoder_layer.self_attn.v_proj.weight, v_proj, output_dtype)
        copy_weight(decoder_layer.self_attn.o_proj.weight, o_proj, output_dtype)
        copy_weight(decoder_layer.mlp.gate_proj.weight, gate_proj, output_dtype)
        copy_weight(decoder_layer.mlp.up_proj.weight, up_proj, output_dtype)
        copy_weight(decoder_layer.mlp.down_proj.weight, down_proj, output_dtype)

        batched_hidden_states = hidden_states.to(output_dtype).unsqueeze(0)
        sequence_length = batched_hidden_states.shape[1]
        position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
        position_embeddings = tuple(x.to(output_dtype) for x in rotary_embedding(batched_hidden_states, position_ids))
        attention_mask = make_causal_mask(sequence_length, output_dtype)

        result = decoder_layer(
            batched_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )[0]

    print(f"{label}={repr(result[0].float().tolist())}")


if __name__ == "__main__":
    hidden_size = 6
    intermediate_size = 10
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 4
    norm_epsilon = 1e-6

    torch.manual_seed(9003)
    hidden_states_f32 = round_tensor(torch.empty(3, hidden_size).uniform_(-1.25, 1.25))
    input_ln_f32 = round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5))
    post_ln_f32 = round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5))
    q_proj_f32 = round_tensor(torch.empty(num_attention_heads * head_dim, hidden_size).uniform_(-1.25, 1.25))
    q_norm_f32 = round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5))
    k_proj_f32 = round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.25, 1.25))
    k_norm_f32 = round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5))
    v_proj_f32 = round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.25, 1.25))
    o_proj_f32 = round_tensor(torch.empty(hidden_size, num_attention_heads * head_dim).uniform_(-1.25, 1.25))
    gate_proj_f32 = round_tensor(torch.empty(intermediate_size, hidden_size).uniform_(-1.25, 1.25))
    up_proj_f32 = round_tensor(torch.empty(intermediate_size, hidden_size).uniform_(-1.25, 1.25))
    down_proj_f32 = round_tensor(torch.empty(hidden_size, intermediate_size).uniform_(-1.25, 1.25))

    run_case(
        "f32_basic",
        hidden_states_f32,
        input_ln_f32,
        post_ln_f32,
        q_proj_f32,
        q_norm_f32,
        k_proj_f32,
        k_norm_f32,
        v_proj_f32,
        o_proj_f32,
        gate_proj_f32,
        up_proj_f32,
        down_proj_f32,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        norm_epsilon,
        torch.float32,
    )
    run_case(
        "bf16_basic",
        quantize_to_dtype(hidden_states_f32, torch.bfloat16),
        quantize_to_dtype(input_ln_f32, torch.bfloat16),
        quantize_to_dtype(post_ln_f32, torch.bfloat16),
        quantize_to_dtype(q_proj_f32, torch.bfloat16),
        quantize_to_dtype(q_norm_f32, torch.bfloat16),
        quantize_to_dtype(k_proj_f32, torch.bfloat16),
        quantize_to_dtype(k_norm_f32, torch.bfloat16),
        quantize_to_dtype(v_proj_f32, torch.bfloat16),
        quantize_to_dtype(o_proj_f32, torch.bfloat16),
        quantize_to_dtype(gate_proj_f32, torch.bfloat16),
        quantize_to_dtype(up_proj_f32, torch.bfloat16),
        quantize_to_dtype(down_proj_f32, torch.bfloat16),
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        norm_epsilon,
        torch.bfloat16,
    )
