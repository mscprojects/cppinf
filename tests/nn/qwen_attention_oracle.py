"""Offline oracle generator for qwen_attention tests against Hugging Face Qwen3Attention.

Run with:
    uv run --no-project --with transformers==4.52.3 --with torch==2.7.1 \
        --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match python tests/nn/qwen_attention_oracle.py
"""

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding


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
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    norm_epsilon: float,
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=13,
        hidden_size=hidden_size,
        intermediate_size=10,
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
    q_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    norm_epsilon: float,
    output_dtype: torch.dtype,
) -> None:
    config = make_config(hidden_states.shape[-1], num_attention_heads, num_key_value_heads, head_dim, norm_epsilon)
    attention = Qwen3Attention(config, layer_idx=0).eval().to(output_dtype)
    rotary_embedding = Qwen3RotaryEmbedding(config)

    with torch.no_grad():
        copy_weight(attention.q_proj.weight, q_proj_weight, output_dtype)
        copy_weight(attention.q_norm.weight, q_norm_weight, output_dtype)
        copy_weight(attention.k_proj.weight, k_proj_weight, output_dtype)
        copy_weight(attention.k_norm.weight, k_norm_weight, output_dtype)
        copy_weight(attention.v_proj.weight, v_proj_weight, output_dtype)
        copy_weight(attention.o_proj.weight, o_proj_weight, output_dtype)

        batched_hidden_states = hidden_states.to(output_dtype).unsqueeze(0)
        sequence_length = batched_hidden_states.shape[1]
        position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
        position_embeddings = tuple(x.to(output_dtype) for x in rotary_embedding(batched_hidden_states, position_ids))
        attention_mask = make_causal_mask(sequence_length, output_dtype)

        result = attention(
            batched_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )[0]

    print(f"{label}={repr(result[0].float().tolist())}")


if __name__ == "__main__":
    hidden_size = 6
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 4
    norm_epsilon = 1e-6

    torch.manual_seed(9001)
    hidden_states_f32 = round_tensor(torch.empty(3, hidden_size).uniform_(-1.4, 1.4))
    q_proj_weight_f32 = round_tensor(torch.empty(num_attention_heads * head_dim, hidden_size).uniform_(-1.4, 1.4))
    q_norm_weight_f32 = round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5))
    k_proj_weight_f32 = round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.4, 1.4))
    k_norm_weight_f32 = round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5))
    v_proj_weight_f32 = round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.4, 1.4))
    o_proj_weight_f32 = round_tensor(torch.empty(hidden_size, num_attention_heads * head_dim).uniform_(-1.4, 1.4))

    run_case(
        "f32_explicit_head_dim",
        hidden_states_f32,
        q_proj_weight_f32,
        q_norm_weight_f32,
        k_proj_weight_f32,
        k_norm_weight_f32,
        v_proj_weight_f32,
        o_proj_weight_f32,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        norm_epsilon,
        torch.float32,
    )
    run_case(
        "bf16_explicit_head_dim",
        quantize_to_dtype(hidden_states_f32, torch.bfloat16),
        quantize_to_dtype(q_proj_weight_f32, torch.bfloat16),
        quantize_to_dtype(q_norm_weight_f32, torch.bfloat16),
        quantize_to_dtype(k_proj_weight_f32, torch.bfloat16),
        quantize_to_dtype(k_norm_weight_f32, torch.bfloat16),
        quantize_to_dtype(v_proj_weight_f32, torch.bfloat16),
        quantize_to_dtype(o_proj_weight_f32, torch.bfloat16),
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        norm_epsilon,
        torch.bfloat16,
    )

    grouped_hidden_size = 6
    grouped_num_attention_heads = 4
    grouped_num_key_value_heads = 2
    grouped_head_dim = 2

    torch.manual_seed(1337)
    grouped_hidden_states_f32 = round_tensor(torch.empty(3, grouped_hidden_size).uniform_(-1.4, 1.4))
    grouped_q_proj_weight_f32 = round_tensor(
        torch.empty(grouped_num_attention_heads * grouped_head_dim, grouped_hidden_size).uniform_(-1.4, 1.4)
    )
    grouped_q_norm_weight_f32 = round_tensor(torch.empty(grouped_head_dim).uniform_(0.5, 1.5))
    grouped_k_proj_weight_f32 = round_tensor(
        torch.empty(grouped_num_key_value_heads * grouped_head_dim, grouped_hidden_size).uniform_(-1.4, 1.4)
    )
    grouped_k_norm_weight_f32 = round_tensor(torch.empty(grouped_head_dim).uniform_(0.5, 1.5))
    grouped_v_proj_weight_f32 = round_tensor(
        torch.empty(grouped_num_key_value_heads * grouped_head_dim, grouped_hidden_size).uniform_(-1.4, 1.4)
    )
    grouped_o_proj_weight_f32 = round_tensor(
        torch.empty(grouped_hidden_size, grouped_num_attention_heads * grouped_head_dim).uniform_(-1.4, 1.4)
    )

    run_case(
        "f32_grouped_kv_heads",
        grouped_hidden_states_f32,
        grouped_q_proj_weight_f32,
        grouped_q_norm_weight_f32,
        grouped_k_proj_weight_f32,
        grouped_k_norm_weight_f32,
        grouped_v_proj_weight_f32,
        grouped_o_proj_weight_f32,
        grouped_num_attention_heads,
        grouped_num_key_value_heads,
        grouped_head_dim,
        norm_epsilon,
        torch.float32,
    )
