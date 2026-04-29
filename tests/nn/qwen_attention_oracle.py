"""Offline oracle generator for qwen_attention tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/qwen_attention_oracle.py
"""

import torch


def round_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 100) / 100


def quantize_to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return x.clone()
    if dtype == torch.bfloat16:
        return x.to(torch.bfloat16).float()
    raise ValueError(f"Unsupported dtype: {dtype}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(
    x: torch.Tensor, output_dtype: torch.dtype, position_offset: int = 1, base: float = 1_000_000.0
) -> torch.Tensor:
    dim = x.shape[-1]
    positions = torch.arange(position_offset, position_offset + x.shape[-2], dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    return quantize_to_dtype(x * cos + rotate_half(x) * sin, output_dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight, output_dtype)


def linear_project(x: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(x @ weight.transpose(0, 1), output_dtype)


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    seq, merged = x.shape
    assert merged == heads * head_dim
    return x.reshape(seq, heads, head_dim).permute(1, 0, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    heads, seq, dim = x.shape
    return x.permute(1, 0, 2).contiguous().reshape(seq, heads * dim)


def causal_self_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    scores = scores.masked_fill(torch.triu(torch.ones(query.shape[1], key.shape[1], dtype=torch.bool), diagonal=1),
                                float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), value.float())
    return quantize_to_dtype(out, output_dtype)


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
    q = linear_project(hidden_states, q_proj_weight, output_dtype)
    k = linear_project(hidden_states, k_proj_weight, output_dtype)
    v = linear_project(hidden_states, v_proj_weight, output_dtype)

    q = split_heads(q, num_attention_heads, head_dim)
    k = split_heads(k, num_key_value_heads, head_dim)
    v = split_heads(v, num_key_value_heads, head_dim)

    q = rms_norm(q, q_norm_weight.view(1, 1, -1), norm_epsilon, output_dtype)
    k = rms_norm(k, k_norm_weight.view(1, 1, -1), norm_epsilon, output_dtype)
    q = apply_rope(q, output_dtype)
    k = apply_rope(k, output_dtype)
    k = k.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    v = v.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)

    out = causal_self_attention(q, k, v, output_dtype)
    out = linear_project(merge_heads(out), o_proj_weight, output_dtype)
    print(f"{label}={repr(out.tolist())}")


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
