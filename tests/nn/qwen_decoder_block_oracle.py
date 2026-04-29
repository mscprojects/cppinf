"""Offline oracle generator for qwen_decoder_block tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/qwen_decoder_block_oracle.py
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
    x: torch.Tensor, output_dtype: torch.dtype, position_offset: int = 2, base: float = 1_000_000.0
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


def silu(x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(torch.nn.functional.silu(x), output_dtype)


def add(lhs: torch.Tensor, rhs: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(lhs + rhs, output_dtype)


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    seq, merged = x.shape
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


def qwen_attention(
    hidden_states: torch.Tensor,
    q_proj: torch.Tensor,
    q_norm: torch.Tensor,
    k_proj: torch.Tensor,
    k_norm: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    norm_epsilon: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    q = linear_project(hidden_states, q_proj, output_dtype)
    k = linear_project(hidden_states, k_proj, output_dtype)
    v = linear_project(hidden_states, v_proj, output_dtype)
    q = split_heads(q, num_attention_heads, head_dim)
    k = split_heads(k, num_key_value_heads, head_dim)
    v = split_heads(v, num_key_value_heads, head_dim)
    q = rms_norm(q, q_norm.view(1, 1, -1), norm_epsilon, output_dtype)
    k = rms_norm(k, k_norm.view(1, 1, -1), norm_epsilon, output_dtype)
    q = apply_rope(q, output_dtype)
    k = apply_rope(k, output_dtype)
    k = k.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    v = v.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    out = causal_self_attention(q, k, v, output_dtype)
    return linear_project(merge_heads(out), o_proj, output_dtype)


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
    attn_input = rms_norm(hidden_states, input_ln.view(1, -1), norm_epsilon, output_dtype)
    attn_out = qwen_attention(attn_input, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj, num_attention_heads,
                              num_key_value_heads, head_dim, norm_epsilon, output_dtype)
    residual = add(hidden_states, attn_out, output_dtype)
    mlp_input = rms_norm(residual, post_ln.view(1, -1), norm_epsilon, output_dtype)
    gate = linear_project(mlp_input, gate_proj, output_dtype)
    up = linear_project(mlp_input, up_proj, output_dtype)
    mlp_out = linear_project(quantize_to_dtype(silu(gate, output_dtype) * up, output_dtype), down_proj, output_dtype)
    out = add(residual, mlp_out, output_dtype)
    print(f"{label}={repr(out.tolist())}")


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
