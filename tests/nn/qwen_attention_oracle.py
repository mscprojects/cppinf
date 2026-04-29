"""Offline oracle generator for qwen_attention tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/qwen_attention_oracle.py
"""

import torch


def round_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 100) / 100


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(x: torch.Tensor, position_offset: int = 1, base: float = 1_000_000.0) -> torch.Tensor:
    dim = x.shape[-1]
    positions = torch.arange(position_offset, position_offset + x.shape[-2], dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    seq, merged = x.shape
    assert merged == heads * head_dim
    return x.reshape(seq, heads, head_dim).permute(1, 0, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    heads, seq, dim = x.shape
    return x.permute(1, 0, 2).contiguous().reshape(seq, heads * dim)


if __name__ == "__main__":
    hidden_size = 6
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 4
    norm_epsilon = 1e-6

    torch.manual_seed(9001)
    hidden_states = round_tensor(torch.empty(3, hidden_size).uniform_(-1.4, 1.4))
    q_proj_weight = round_tensor(torch.empty(num_attention_heads * head_dim, hidden_size).uniform_(-1.4, 1.4))
    q_norm_weight = round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5))
    k_proj_weight = round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.4, 1.4))
    k_norm_weight = round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5))
    v_proj_weight = round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.4, 1.4))
    o_proj_weight = round_tensor(torch.empty(hidden_size, num_attention_heads * head_dim).uniform_(-1.4, 1.4))

    q = hidden_states @ q_proj_weight.transpose(0, 1)
    k = hidden_states @ k_proj_weight.transpose(0, 1)
    v = hidden_states @ v_proj_weight.transpose(0, 1)

    q = split_heads(q, num_attention_heads, head_dim)
    k = split_heads(k, num_key_value_heads, head_dim)
    v = split_heads(v, num_key_value_heads, head_dim)

    q = rms_norm(q, q_norm_weight.view(1, 1, -1), norm_epsilon)
    k = rms_norm(k, k_norm_weight.view(1, 1, -1), norm_epsilon)
    q = apply_rope(q)
    k = apply_rope(k)
    k = k.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    v = v.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)

    scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
    scores = scores.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1), float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    out = merge_heads(out) @ o_proj_weight.transpose(0, 1)

    print(f"f32_explicit_head_dim={repr(out.tolist())}")
