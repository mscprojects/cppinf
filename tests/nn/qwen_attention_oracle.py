"""Offline oracle generator for qwen_attention tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/qwen_attention_oracle.py
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(x: torch.Tensor, base: float = 1_000_000.0, position_offset: int = 0) -> torch.Tensor:
    dim = x.shape[-1]
    positions = torch.arange(position_offset, position_offset + x.shape[-2], dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    seq, merged = x.shape
    dim = merged // heads
    return x.reshape(seq, heads, dim).permute(1, 0, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    heads, seq, dim = x.shape
    return x.permute(1, 0, 2).contiguous().reshape(seq, heads * dim)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def run_f32_case() -> None:
    hidden = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = hidden // num_heads

    torch.manual_seed(123)
    hidden_states = torch.round(torch.empty(3, hidden).uniform_(-1.5, 1.5) * 100) / 100
    q_weight = torch.round(torch.empty(hidden, hidden).uniform_(-1.5, 1.5) * 100) / 100
    k_weight = torch.round(torch.empty(num_kv_heads * head_dim, hidden).uniform_(-1.5, 1.5) * 100) / 100
    v_weight = torch.round(torch.empty(num_kv_heads * head_dim, hidden).uniform_(-1.5, 1.5) * 100) / 100
    o_weight = torch.round(torch.empty(hidden, hidden).uniform_(-1.5, 1.5) * 100) / 100

    q = split_heads(hidden_states @ q_weight.transpose(0, 1), num_heads)
    k = split_heads(hidden_states @ k_weight.transpose(0, 1), num_kv_heads)
    v = split_heads(hidden_states @ v_weight.transpose(0, 1), num_kv_heads)
    q = apply_rope(q)
    k = apply_rope(k)
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=0)
    v = v.repeat_interleave(num_heads // num_kv_heads, dim=0)
    out = merge_heads(causal_attention(q, k, v)) @ o_weight.transpose(0, 1)
    print(f"f32_grouped_kv={repr(out.tolist())}")


def run_bf16_case() -> None:
    hidden = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = hidden // num_heads

    torch.manual_seed(321)
    hidden_states = (torch.round(torch.empty(2, hidden).uniform_(-1.25, 1.25) * 100) / 100).to(torch.bfloat16)
    q_weight = (torch.round(torch.empty(hidden, hidden).uniform_(-1.25, 1.25) * 100) / 100).to(torch.bfloat16)
    k_weight = (torch.round(torch.empty(num_kv_heads * head_dim, hidden).uniform_(-1.25, 1.25) * 100) / 100).to(
        torch.bfloat16
    )
    v_weight = (torch.round(torch.empty(num_kv_heads * head_dim, hidden).uniform_(-1.25, 1.25) * 100) / 100).to(
        torch.bfloat16
    )
    o_weight = (torch.round(torch.empty(hidden, hidden).uniform_(-1.25, 1.25) * 100) / 100).to(torch.bfloat16)

    q = split_heads(hidden_states.float() @ q_weight.float().transpose(0, 1), num_heads)
    k = split_heads(hidden_states.float() @ k_weight.float().transpose(0, 1), num_kv_heads)
    v = split_heads(hidden_states.float() @ v_weight.float().transpose(0, 1), num_kv_heads)
    q = apply_rope(q, position_offset=1)
    k = apply_rope(k, position_offset=1)
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=0)
    v = v.repeat_interleave(num_heads // num_kv_heads, dim=0)
    out = (merge_heads(causal_attention(q, k, v)) @ o_weight.float().transpose(0, 1)).to(torch.bfloat16).float()
    print(f"bf16_grouped_kv={repr(out.tolist())}")


if __name__ == "__main__":
    run_f32_case()
    run_bf16_case()
