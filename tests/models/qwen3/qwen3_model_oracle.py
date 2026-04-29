"""Offline oracle generator for qwen3_model tests.

Run with:
    uv run --with torch==2.11.0 python tests/models/qwen3/qwen3_model_oracle.py
"""

import torch


def round_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 100) / torch.tensor(100)


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
    x: torch.Tensor, output_dtype: torch.dtype, base: float = 1_000_000.0, position_offset: int = 0
) -> torch.Tensor:
    dim = x.shape[-1]
    positions = torch.arange(
        position_offset, position_offset + x.shape[-2], dtype=torch.float32
    )
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


def add(lhs: torch.Tensor, rhs: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(lhs + rhs, output_dtype)


def silu(x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(torch.nn.functional.silu(x), output_dtype)


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    seq, merged = x.shape
    return x.reshape(seq, heads, head_dim).permute(1, 0, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    heads, seq, dim = x.shape
    return x.permute(1, 0, 2).contiguous().reshape(seq, heads * dim)


def qwen_attention_module(
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
    position_offset: int = 0,
) -> torch.Tensor:
    q = linear_project(hidden_states, q_proj_weight, output_dtype)
    k = linear_project(hidden_states, k_proj_weight, output_dtype)
    v = linear_project(hidden_states, v_proj_weight, output_dtype)
    q = split_heads(q, num_attention_heads, head_dim)
    k = split_heads(k, num_key_value_heads, head_dim)
    v = split_heads(v, num_key_value_heads, head_dim)
    q = rms_norm(q, q_norm_weight.view(1, 1, -1), norm_epsilon, output_dtype)
    k = rms_norm(k, k_norm_weight.view(1, 1, -1), norm_epsilon, output_dtype)
    q = apply_rope(q, output_dtype, position_offset=position_offset)
    k = apply_rope(k, output_dtype, position_offset=position_offset)
    k = k.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    v = v.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (head_dim**-0.5)
    scores = scores.masked_fill(
        torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    out = quantize_to_dtype(torch.matmul(torch.softmax(scores, dim=-1), v.float()), output_dtype)
    return linear_project(merge_heads(out), o_proj_weight, output_dtype)


def qwen_mlp_module(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    gate = linear_project(hidden_states, gate_proj_weight, output_dtype)
    up = linear_project(hidden_states, up_proj_weight, output_dtype)
    return linear_project(quantize_to_dtype(silu(gate, output_dtype) * up, output_dtype), down_proj_weight,
                          output_dtype)


def qwen_decoder_block_module(
    hidden_states: torch.Tensor,
    input_ln_weight: torch.Tensor,
    post_ln_weight: torch.Tensor,
    params: dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    attn_input = rms_norm(hidden_states, input_ln_weight.view(1, -1), eps, torch.bfloat16)
    attn_out = qwen_attention_module(
        attn_input,
        params["q_proj"],
        params["q_norm"],
        params["k_proj"],
        params["k_norm"],
        params["v_proj"],
        params["o_proj"],
        2,
        1,
        4,
        eps,
        torch.bfloat16,
        0,
    )
    hidden_states = add(hidden_states, attn_out, torch.bfloat16)
    mlp_input = rms_norm(hidden_states, post_ln_weight.view(1, -1), eps, torch.bfloat16)
    mlp_out = qwen_mlp_module(mlp_input, params["gate"], params["up"], params["down"], torch.bfloat16)
    return add(hidden_states, mlp_out, torch.bfloat16)


if __name__ == "__main__":
    hidden_size = 6
    intermediate_size = 10
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 4
    vocab_size = 13
    num_hidden_layers = 2
    norm_epsilon = 1e-6

    torch.manual_seed(9004)
    embed = quantize_to_dtype(round_tensor(torch.empty(vocab_size, hidden_size).uniform_(-1.1, 1.1)), torch.bfloat16)
    layer_params = []
    for _ in range(num_hidden_layers):
        layer_params.append(
            {
                "input_ln": round_tensor(
                    torch.empty(hidden_size).uniform_(0.5, 1.5)
                ).to(torch.bfloat16).float(),
                "post_ln": round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)).to(torch.bfloat16).float(),
                "q_proj": round_tensor(
                    torch.empty(num_attention_heads * head_dim, hidden_size).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16).float(),
                "q_norm": round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5)).to(torch.bfloat16).float(),
                "k_proj": round_tensor(
                    torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16).float(),
                "k_norm": round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5)).to(torch.bfloat16).float(),
                "v_proj": round_tensor(
                    torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16).float(),
                "o_proj": round_tensor(
                    torch.empty(hidden_size, num_attention_heads * head_dim).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16).float(),
                "gate": round_tensor(
                    torch.empty(intermediate_size, hidden_size).uniform_(-1.1, 1.1)
                ).to(torch.bfloat16).float(),
                "up": round_tensor(
                    torch.empty(intermediate_size, hidden_size).uniform_(-1.1, 1.1)
                ).to(torch.bfloat16).float(),
                "down": round_tensor(
                    torch.empty(hidden_size, intermediate_size).uniform_(-1.1, 1.1)
                ).to(torch.bfloat16).float(),
            }
        )
    final_norm = round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)).to(torch.bfloat16).float()

    token_ids = torch.tensor([1, 5, 3, 2], dtype=torch.int64)
    hidden_states = embed[token_ids]
    for params in layer_params:
        hidden_states = qwen_decoder_block_module(
            hidden_states, params["input_ln"], params["post_ln"], params, norm_epsilon
        )
    hidden_states = rms_norm(hidden_states, final_norm.view(1, -1), norm_epsilon, torch.bfloat16)
    logits = linear_project(hidden_states, embed, torch.bfloat16)
    print(f"bf16_tiny_model={repr(logits.tolist())}")
