"""Offline oracle generator for qwen3_model tests.

Run with:
    uv run --with torch==2.11.0 python tests/models/qwen3/qwen3_model_oracle.py
"""

import torch


def round_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 100) / torch.tensor(100)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(
    x: torch.Tensor, base: float = 1_000_000.0, position_offset: int = 0
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
    return x * cos + rotate_half(x) * sin


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


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
    position_offset: int = 0,
) -> torch.Tensor:
    q = hidden_states @ q_proj_weight.transpose(0, 1)
    k = hidden_states @ k_proj_weight.transpose(0, 1)
    v = hidden_states @ v_proj_weight.transpose(0, 1)
    q = split_heads(q, num_attention_heads, head_dim)
    k = split_heads(k, num_key_value_heads, head_dim)
    v = split_heads(v, num_key_value_heads, head_dim)
    q = rms_norm(q, q_norm_weight.view(1, 1, -1), norm_epsilon)
    k = rms_norm(k, k_norm_weight.view(1, 1, -1), norm_epsilon)
    q = apply_rope(q, position_offset=position_offset)
    k = apply_rope(k, position_offset=position_offset)
    k = k.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    v = v.repeat_interleave(num_attention_heads // num_key_value_heads, dim=0)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
    scores = scores.masked_fill(
        torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return merge_heads(out) @ o_proj_weight.transpose(0, 1)


def qwen_mlp_module(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    gate = hidden_states @ gate_proj_weight.transpose(0, 1)
    up = hidden_states @ up_proj_weight.transpose(0, 1)
    return (torch.nn.functional.silu(gate) * up) @ down_proj_weight.transpose(0, 1)


def qwen_decoder_block_module(
    hidden_states: torch.Tensor,
    input_ln_weight: torch.Tensor,
    post_ln_weight: torch.Tensor,
    params: dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    attn_input = rms_norm(hidden_states_f32, input_ln_weight.float().view(1, -1), eps)
    attn_out = qwen_attention_module(
        attn_input,
        params["q_proj"].float(),
        params["q_norm"].float(),
        params["k_proj"].float(),
        params["k_norm"].float(),
        params["v_proj"].float(),
        params["o_proj"].float(),
        2,
        1,
        4,
        eps,
        0,
    )
    hidden_states = hidden_states_f32 + attn_out
    mlp_input = rms_norm(hidden_states, post_ln_weight.float().view(1, -1), eps)
    mlp_out = qwen_mlp_module(
        mlp_input, params["gate"].float(), params["up"].float(), params["down"].float()
    )
    return (hidden_states + mlp_out).to(torch.bfloat16)


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
    embed = round_tensor(torch.empty(vocab_size, hidden_size).uniform_(-1.1, 1.1)).to(
        torch.bfloat16
    )
    layer_params = []
    for _ in range(num_hidden_layers):
        layer_params.append(
            {
                "input_ln": round_tensor(
                    torch.empty(hidden_size).uniform_(0.5, 1.5)
                ).to(torch.bfloat16),
                "post_ln": round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)).to(
                    torch.bfloat16
                ),
                "q_proj": round_tensor(
                    torch.empty(num_attention_heads * head_dim, hidden_size).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16),
                "q_norm": round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5)).to(
                    torch.bfloat16
                ),
                "k_proj": round_tensor(
                    torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16),
                "k_norm": round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5)).to(
                    torch.bfloat16
                ),
                "v_proj": round_tensor(
                    torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16),
                "o_proj": round_tensor(
                    torch.empty(hidden_size, num_attention_heads * head_dim).uniform_(
                        -1.1, 1.1
                    )
                ).to(torch.bfloat16),
                "gate": round_tensor(
                    torch.empty(intermediate_size, hidden_size).uniform_(-1.1, 1.1)
                ).to(torch.bfloat16),
                "up": round_tensor(
                    torch.empty(intermediate_size, hidden_size).uniform_(-1.1, 1.1)
                ).to(torch.bfloat16),
                "down": round_tensor(
                    torch.empty(hidden_size, intermediate_size).uniform_(-1.1, 1.1)
                ).to(torch.bfloat16),
            }
        )
    final_norm = round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)).to(
        torch.bfloat16
    )

    token_ids = torch.tensor([1, 5, 3, 2], dtype=torch.int64)
    hidden_states = embed[token_ids]
    for params in layer_params:
        hidden_states = qwen_decoder_block_module(
            hidden_states, params["input_ln"], params["post_ln"], params, norm_epsilon
        )
    hidden_states = rms_norm(
        hidden_states.float(), final_norm.float().view(1, -1), norm_epsilon
    ).to(torch.bfloat16)
    logits = (hidden_states.float() @ embed.float().transpose(0, 1)).to(torch.bfloat16)
    print(f"bf16_tiny_model={repr(logits.float().tolist())}")
