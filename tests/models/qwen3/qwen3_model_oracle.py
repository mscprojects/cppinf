"""Offline oracle generator for qwen3_model tests against Hugging Face Qwen3ForCausalLM.

Run with:
    uv run --no-project --with transformers==4.52.3 --with torch==2.7.1 \
        --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match python tests/models/qwen3/qwen3_model_oracle.py
"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


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
    vocab_size: int,
    num_hidden_layers: int,
    norm_epsilon: float,
) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
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


def copy_weight(parameter: torch.nn.Parameter, weight: torch.Tensor, dtype: torch.dtype) -> None:
    parameter.copy_(weight.to(dtype))


if __name__ == "__main__":
    hidden_size = 6
    intermediate_size = 10
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 4
    vocab_size = 13
    num_hidden_layers = 2
    norm_epsilon = 1e-6
    output_dtype = torch.bfloat16

    torch.manual_seed(9004)
    embed = quantize_to_dtype(round_tensor(torch.empty(vocab_size, hidden_size).uniform_(-1.1, 1.1)), output_dtype)
    layer_params = []
    for _ in range(num_hidden_layers):
        layer_params.append(
            {
                "input_ln": quantize_to_dtype(round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)), output_dtype),
                "post_ln": quantize_to_dtype(round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)), output_dtype),
                "q_proj": quantize_to_dtype(
                    round_tensor(torch.empty(num_attention_heads * head_dim, hidden_size).uniform_(-1.1, 1.1)),
                    output_dtype,
                ),
                "q_norm": quantize_to_dtype(round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5)), output_dtype),
                "k_proj": quantize_to_dtype(
                    round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.1, 1.1)),
                    output_dtype,
                ),
                "k_norm": quantize_to_dtype(round_tensor(torch.empty(head_dim).uniform_(0.5, 1.5)), output_dtype),
                "v_proj": quantize_to_dtype(
                    round_tensor(torch.empty(num_key_value_heads * head_dim, hidden_size).uniform_(-1.1, 1.1)),
                    output_dtype,
                ),
                "o_proj": quantize_to_dtype(
                    round_tensor(torch.empty(hidden_size, num_attention_heads * head_dim).uniform_(-1.1, 1.1)),
                    output_dtype,
                ),
                "gate": quantize_to_dtype(
                    round_tensor(torch.empty(intermediate_size, hidden_size).uniform_(-1.1, 1.1)), output_dtype
                ),
                "up": quantize_to_dtype(
                    round_tensor(torch.empty(intermediate_size, hidden_size).uniform_(-1.1, 1.1)), output_dtype
                ),
                "down": quantize_to_dtype(
                    round_tensor(torch.empty(hidden_size, intermediate_size).uniform_(-1.1, 1.1)), output_dtype
                ),
            }
        )
    final_norm = quantize_to_dtype(round_tensor(torch.empty(hidden_size).uniform_(0.5, 1.5)), output_dtype)

    config = make_config(
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        num_hidden_layers,
        norm_epsilon,
    )
    model = Qwen3ForCausalLM(config).eval().to(output_dtype)

    with torch.no_grad():
        copy_weight(model.model.embed_tokens.weight, embed, output_dtype)
        for layer, params in zip(model.model.layers, layer_params, strict=True):
            copy_weight(layer.input_layernorm.weight, params["input_ln"], output_dtype)
            copy_weight(layer.post_attention_layernorm.weight, params["post_ln"], output_dtype)
            copy_weight(layer.self_attn.q_proj.weight, params["q_proj"], output_dtype)
            copy_weight(layer.self_attn.q_norm.weight, params["q_norm"], output_dtype)
            copy_weight(layer.self_attn.k_proj.weight, params["k_proj"], output_dtype)
            copy_weight(layer.self_attn.k_norm.weight, params["k_norm"], output_dtype)
            copy_weight(layer.self_attn.v_proj.weight, params["v_proj"], output_dtype)
            copy_weight(layer.self_attn.o_proj.weight, params["o_proj"], output_dtype)
            copy_weight(layer.mlp.gate_proj.weight, params["gate"], output_dtype)
            copy_weight(layer.mlp.up_proj.weight, params["up"], output_dtype)
            copy_weight(layer.mlp.down_proj.weight, params["down"], output_dtype)
        copy_weight(model.model.norm.weight, final_norm, output_dtype)
        model.tie_weights()

        token_ids = torch.tensor([[1, 5, 3, 2]], dtype=torch.int64)
        logits = model(input_ids=token_ids, use_cache=False).logits

    print(f"bf16_tiny_model={repr(logits[0].float().tolist())}")
