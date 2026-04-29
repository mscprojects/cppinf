"""Offline oracle generator for qwen_mlp tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/qwen_mlp_oracle.py
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


def linear_project(x: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(x @ weight.transpose(0, 1), output_dtype)


def silu(x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(torch.nn.functional.silu(x), output_dtype)


def mul(lhs: torch.Tensor, rhs: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return quantize_to_dtype(lhs * rhs, output_dtype)


def run_case(
    label: str,
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> None:
    gate = linear_project(hidden_states, gate_proj_weight, output_dtype)
    up = linear_project(hidden_states, up_proj_weight, output_dtype)
    out = linear_project(mul(silu(gate, output_dtype), up, output_dtype), down_proj_weight, output_dtype)
    print(f"{label}={repr(out.tolist())}")


if __name__ == "__main__":
    torch.manual_seed(9002)
    hidden_states_f32 = round_tensor(torch.empty(3, 6).uniform_(-1.3, 1.3))
    gate_proj_weight_f32 = round_tensor(torch.empty(10, 6).uniform_(-1.3, 1.3))
    up_proj_weight_f32 = round_tensor(torch.empty(10, 6).uniform_(-1.3, 1.3))
    down_proj_weight_f32 = round_tensor(torch.empty(6, 10).uniform_(-1.3, 1.3))

    run_case(
        "f32_basic",
        hidden_states_f32,
        gate_proj_weight_f32,
        up_proj_weight_f32,
        down_proj_weight_f32,
        torch.float32,
    )
    run_case(
        "bf16_basic",
        quantize_to_dtype(hidden_states_f32, torch.bfloat16),
        quantize_to_dtype(gate_proj_weight_f32, torch.bfloat16),
        quantize_to_dtype(up_proj_weight_f32, torch.bfloat16),
        quantize_to_dtype(down_proj_weight_f32, torch.bfloat16),
        torch.bfloat16,
    )
