"""Offline oracle generator for qwen_mlp tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/qwen_mlp_oracle.py
"""

import torch


def round_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 100) / 100


if __name__ == "__main__":
    torch.manual_seed(9002)
    hidden_states = round_tensor(torch.empty(3, 6).uniform_(-1.3, 1.3))
    gate_proj_weight = round_tensor(torch.empty(10, 6).uniform_(-1.3, 1.3))
    up_proj_weight = round_tensor(torch.empty(10, 6).uniform_(-1.3, 1.3))
    down_proj_weight = round_tensor(torch.empty(6, 10).uniform_(-1.3, 1.3))

    gate = hidden_states @ gate_proj_weight.transpose(0, 1)
    up = hidden_states @ up_proj_weight.transpose(0, 1)
    out = (torch.nn.functional.silu(gate) * up) @ down_proj_weight.transpose(0, 1)
    print(f"f32_basic={repr(out.tolist())}")
