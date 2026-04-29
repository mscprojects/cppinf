"""Offline oracle generator for rope tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/rope_oracle.py
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(x: torch.Tensor, base: float = 1_000_000.0, position_offset: int = 2) -> torch.Tensor:
    dim = x.shape[-1]
    positions = torch.arange(position_offset, position_offset + x.shape[-2], dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    return x * cos + rotate_half(x) * sin


if __name__ == "__main__":
    x = torch.tensor(
        [
            [[1.0, -0.5, 0.25, 1.5], [-1.25, 0.75, 1.0, -0.25], [0.5, 1.25, -0.75, 0.5]],
            [[-0.5, 1.0, 1.5, -1.25], [0.75, -1.5, 0.5, 1.0], [1.25, 0.25, -1.0, 0.75]],
        ],
        dtype=torch.bfloat16,
    )
    print(f"bf16_position_offset_2={repr(apply_rope(x.float()).to(torch.bfloat16).float().tolist())}")
