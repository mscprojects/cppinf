"""Offline oracle generator for causal_self_attention tests.

Run with:
    uv run --with torch==2.11.0 python tests/nn/causal_self_attention_oracle.py
"""

import torch


def print_case(name: str, output: torch.Tensor) -> None:
    print(f"{name}={repr(output.tolist())}")


def run_f32_full_sequence_case() -> None:
    q = torch.tensor(
        [[[0.5, -1.0], [1.25, 0.75], [-0.5, 2.0]], [[1.5, 0.5], [-1.0, 1.0], [0.25, -0.75]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[1.0, 0.0], [0.5, -1.5], [-0.75, 1.25]], [[0.25, 1.5], [-1.25, 0.5], [1.0, -0.5]]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[[2.0, -1.0], [0.5, 1.5], [-1.25, 0.75]], [[-0.5, 2.0], [1.75, -1.25], [0.25, 0.5]]],
        dtype=torch.float32,
    )

    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    scores = scores.masked_fill(
        torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    print_case("f32_full_sequence", torch.matmul(torch.softmax(scores, dim=-1), v))


def run_bf16_with_past_case() -> None:
    q = torch.tensor(
        [[[1.0, -0.5], [0.25, 1.5]], [[-1.25, 0.75], [1.5, -0.25]]],
        dtype=torch.bfloat16,
    )
    k = torch.tensor(
        [
            [[0.5, -1.0], [1.25, 0.5], [-0.75, 1.5], [1.0, -0.25]],
            [[-0.5, 1.0], [1.5, -1.25], [0.75, 0.25], [-1.0, 0.5]],
        ],
        dtype=torch.bfloat16,
    )
    v = torch.tensor(
        [
            [[1.0, 0.5], [-1.5, 0.75], [0.25, -0.5], [1.75, 1.25]],
            [[-0.75, 1.5], [1.25, -0.25], [0.5, 0.75], [-1.5, 1.0]],
        ],
        dtype=torch.bfloat16,
    )

    qf = q.float()
    kf = k.float()
    vf = v.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool)
    past = 2
    for query_index in range(q.shape[-2]):
        mask[query_index, : past + query_index + 1] = False
    scores = scores.masked_fill(mask, float("-inf"))
    output = torch.matmul(torch.softmax(scores, dim=-1), vf).to(torch.bfloat16).float()
    print_case("bf16_with_past", output)


if __name__ == "__main__":
    run_f32_full_sequence_case()
    run_bf16_with_past_case()
