"""Offline oracle generator for the real Qwen3-0.6B-Base checkpoint.

Run with:
    uv run --with torch==2.11.0 --with transformers==4.52.3 \
        python tests/models/qwen3/qwen3_real_model_oracle.py \
        --model-dir /home/q618175/Sources/models/Qwen3-0.6B-Base
"""

import argparse

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    token_ids = [151643, 42, 1024, 4096]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="eager",
    )
    model.eval()

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits

    last_token = logits[0, -1].float()
    topk_values, topk_indices = torch.topk(last_token, 8)

    print(f"token_ids={token_ids!r}")
    print(f"last_token_topk_ids={topk_indices.tolist()!r}")
    print(f"last_token_topk_values={topk_values.tolist()!r}")
    print(f"last_token_slice_0={last_token[:16].tolist()!r}")
    print(f"last_token_slice_1024={last_token[1024:1040].tolist()!r}")
    print(f"last_token_slice_tail={last_token[-16:].tolist()!r}")
