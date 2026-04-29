"""Offline oracle generator for HfTokenizer tests against the real Qwen3 tokenizer.

Run with:
    uv run --with transformers==4.52.3 python tests/tokenizers/hf/hf_tokenizer_oracle.py \
        --model-dir /home/q618175/Sources/models/Qwen3-0.6B-Base
"""

import argparse

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False, use_fast=True)
    examples = [
        "Hello world",
        "Hello  world\n",
        "The quick brown fox jumps over 13 lazy dogs.",
        "<|im_start|>user\nHello<|im_end|>\n",
    ]

    for text in examples:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"text={text!r}")
        print(f"ids={token_ids!r}")
        print(f"decoded={tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)!r}")
        print("---")
