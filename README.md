# cppinf

`cppinf` is a small C++20 LLM inference project. It currently focuses on CPU reference inference for Qwen3-style
Hugging Face checkpoints, with safetensors loading, a minimal Hugging Face tokenizer, Qwen decoder components, cached
autoregressive generation, and GoogleTest coverage backed by PyTorch oracle scripts.

The implementation is intentionally compact and readable. It is useful as an inference learning project, not a
production runtime.

## Features

- Loads unsharded Hugging Face model directories with `config.json`, `model.safetensors`, `tokenizer.json`, and
  `tokenizer_config.json`.
- Supports Qwen3 dense decoder models with tied embeddings, grouped-query causal attention, RoPE, q/k RMSNorm, gated
  MLPs, BF16/F32 tensor paths, and K/V caching.
- Provides tensor, shape, dtype, tensor-view, safetensors, tokenizer, and model-summary utilities.
- Uses oneDNN for selected CPU tensor operations.
- Tests core math and model pieces with GoogleTest, including oracle values generated from Python/PyTorch scripts.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler, the `justfile` uses `clang` and `clang++`
- `just`
- `clang-format` and `clang-tidy`
- OpenMP

CMake fetches CLI11, fmt, GoogleTest, nlohmann/json, oneDNN, and spdlog during configure.

## Build

```sh
just build
```

or:

```sh
cmake -S . -B build -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++
cmake --build build --parallel
```

## Run

```sh
just run
```

Inspect a Hugging Face model directory:

```sh
just run inspect hf ~/Sources/models/Qwen3-0.6B --limit 32
```

Show every tensor in the weights file:

```sh
just run inspect hf ~/Sources/models/Qwen3-0.6B --all
```

Generate text:

```sh
just run run hf ~/Sources/models/Qwen3-0.6B --prompt "Hello" --max-new-tokens 32 --temperature 0
```

`--temperature 0` selects deterministic greedy tokens. Higher non-negative values sample from the last-token logits.

## Test And Format

```sh
just test
just format
```

`just commit` runs formatting and the full test suite.

Some real-checkpoint tests require `CPPINF_QWEN3_REAL_MODEL_DIR` to point at a local Qwen3 Hugging Face model directory.

## Layout

- `src/cli/` contains the command-line app.
- `src/files/` contains safetensors loading.
- `src/loaders/hf/` contains Hugging Face model-file and config loading.
- `src/models/qwen3/` contains the high-level Qwen3 wrapper and model notes.
- `src/nn/` contains Qwen decoder, attention, MLP, RoPE, and cache components.
- `src/ops/` contains tensor operations and oneDNN helpers.
- `src/tensors/` contains dtype, shape, owning tensor, and tensor-view primitives.
- `src/tokenizers/hf/` contains the tokenizer loader and encoder/decoder.
- `tests/` mirrors the source layout.

## Current Limits

Sharded safetensors checkpoints are not supported yet. The current model path targets Qwen3 checkpoints with tied word
embeddings and the dense architecture covered by the tests.
