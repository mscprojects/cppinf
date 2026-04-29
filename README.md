# cppinf

A small C++/CMake scaffold for a future LLM inference project.

Right now it is intentionally minimal:

- `src/files/`
- `src/loaders/hf/`
- `src/ops/`
- `src/main.cpp`
- `src/tensors/`
- `tests/files/`
- `tests/loaders/hf/`
- `tests/ops/`
- `tests/tensors/`
- `CMakeLists.txt`
- `.clang-format`
- `justfile`
- `.gitignore`

The project fetches `CLI11`, `fmt`, `spdlog`, and `GoogleTest` during CMake configure.

## Build

```sh
just build
```

or:

```sh
cmake -S . -B build
cmake --build build
```

## Test

```sh
just test
```

That runs the GoogleTest suite through CTest.

`src/files/` contains file-oriented loaders such as the weights-only
`SafetensorsFile`.

`src/loaders/hf/` contains Hugging Face-specific directory and config loaders.

`src/ops/` contains the first CPU reference tensor operations.

`src/tensors/` contains the tensor scaffolding: dtypes, shapes, tensor info,
tensor views, an owning tensor, and readable `to_string(...)` helpers.

`tests/` mirrors the `src/` layout.

## Lint

```sh
just lint
```

That formats all C++ source and header files in `src/` and `tests/` with
`clang-format`.

## Run

```sh
just run
```

or:

```sh
./build/cppinf
```

To inspect a Hugging Face model directory:

```sh
just run inspect hf ~/Sources/models/Qwen3-0.6B/
```

To show every tensor in the weights file:

```sh
just run inspect hf ~/Sources/models/Qwen3-0.6B/ --all
```

To limit the tensor listing:

```sh
just run inspect hf ~/Sources/models/Qwen3-0.6B/ --limit 32
```
