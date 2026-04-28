# cppinf

A small C++/CMake scaffold for a future LLM inference project.

Right now it is intentionally minimal:

- `src/files/`
- `src/loaders/hf/`
- `src/main.cpp`
- `src/tensors/`
- `tests/files/`
- `tests/loaders/hf/`
- `tests/tensors/`
- `CMakeLists.txt`
- `.clang-format`
- `justfile`
- `.gitignore`

The project fetches `fmt`, `spdlog`, and `GoogleTest` during CMake configure.

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

`src/tensors/` contains a small metadata-first tensor scaffold: dtypes, shapes,
tensor info, tensor views, and readable `to_string(...)` helpers.

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
just run inspect hf ~/Sources/models/Qwen3-0.6B-Base/
```
