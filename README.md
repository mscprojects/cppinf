# cppinf

A small C++/CMake scaffold for a future LLM inference project.

Right now it is intentionally minimal:

- `src/main.cpp`
- `src/tensors/`
- `tests/tensor_test.cpp`
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

That runs the hello-world smoke test plus the GoogleTest suite through CTest.

`src/tensors/` contains a small metadata-first tensor scaffold: dtypes, shapes,
tensor info, tensor views, and readable `to_string(...)` helpers.

## Lint

```sh
just lint
```

That formats all C++ source and header files in `src/` and `tests/` with
`clang-format`.

## Run

```sh
./build/cppinf
```
