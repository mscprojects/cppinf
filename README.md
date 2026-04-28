# cppinf

A small C++/CMake scaffold for a future LLM inference project.

Right now it is intentionally minimal:

- `src/main.cpp`
- `CMakeLists.txt`
- `justfile`
- `.gitignore`

The project fetches `fmt` and `spdlog` during CMake configure.

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

## Run

```sh
./build/cppinf
```
