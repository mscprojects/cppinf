# cppinf

A small C++/CMake scaffold for a future LLM inference project.

Right now it is intentionally minimal:

- `src/greeting.cpp`
- `src/greeting.h`
- `src/main.cpp`
- `tests/greeting_test.cpp`
- `tests/test.h`
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

That runs both the hello-world smoke test and the small test executable under
CTest.

## Run

```sh
./build/cppinf
```
