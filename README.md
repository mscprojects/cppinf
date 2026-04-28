# cppinf

A small C++/CMake scaffold for a future LLM inference project.

Right now it is intentionally minimal:

- `src/main.cpp`
- `CMakeLists.txt`
- `.gitignore`

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/cppinf
```
