build:
    cmake -S . -B build
    cmake --build build

run *args: build
    ./build/cppinf {{args}}

test: build
    ctest --test-dir build --output-on-failure

lint:
    find src tests -type f \( -name '*.cpp' -o -name '*.h' \) -print0 | xargs -0 clang-format -i
