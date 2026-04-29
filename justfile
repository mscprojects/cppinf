build:
    cmake -S . -B build -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D CPPINF_ENABLE_CLANG_TIDY=ON
    cmake --build build --parallel

run *args: build
    ./build/cppinf {{args}}

test: build
    ctest --test-dir build --output-on-failure -j

format:
    find src tests -type f \( -name '*.cpp' -o -name '*.h' \) -print0 | xargs -0 clang-format -i

commit: format test
