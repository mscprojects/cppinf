cmake_minimum_required(VERSION 3.20)

project(cppinf LANGUAGES CXX)

include(CTest)
include(FetchContent)
include(GoogleTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DNNL_CPU_RUNTIME OMP CACHE STRING "" FORCE)
set(DNNL_ENABLE_PRIMITIVE "BINARY;ELTWISE;MATMUL;REDUCTION;REORDER;SOFTMAX" CACHE STRING "" FORCE)
set(DNNL_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.4
)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_Declare(
  oneDNN
  GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN.git
  GIT_TAG v3.11
)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
)

FetchContent_MakeAvailable(CLI11 fmt googletest nlohmann_json oneDNN spdlog)
find_package(OpenMP REQUIRED)

set(CPPINF_SOURCES
    src/cli/cli_app.cpp
    src/cli/cli_app.h
    src/files/safetensors_file.cpp
    src/files/safetensors_file.h
    src/loaders/hf/hf_config.cpp
    src/loaders/hf/hf_config.h
     src/loaders/hf/hf_model_files.cpp
     src/loaders/hf/hf_model_files.h
     src/loaders/hf/hf_model_summary.cpp
     src/loaders/hf/hf_model_summary.h
     src/models/qwen3/qwen3_model.cpp
     src/models/qwen3/qwen3_model.h
     src/nn/causal_self_attention.cpp
     src/nn/causal_self_attention.h
     src/nn/qwen_decoder_block.cpp
     src/nn/qwen_decoder_block.h
     src/nn/qwen_attention.cpp
     src/nn/qwen_attention.h
     src/nn/qwen_mlp.cpp
     src/nn/qwen_mlp.h
     src/nn/rope.cpp
     src/nn/rope.h
     src/ops/elementwise_ops.cpp
     src/ops/elementwise_ops.h
    src/ops/matmul.cpp
    src/ops/matmul.h
    src/ops/nn_ops.cpp
    src/ops/nn_ops.h
    src/ops/one_dnn_utils.cpp
    src/ops/one_dnn_utils.h
    src/ops/op_utils.h
    src/ops/tensor_ops.cpp
    src/ops/tensor_ops.h
    src/tensors/dtype.cpp
    src/tensors/dtype.h
    src/tensors/bfloat16.cpp
    src/tensors/bfloat16.h
    src/tensors/shape.cpp
    src/tensors/shape.h
    src/tensors/tensor.cpp
    src/tensors/tensor.h
    src/tensors/tensor_info.cpp
    src/tensors/tensor_info.h
    src/tokenizers/hf/hf_tokenizer.cpp
    src/tokenizers/hf/hf_tokenizer.h
    src/tensors/tensor_view.cpp
    src/tensors/tensor_view.h
)

add_executable(cppinf
    ${CPPINF_SOURCES}
    src/main.cpp
)
target_include_directories(cppinf PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(cppinf PRIVATE CLI11::CLI11 dnnl fmt::fmt nlohmann_json::nlohmann_json
                                     OpenMP::OpenMP_CXX spdlog::spdlog)

if(BUILD_TESTING)
    add_executable(cppinf_tests
        ${CPPINF_SOURCES}
        tests/cli/cli_app_test.cpp
        tests/files/safetensors_file_test.cpp
         tests/loaders/hf/hf_config_test.cpp
         tests/loaders/hf/hf_model_files_test.cpp
         tests/loaders/hf/hf_model_summary_test.cpp
         tests/models/qwen3/qwen3_model_test.cpp
         tests/models/qwen3/qwen3_real_model_test.cpp
        tests/nn/causal_self_attention_test.cpp
        tests/nn/qwen_decoder_block_test.cpp
        tests/nn/qwen_attention_test.cpp
        tests/nn/qwen_mlp_test.cpp
        tests/nn/rope_test.cpp
        tests/tokenizers/hf/hf_tokenizer_test.cpp
        tests/ops/ops_test.cpp
        tests/tensors/tensor_test.cpp
    )
    target_include_directories(cppinf_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests
    )
    target_link_libraries(cppinf_tests PRIVATE CLI11::CLI11 dnnl fmt::fmt GTest::gtest_main
                                               nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

    gtest_discover_tests(cppinf_tests)
endif()
