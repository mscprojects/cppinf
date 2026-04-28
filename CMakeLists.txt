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
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

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
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
)

FetchContent_MakeAvailable(fmt googletest nlohmann_json spdlog)

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
    src/tensors/dtype.cpp
    src/tensors/dtype.h
    src/tensors/shape.cpp
  src/tensors/shape.h
  src/tensors/tensor_info.cpp
  src/tensors/tensor_info.h
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
target_link_libraries(cppinf PRIVATE fmt::fmt nlohmann_json::nlohmann_json spdlog::spdlog)

if(BUILD_TESTING)
    add_executable(cppinf_tests
        ${CPPINF_SOURCES}
        tests/cli/cli_app_test.cpp
        tests/files/safetensors_file_test.cpp
        tests/loaders/hf/hf_config_test.cpp
        tests/loaders/hf/hf_model_files_test.cpp
        tests/loaders/hf/hf_model_summary_test.cpp
        tests/tensors/tensor_test.cpp
    )
  target_include_directories(cppinf_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src
  )
  target_link_libraries(cppinf_tests PRIVATE fmt::fmt GTest::gtest_main nlohmann_json::nlohmann_json)

  gtest_discover_tests(cppinf_tests)
endif()
