cmake_minimum_required(VERSION 3.20)

project(cppinf LANGUAGES CXX)

include(CTest)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
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
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
)

FetchContent_MakeAvailable(fmt spdlog)

add_executable(cppinf
  src/greeting.cpp
  src/greeting.h
  src/main.cpp
)
target_link_libraries(cppinf PRIVATE fmt::fmt spdlog::spdlog)

if(BUILD_TESTING)
  add_executable(cppinf_tests
    src/greeting.cpp
    src/greeting.h
    tests/greeting_test.cpp
    tests/test.h
  )
  target_include_directories(cppinf_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tests
  )
  target_link_libraries(cppinf_tests PRIVATE fmt::fmt)

  add_test(NAME cppinf_hello COMMAND cppinf)
  add_test(NAME cppinf_tests COMMAND cppinf_tests)
  set_tests_properties(cppinf_hello PROPERTIES
    PASS_REGULAR_EXPRESSION "Hello, cppinf!"
  )
endif()
