cmake_minimum_required(VERSION 3.24)

project(cppNMF VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)

option(CPPNMF_BUILD_TESTS "Build cppNMF tests" ON)
option(CPPNMF_BUILD_CLI "Build the cppNMF command line application" ON)
option(CPPNMF_BUILD_BENCHMARKS "Build cppNMF benchmarks" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/eigen/CMakeLists.txt")
  FetchContent_Declare(
    eigen
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/eigen")
else()
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
endif()

set(KISSFFT_DATATYPE double CACHE STRING "" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/kissfft/CMakeLists.txt")
  FetchContent_Declare(
    kissfft
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/kissfft")
else()
  FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG 131.2.0
    GIT_SHALLOW TRUE)
endif()

FetchContent_MakeAvailable(eigen kissfft)

add_library(cppnmf_core
  src/dnmf.cpp
  src/features.cpp
  src/io.cpp
  src/pipeline.cpp
  src/reconstruction.cpp
  src/signal.cpp
  src/stft.cpp)
add_library(cppnmf::core ALIAS cppnmf_core)

target_include_directories(cppnmf_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cppnmf_core
  PUBLIC Eigen3::Eigen
  PRIVATE kissfft::kissfft-double)
target_compile_features(cppnmf_core PUBLIC cxx_std_17)

if(CPPNMF_BUILD_CLI)
  add_executable(cppnmf_cli apps/cppnmf_cli.cpp)
  target_link_libraries(cppnmf_cli PRIVATE cppnmf::core)
endif()

if(CPPNMF_BUILD_BENCHMARKS)
  add_executable(cppnmf_benchmark benchmarks/cppnmf_benchmark.cpp)
  target_link_libraries(cppnmf_benchmark PRIVATE cppnmf::core)
endif()

if(CPPNMF_BUILD_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/googletest/CMakeLists.txt")
    FetchContent_Declare(
      googletest
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/googletest")
  else()
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.18.0
      GIT_SHALLOW TRUE)
  endif()
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(cppnmf_tests
    tests/dnmf_tests.cpp
    tests/features_tests.cpp
    tests/reconstruction_tests.cpp
    tests/signal_tests.cpp
    tests/stft_tests.cpp)
  target_link_libraries(cppnmf_tests PRIVATE cppnmf::core GTest::gtest_main)
  target_compile_definitions(cppnmf_tests PRIVATE
    CPPNMF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/validation/public_golden")
  gtest_discover_tests(cppnmf_tests)
endif()
