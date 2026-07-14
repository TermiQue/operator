cmake_minimum_required(VERSION 3.20)

project(i8_tn_256x256x128_raw_arrays LANGUAGES NONE)

set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "Path to MACA SDK")
if(NOT MACA_PATH)
  set(MACA_PATH "/opt/maca")
endif()

find_program(MXCC
  NAMES mxcc
  PATHS "${MACA_PATH}/mxgpu_llvm/bin"
  NO_DEFAULT_PATH)

if(NOT MXCC)
  message(FATAL_ERROR "mxcc not found under ${MACA_PATH}/mxgpu_llvm/bin")
endif()

set(STANDALONE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC "${STANDALONE_ROOT}/src/test.cpp")
set(BIN "${CMAKE_CURRENT_BINARY_DIR}/i8_tn_256x256x128_raw_arrays_test")

add_custom_command(
  OUTPUT "${BIN}"
  COMMAND "${MXCC}"
          -std=c++17
          -xmaca
          -I"${STANDALONE_ROOT}/include"
          -I"${MACA_PATH}/include"
          "${SRC}"
          -L"${MACA_PATH}/lib"
          -lmcruntime
          -o "${BIN}"
  DEPENDS
          "${SRC}"
          "${STANDALONE_ROOT}/include/standalone_maca_kernel_utils.hpp"
          "${STANDALONE_ROOT}/include/gemm_i8_tn_256x256x128_raw_arrays.hpp"
  VERBATIM)

add_custom_target(build_i8_tn_256x256x128_raw_arrays_test ALL DEPENDS "${BIN}")

add_custom_target(
  run
  COMMAND "${BIN}"
  DEPENDS "${BIN}"
  USES_TERMINAL)
