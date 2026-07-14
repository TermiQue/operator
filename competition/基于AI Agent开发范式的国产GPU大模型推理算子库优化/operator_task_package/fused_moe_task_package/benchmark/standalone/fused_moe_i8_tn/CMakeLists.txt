cmake_minimum_required(VERSION 3.20)

project(fused_moe_i8_tn LANGUAGES NONE)

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

set(EXAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC "${EXAMPLE_ROOT}/src/fused_moe_i8_tn_example.cpp")
set(BIN "${CMAKE_CURRENT_BINARY_DIR}/fused_moe_i8_tn_example")

add_custom_command(
  OUTPUT "${BIN}"
  COMMAND "${MXCC}"
          -std=c++17
          -xmaca
          -I"${EXAMPLE_ROOT}/src"
          -I"${MACA_PATH}/include"
          "${SRC}"
          -L"${MACA_PATH}/lib"
          -lmcruntime
          -o "${BIN}"
  DEPENDS "${SRC}"
  WORKING_DIRECTORY "${EXAMPLE_ROOT}"
  VERBATIM)

add_custom_target(build_fused_moe_i8_tn_example ALL DEPENDS "${BIN}")

add_custom_target(
  run
  COMMAND "${BIN}"
  DEPENDS "${BIN}"
  USES_TERMINAL)
