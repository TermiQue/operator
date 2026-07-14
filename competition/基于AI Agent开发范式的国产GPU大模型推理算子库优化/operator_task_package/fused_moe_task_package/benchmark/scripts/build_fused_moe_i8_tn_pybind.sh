#!/usr/bin/env bash

set -euo pipefail

MACA_PATH="${MACA_PATH:-/opt/maca}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MACA_PATH
export LD_LIBRARY_PATH="$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/lib:${LD_LIBRARY_PATH:-}"

BUILD_DIR="$ROOT_DIR/standalone/fused_moe_i8_tn/build"
SO="$BUILD_DIR/fused_moe_i8_tn_pybind.so"
SRC="$ROOT_DIR/standalone/fused_moe_i8_tn/src/fused_moe_i8_tn_pybind.cpp"

mkdir -p "$BUILD_DIR"

PYTHON_INCLUDE="$("${PYTHON_BIN}" -c "import sysconfig; print(sysconfig.get_path('include'))")"
PYTHON_LIB="$("${PYTHON_BIN}" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
PYTHON_LDLIB="$("${PYTHON_BIN}" -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")"

PYBIND11_INCLUDE="$ROOT_DIR/third_party/pybind11/include"

"$MACA_PATH/mxgpu_llvm/bin/mxcc" \
    -std=c++17 \
    -O2 \
    -c \
    -xmaca \
    -fPIC \
    -I"$ROOT_DIR/include" \
    -I"$MACA_PATH/include" \
    -I"$PYTHON_INCLUDE" \
    -I"$PYBIND11_INCLUDE" \
    "$SRC" \
    -o "$BUILD_DIR/fused_moe_i8_tn_pybind.o"


g++ \
    -std=c++17 \
    -O2 \
    -shared \
    -fPIC \
    "$BUILD_DIR/fused_moe_i8_tn_pybind.o" \
    -L"$MACA_PATH/lib" \
    -L"/opt/conda/lib" \
    -Wl,-rpath,"/opt/conda/lib" \
    -l:"libpython3.12.so" \
    -lmcruntime \
    -lmccompiler \
    -o "$SO"

echo "[SUCCESS] $SO"
