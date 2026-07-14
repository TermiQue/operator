#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MACA_PATH="${MACA_PATH:-/opt/maca-20260318}"

export MACA_PATH
export LD_LIBRARY_PATH="$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/lib:${LD_LIBRARY_PATH:-}"

"$PYTHON_BIN" "$ROOT_DIR/standalone/fused_moe_i8_tn/python/test_fused_moe_i8_tn_pybind.py" "$@"
