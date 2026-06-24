#!/usr/bin/env bash
# Sweeps every kernel (0=cuBLAS reference, 1-7=custom) across a range
# of matrix sizes and prints a GFLOPS table. Run this after building:
#
#   cmake -B build && cmake --build build --parallel
#   ./scripts/run_benchmark.sh
#
# Tip: redirect to a file to save results for your README/resume:
#   ./scripts/run_benchmark.sh | tee benchmark_results.txt

set -euo pipefail

BIN="./build/sgemm"
SIZES=(128 256 512 1024 2048 4096)
KERNELS=(0 1 2 3 4 5 6 7)
ITERS=20

if [ ! -x "$BIN" ]; then
  echo "Binary not found at $BIN — build the project first:"
  echo "  cmake -B build && cmake --build build --parallel"
  exit 1
fi

for size in "${SIZES[@]}"; do
  echo "=== Matrix size: ${size}x${size}x${size} ==="
  for k in "${KERNELS[@]}"; do
    "$BIN" "$k" "$size" "$ITERS" | tail -n 1
  done
  echo ""
done
