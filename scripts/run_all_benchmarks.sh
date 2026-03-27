#!/bin/bash
# Run full benchmark sweep: 15 configs (5 S_kv × 3 B)
# Usage: bash scripts/run_all_benchmarks.sh

set -e

echo "================================================"
echo "Fused QKV INT8 Attention — Full Benchmark Sweep"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo "================================================"

# Step 1: Build kernels
echo ""
echo "[1/4] Building kernels..."
make clean && make all

# Step 2: Run correctness tests
echo ""
echo "[2/4] Running correctness tests..."
python tests/test_quantize.py
python tests/test_correctness_fp16.py
python tests/test_correctness_int8.py

# Step 3: Run benchmarks
echo ""
echo "[3/4] Running benchmarks..."
python bench/bench_fp16.py
python bench/bench_int8kv.py

# Step 4: NCU profiling (S_kv=4096, B=1 only)
echo ""
echo "[4/4] NCU profiling (S_kv=4096, B=1)..."
echo "NOTE: NCU profiling is slow. Skip with Ctrl+C if not needed."
make profile-baseline
make profile-int8kv

echo ""
echo "================================================"
echo "Done. Results in bench/results/, profiles in profiles/"
echo "================================================"
