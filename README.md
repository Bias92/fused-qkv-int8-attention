# Fused Quantized-KV Attention Kernel

Custom CUDA kernel that fuses INT8 KV-cache dequantization with FlashAttention-style tiled decode attention on A100 GPUs.

## Claim

> Storing KV-cache in per-token symmetric INT8 and performing fused dequantization inside the attention kernel reduces HBM traffic and improves latency for decode attention compared to the FP16 baseline.

## Why

KV-cache is the primary memory bottleneck in LLM inference. Decode attention (`S_q=1`) is memory-bound, so storing KV in INT8 halves the HBM→SM transfer volume, directly reducing latency. Additionally, the ~48% reduction in shared memory usage enables double buffering, which improves compute-memory overlap.

## Key Idea

```
HBM (INT8 KV) → Shared Memory (INT8) → Registers (INT8→FP16 dequant + scale) → FP16 Tensor Core mma
```

- **Storage:** K, V quantized to per-token symmetric INT8 and stored in HBM
- **Compute:** Register-level dequantization inside the kernel → FP16 Tensor Core mma (`hmma.m16n8k16`)
- **Benefit:** Reduced HBM read + smaller smem footprint → double buffering possible (not possible with FP16)

## Target Hardware

| Spec | Value |
|------|-------|
| GPU | A100 SXM 80GB |
| Compute Capability | 8.0 |
| HBM2e Bandwidth | 2,039 GB/s |
| Shared Memory / SM | Up to 164 KB |
| FP16 Tensor Core | 312 TFLOPS |

## Experiment Setup

| Fixed | Value |
|-------|-------|
| Direction | Forward only (inference) |
| Scenario | Decode (`S_q = 1`) |
| Head dim | 128 |
| Num heads | 32 |
| Quantization | Per-token symmetric INT8 |

| Sweep | Values |
|-------|--------|
| S_kv | 1024, 2048, 4096, 8192, 16384 |
| Batch | 1, 4, 8 |

## Results

> TODO: Update after experiments

### Correctness

| Config | cos_sim | max_diff | rel_l2_err | PASS/FAIL |
|--------|---------|----------|------------|-----------|
| ... | ... | ... | ... | ... |

### Latency & HBM Traffic

> TODO: Add graphs

### NCU Analysis

> TODO: Add profiling results

## Project Structure

```
fused-qkv-int8-attention/
├── kernels/
│   ├── flash_attn_fp16.cu           # FP16 baseline decode attention
│   └── flash_attn_int8kv.cu         # INT8 KV fused decode attention
├── csrc/
│   ├── attention.h                   # Common headers, macros
│   └── mma_utils.h                   # Tensor Core mma wrappers
├── utils/
│   ├── quantize.py                   # Per-token INT8 quantization
│   └── benchmark.py                  # Benchmark harness
├── tests/
│   ├── test_correctness_fp16.py      # FP16 kernel vs PyTorch SDPA
│   └── test_correctness_int8.py      # INT8 KV kernel vs PyTorch SDPA
├── bench/
│   ├── bench_fp16.py                 # FP16 baseline benchmark
│   ├── bench_int8kv.py               # INT8 KV benchmark
│   └── results/                      # CSV + plots
├── profiles/                         # NCU report files (.ncu-rep)
├── scripts/
│   └── run_all_benchmarks.sh         # Full sweep script
├── Makefile
└── README.md
```

## Build

```bash
make baseline    # FP16 baseline kernel
make int8kv      # INT8 KV fused kernel
make all         # Both
```

Requires: CUDA Toolkit 12.x, Python 3.10+, PyTorch 2.x

## Related Work

- [flashattn-cuda-metal](https://github.com/Bias92/flashattn-cuda-metal) — FlashAttention CUDA/Metal cross-platform implementation & profiling
- [sdpa-attention-benchmark](https://github.com/Bias92/sdpa-attention-benchmark) — Benchmark PyTorch SDPA backends (math vs flash) on RTX 4060 Ti
- Dao et al., "FlashAttention" (NeurIPS 2022)
- Hooper et al., "KVQuant" (NeurIPS 2024)

## Author

[@Bias92](https://github.com/Bias92)
