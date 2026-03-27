# Fused Quantized-KV Attention Kernel

Custom CUDA kernel that fuses INT8 KV-cache dequantization with FlashAttention-style tiled attention on A100 GPUs.

## Claim

> KV-cache를 per-token symmetric INT8로 저장하고 attention kernel 내부에서 fused dequant하면, FP16 baseline 대비 decode attention의 HBM traffic이 줄고 latency가 개선된다.

## Why

LLM inference에서 KV-cache는 메모리 병목의 주범이다. Decode attention (`S_q=1`)은 memory-bound이므로, KV를 INT8로 저장하면 HBM→SM 전송량이 절반으로 줄어 직접적인 latency 감소로 이어진다. 추가로 shared memory 사용량 감소 (48.4%)로 double buffering이 가능해져 compute-memory overlap이 개선된다.

## Key Idea

```
HBM (INT8 KV) → Shared Memory (INT8) → Registers (INT8→FP16 dequant + scale) → FP16 Tensor Core mma
```

- **Storage:** K, V를 per-token symmetric INT8로 quantize하여 HBM에 저장
- **Compute:** Kernel 내부에서 register-level dequant → FP16 Tensor Core mma (`hmma.m16n8k16`)
- **Benefit:** HBM read 감소 + smem 절감 → double buffering 가능 (FP16은 불가)

## Target Hardware

| Spec | Value |
|------|-------|
| GPU | A100 SXM 80GB |
| Compute Capability | 8.0 |
| HBM2e Bandwidth | 2,039 GB/s |
| Shared Memory / SM | 최대 164 KB |
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

> TODO: 실험 완료 후 업데이트

### Correctness

| Config | cos_sim | max_diff | rel_l2_err | PASS/FAIL |
|--------|---------|----------|------------|-----------|
| ... | ... | ... | ... | ... |

### Latency & HBM Traffic

> TODO: 그래프 추가

### NCU Analysis

> TODO: 프로파일링 결과 추가

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
- Dao et al., "FlashAttention" (NeurIPS 2022)
- Hooper et al., "KVQuant" (NeurIPS 2024)

## Author

[@Bias92](https://github.com/Bias92)
