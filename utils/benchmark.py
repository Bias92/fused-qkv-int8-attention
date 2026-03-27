"""Benchmark harness for attention kernels.

Provides consistent timing with CUDA events, warmup, and statistical reporting.
"""

import torch
import time
import csv
import os
from dataclasses import dataclass


@dataclass
class BenchConfig:
    batch: int
    num_heads: int
    seq_kv: int
    head_dim: int = 128
    seq_q: int = 1  # decode


@dataclass
class BenchResult:
    config: BenchConfig
    kernel_name: str
    median_us: float
    mean_us: float
    min_us: float
    max_us: float
    std_us: float


# Default sweep configs (15 total)
DEFAULT_CONFIGS = [
    BenchConfig(batch=b, num_heads=32, seq_kv=s)
    for b in [1, 4, 8]
    for s in [1024, 2048, 4096, 8192, 16384]
]


def benchmark_kernel(
    fn,
    warmup: int = 10,
    repeats: int = 100,
) -> list[float]:
    """Time a kernel function using CUDA events.

    Args:
        fn: callable that runs the kernel (no args)
        warmup: number of warmup iterations
        repeats: number of timed iterations

    Returns:
        list of elapsed times in microseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms → μs

    return times


def compute_stats(times: list[float]) -> dict:
    """Compute statistics from timing results."""
    t = torch.tensor(times)
    return {
        "median_us": t.median().item(),
        "mean_us": t.mean().item(),
        "min_us": t.min().item(),
        "max_us": t.max().item(),
        "std_us": t.std().item(),
    }


def save_results(results: list[BenchResult], path: str):
    """Save benchmark results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kernel", "batch", "num_heads", "seq_kv", "head_dim",
            "median_us", "mean_us", "min_us", "max_us", "std_us",
        ])
        for r in results:
            writer.writerow([
                r.kernel_name,
                r.config.batch,
                r.config.num_heads,
                r.config.seq_kv,
                r.config.head_dim,
                f"{r.median_us:.2f}",
                f"{r.mean_us:.2f}",
                f"{r.min_us:.2f}",
                f"{r.max_us:.2f}",
                f"{r.std_us:.2f}",
            ])
    print(f"Results saved to {path}")
