"""Test per-token INT8 quantization correctness.

This test is runnable immediately (no kernel compilation needed).
Verifies that quantize → dequantize roundtrip has acceptable error.

Usage:
    python tests/test_quantize.py
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quantize import quantize_per_token_int8, quantization_error


def test_roundtrip(B, H, S, D=128):
    """Test quantize → dequantize roundtrip error."""
    torch.manual_seed(42)
    x = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

    x_int8, scale = quantize_per_token_int8(x)
    metrics = quantization_error(x, x_int8, scale)

    pass_cos = metrics["cos_sim"] > 0.999
    pass_diff = metrics["max_abs_diff"] < 0.1
    pass_l2 = metrics["relative_l2_error"] < 0.02

    status = "PASS" if (pass_cos and pass_diff and pass_l2) else "FAIL"
    print(f"({B:2d}, {H:2d}, {S:5d}, {D}) | "
          f"cos={metrics['cos_sim']:.6f} | "
          f"max={metrics['max_abs_diff']:.6f} | "
          f"l2={metrics['relative_l2_error']:.6f} | "
          f"mem={x_int8.nbytes/x.nbytes*100:.0f}% | "
          f"{status}")

    return status == "PASS"


if __name__ == "__main__":
    print("=" * 80)
    print("Per-Token INT8 Quantization — Roundtrip Test")
    print("=" * 80)

    configs = [
        (1, 32, 1024),
        (1, 32, 4096),
        (1, 32, 16384),
        (4, 32, 4096),
        (8, 32, 4096),
    ]

    all_pass = True
    for B, H, S in configs:
        if not test_roundtrip(B, H, S):
            all_pass = False

    print("=" * 80)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
