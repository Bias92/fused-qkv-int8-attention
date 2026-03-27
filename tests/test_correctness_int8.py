"""Test INT8 KV fused decode attention kernel against PyTorch SDPA reference.

Success criteria (all 3 must pass):
  - cosine_similarity > 0.999
  - max_abs_diff < 0.05
  - relative_l2_error < 0.01

Usage:
    python tests/test_correctness_int8.py
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quantize import quantize_per_token_int8

# TODO: import compiled INT8 KV kernel after Step 2
# from build import flash_attn_int8kv


CONFIGS = [
    # (B, H, S_kv)
    (1, 32, 1024),
    (1, 32, 2048),
    (1, 32, 4096),
    (1, 32, 8192),
    (1, 32, 16384),
    (4, 32, 1024),
    (4, 32, 2048),
    (4, 32, 4096),
    (4, 32, 8192),
    (4, 32, 16384),
    (8, 32, 1024),
    (8, 32, 2048),
    (8, 32, 4096),
    (8, 32, 8192),
    (8, 32, 16384),
]


def test_int8kv_correctness(B, H, S_kv, D=128):
    """Test INT8 KV kernel against PyTorch SDPA.

    Note: Includes quantization error, so thresholds are relaxed vs FP16 test.
    """
    torch.manual_seed(42)

    Q = torch.randn(B, H, 1, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")

    # Reference: FP16 SDPA (correctness reference)
    ref = F.scaled_dot_product_attention(Q, K, V)

    # Quantize KV
    K_int8, K_scale = quantize_per_token_int8(K)
    V_int8, V_scale = quantize_per_token_int8(V)

    # Our kernel
    # out = flash_attn_int8kv(Q, K_int8, K_scale, V_int8, V_scale)  # TODO

    # Metrics
    # ref_flat = ref.flatten().float()
    # out_flat = out.flatten().float()
    # diff = ref_flat - out_flat
    #
    # cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), out_flat.unsqueeze(0)).item()
    # max_diff = diff.abs().max().item()
    # rel_l2 = diff.norm().item() / ref_flat.norm().item()
    #
    # pass_cos = cos_sim > 0.999
    # pass_diff = max_diff < 0.05
    # pass_l2 = rel_l2 < 0.01
    # all_pass = pass_cos and pass_diff and pass_l2
    #
    # status = "PASS" if all_pass else "FAIL"
    # print(f"({B:2d}, {H:2d}, {S_kv:5d}, {D}) | "
    #       f"cos={cos_sim:.4f} {'✓' if pass_cos else '✗'} | "
    #       f"max={max_diff:.4f} {'✓' if pass_diff else '✗'} | "
    #       f"l2={rel_l2:.4f} {'✓' if pass_l2 else '✗'} | "
    #       f"{status}")

    print(f"({B:2d}, {H:2d}, {S_kv:5d}, {D}) — TODO: implement kernel first")


if __name__ == "__main__":
    print("=" * 80)
    print("INT8 KV Fused Decode Attention — Correctness Test (15 configs)")
    print("Reference: torch.nn.functional.scaled_dot_product_attention (FP16)")
    print("Criteria: cos_sim > 0.999, max_diff < 0.05, rel_l2_err < 0.01")
    print("=" * 80)
    print(f"{'Config':>28s} | {'cos_sim':>8s}   | {'max_diff':>8s}   | {'rel_l2':>8s}   | Status")
    print("-" * 80)

    for B, H, S_kv in CONFIGS:
        test_int8kv_correctness(B, H, S_kv)

    print("=" * 80)
