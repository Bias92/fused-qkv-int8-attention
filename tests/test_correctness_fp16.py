"""Test FP16 decode attention kernel against PyTorch SDPA reference.

Success criteria:
  - cosine_similarity > 0.9999
  - max_abs_diff < 0.01

Usage:
    python tests/test_correctness_fp16.py
"""

import torch
import torch.nn.functional as F
import sys

# TODO: import compiled FP16 kernel after Step 1
# from build import flash_attn_fp16


def test_fp16_correctness(B, H, S_kv, D=128):
    """Test FP16 kernel against PyTorch SDPA."""
    torch.manual_seed(42)

    Q = torch.randn(B, H, 1, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, S_kv, D, dtype=torch.float16, device="cuda")

    # Reference
    ref = F.scaled_dot_product_attention(Q, K, V)

    # Our kernel
    # out = flash_attn_fp16(Q, K, V)  # TODO: uncomment after Step 1

    # Metrics
    # out_flat = out.flatten().float()
    # ref_flat = ref.flatten().float()
    # cos_sim = F.cosine_similarity(out_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
    # max_diff = (out - ref).abs().max().item()

    # print(f"Config: B={B}, H={H}, S_kv={S_kv}, D={D}")
    # print(f"  cos_sim:  {cos_sim:.6f} {'PASS' if cos_sim > 0.9999 else 'FAIL'}")
    # print(f"  max_diff: {max_diff:.6f} {'PASS' if max_diff < 0.01 else 'FAIL'}")

    print(f"Config: B={B}, H={H}, S_kv={S_kv}, D={D} — TODO: implement kernel first")


if __name__ == "__main__":
    configs = [
        (1, 32, 1024),
        (1, 32, 4096),
        (4, 32, 4096),
        (8, 32, 4096),
        (1, 32, 16384),
    ]

    print("=" * 60)
    print("FP16 Decode Attention — Correctness Test")
    print("Reference: torch.nn.functional.scaled_dot_product_attention")
    print("=" * 60)

    for B, H, S_kv in configs:
        test_fp16_correctness(B, H, S_kv)
        print()
