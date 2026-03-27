"""Per-token symmetric INT8 quantization for KV-cache.

Usage:
    from utils.quantize import quantize_per_token_int8, dequantize_per_token_int8
"""

import torch


def quantize_per_token_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8 quantization.

    Each token (last dim = head_dim) gets its own scale factor.

    Args:
        x: [..., D] tensor in FP16 or FP32.

    Returns:
        x_int8: [..., D] in torch.int8
        scale:  [...] in torch.float32 (one scale per token)
    """
    # Per-token absmax along head_dim
    absmax = x.abs().amax(dim=-1, keepdim=True)  # [..., 1]
    scale = (absmax / 127.0).clamp(min=1e-10)    # avoid div-by-zero

    x_scaled = x / scale
    x_int8 = x_scaled.round().clamp(-128, 127).to(torch.int8)

    return x_int8, scale.squeeze(-1).float()


def dequantize_per_token_int8(
    x_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reference dequantization (for correctness checking).

    Args:
        x_int8: [..., D] in torch.int8
        scale:  [...] in torch.float32

    Returns:
        x_fp: [..., D] in torch.float32
    """
    return x_int8.float() * scale.unsqueeze(-1)


def quantization_error(
    x_original: torch.Tensor,
    x_int8: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    """Compute quantization error metrics.

    Args:
        x_original: original tensor (FP16/FP32)
        x_int8: quantized tensor (INT8)
        scale: scale factors

    Returns:
        dict with max_abs_diff, cos_sim, relative_l2_error
    """
    x_reconstructed = dequantize_per_token_int8(x_int8, scale)
    x_orig_flat = x_original.float().flatten()
    x_recon_flat = x_reconstructed.flatten()

    diff = (x_orig_flat - x_recon_flat)

    return {
        "max_abs_diff": diff.abs().max().item(),
        "cos_sim": torch.nn.functional.cosine_similarity(
            x_orig_flat.unsqueeze(0),
            x_recon_flat.unsqueeze(0),
        ).item(),
        "relative_l2_error": (
            diff.norm() / x_orig_flat.norm()
        ).item(),
    }


if __name__ == "__main__":
    # Quick sanity check
    torch.manual_seed(42)
    x = torch.randn(2, 32, 4096, 128, dtype=torch.float16, device="cuda")

    x_int8, scale = quantize_per_token_int8(x)
    metrics = quantization_error(x, x_int8, scale)

    print(f"Shape: {x.shape}")
    print(f"INT8 shape: {x_int8.shape}, Scale shape: {scale.shape}")
    print(f"Max abs diff:      {metrics['max_abs_diff']:.6f}")
    print(f"Cosine similarity: {metrics['cos_sim']:.6f}")
    print(f"Relative L2 error: {metrics['relative_l2_error']:.6f}")
    print(f"Memory: FP16={x.nbytes/1e6:.1f}MB, INT8={x_int8.nbytes/1e6:.1f}MB "
          f"({x_int8.nbytes/x.nbytes*100:.1f}%)")
