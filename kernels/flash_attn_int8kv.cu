/**
 * flash_attn_int8kv.cu — Fused INT8 KV Decode Attention
 * 
 * INT8 KV-cache storage + on-the-fly register-level dequant + FP16 Tensor Core mma.
 * FlashAttention-style tiling with online softmax.
 * 
 * Key differences from FP16 baseline:
 *   - K, V loaded as INT8 (1 byte vs 2 bytes) → HBM read halved for KV
 *   - Shared memory usage ~48% less → enables double buffering at Bc=256
 *   - Register-level dequant: INT8 → FP16 + scale multiplication before mma
 * 
 * Target: A100 SXM 80GB (sm_80)
 * 
 * TODO: Implement in Step 2
 */

#include "attention.h"

// TODO: Step 2 implementation
