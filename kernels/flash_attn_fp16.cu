/**
 * flash_attn_fp16.cu — FP16 Decode Attention Baseline
 * 
 * FlashAttention-style tiled decode attention (S_q=1, D=128).
 * FP16 Tensor Core mma, online softmax.
 * 
 * This is the PERFORMANCE BASELINE for latency comparison.
 * Correctness reference is PyTorch SDPA.
 * 
 * Target: A100 SXM 80GB (sm_80)
 * 
 * TODO: Implement in Step 1
 */

#include "attention.h"

// TODO: Step 1 implementation
