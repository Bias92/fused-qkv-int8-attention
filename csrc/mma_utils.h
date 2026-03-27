#pragma once

/**
 * mma_utils.h — Tensor Core MMA wrappers for A100 (sm_80)
 *
 * FP16 mma: hmma.m16n8k16 (input FP16, accumulator FP32)
 * Used for both QK^T and PV matmuls.
 *
 * INT8 → FP16 dequant helpers are also defined here.
 *
 * TODO: Implement in Step 1/2
 */

#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

// Dequantize a single INT8 value to FP16 with scale
__device__ __forceinline__
half dequant_scalar(int8_t val, float scale) {
    return __float2half(__int2float_rn(val) * scale);
}

// Vectorized dequant: 4 × INT8 → 4 × FP16 (packed int32 load)
__device__ __forceinline__
void dequant_vec4(const int8_t* src, half* dst, float scale) {
    int32_t packed = *reinterpret_cast<const int32_t*>(src);

    dst[0] = __float2half(__int2float_rn((int8_t)(packed & 0xFF)) * scale);
    dst[1] = __float2half(__int2float_rn((int8_t)((packed >> 8) & 0xFF)) * scale);
    dst[2] = __float2half(__int2float_rn((int8_t)((packed >> 16) & 0xFF)) * scale);
    dst[3] = __float2half(__int2float_rn((int8_t)((packed >> 24) & 0xFF)) * scale);
}
