NVCC := nvcc
ARCH := sm_80
CUDA_FLAGS := -O3 -arch=$(ARCH) --use_fast_math -Xcompiler -fPIC --expt-relaxed-constexpr
INCLUDE := -Icsrc
BUILD_DIR := build

.PHONY: all baseline int8kv clean

all: baseline int8kv

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# FP16 baseline decode attention kernel
baseline: $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE) \
		kernels/flash_attn_fp16.cu \
		-o $(BUILD_DIR)/flash_attn_fp16.so \
		-shared

# INT8 KV fused decode attention kernel
int8kv: $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE) \
		kernels/flash_attn_int8kv.cu \
		-o $(BUILD_DIR)/flash_attn_int8kv.so \
		-shared

# Profile with NCU (S_kv=4096, B=1)
profile-baseline: baseline
	ncu --set full \
		--target-processes all \
		-o profiles/baseline_fp16 \
		python bench/bench_fp16.py --seq_kv 4096 --batch 1 --profile

profile-int8kv: int8kv
	ncu --set full \
		--target-processes all \
		-o profiles/int8kv \
		python bench/bench_int8kv.py --seq_kv 4096 --batch 1 --profile

clean:
	rm -rf $(BUILD_DIR)
