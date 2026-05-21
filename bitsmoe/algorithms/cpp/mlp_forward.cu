#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

using torch::Tensor;

namespace {

constexpr int K_STAGE = 16;
constexpr int N_TILE = 128;
constexpr int N_MICRO = 32;
constexpr int MICROS_PER_TILE = N_TILE / N_MICRO;
constexpr int GATE_TC_M = 16;
constexpr int GATE_TC_N = 16;
constexpr int GATE_TC_K = 16;
constexpr int GATE_TC_COMPUTE_WARPS = 4;
constexpr int GATE_TC_BLOCK_THREADS = 256;
constexpr int GATE_SMALL_GROUP = 4;
constexpr int GATE_SMALL_SUBTILES = MICROS_PER_TILE;
constexpr int GATE_SMALL_FULL_WARPS = GATE_SMALL_SUBTILES;
constexpr int GATE_SMALL_FULL_THREADS = GATE_SMALL_FULL_WARPS * 32;
constexpr int GATE_SMALL_MICROS_PER_CTA = 2;
constexpr int GATE_SMALL_WARPS = GATE_SMALL_MICROS_PER_CTA;
constexpr int GATE_SMALL_THREADS = GATE_SMALL_WARPS * 32;
constexpr int GATE_SMALL_PACK_WORDS_MAX = K_STAGE * 16;
constexpr int GATE_PACK_WORDS_MAX = K_STAGE * MICROS_PER_TILE * 16;
constexpr int DOWN_TC_M = 16;
constexpr int DOWN_TC_N = 16;
constexpr int DOWN_TC_K = 16;
constexpr int DOWN_TC_WARPS = N_TILE / 32;
// Keep 2 compute warps by default: 3 compute warps would significantly inflate
// shared stage buffers and can reduce occupancy on 100KB-SMEM parts.
constexpr int DOWN_TC_COMPUTE_WARPS = 2;
constexpr int DOWN_TC_LOADER_WARPS = DOWN_TC_WARPS - DOWN_TC_COMPUTE_WARPS;
constexpr int DOWN_TC_ROWS_PER_WARP = DOWN_TC_M;
constexpr int DOWN_TC_ROW_BATCH = DOWN_TC_ROWS_PER_WARP * DOWN_TC_COMPUTE_WARPS;
constexpr int DOWN_TC_A_BATCH = DOWN_TC_N;
constexpr int DOWN_TC_A_GROUP_BATCHES = 2;
constexpr int DOWN_TC_A_GROUP = DOWN_TC_A_BATCH * DOWN_TC_A_GROUP_BATCHES;
constexpr int DOWN_TC_SMALL_ASSIGN_THRESHOLD = 8;
constexpr int DOWN_SMALL_MAX_ASSIGN = 8;
constexpr int DOWN_SMALL_A_GROUP = DOWN_SMALL_MAX_ASSIGN;
constexpr int DOWN_SMALL_THREADS = N_TILE;
constexpr int DOWN_SMALL_WARPS = DOWN_SMALL_THREADS / 32;
constexpr int DOWN_SMALL_PACK_WORDS_MAX = MICROS_PER_TILE * 16;
constexpr int DECODE_SMALL_MAX_NUM_ASSIGN = 64;
constexpr float DECODE_SMALL_AVG_ASSIGN_THRESHOLD = 2.0f;

#ifndef BITSMOE_SMALL_GATE_UP_SEPARATE_LAUNCH
#define BITSMOE_SMALL_GATE_UP_SEPARATE_LAUNCH 0
#endif

#ifndef BITSMOE_SMALL_GATE_UP_DUAL_FALLBACK_128T
#define BITSMOE_SMALL_GATE_UP_DUAL_FALLBACK_128T 0
#endif

__device__ __forceinline__ int bits_offset(int bits) {
    switch (bits) {
        case 1: return 1;
        case 2: return 2;
        case 3: return 4;
        case 4: return 8;
        case 6: return 32;
        case 8: return 128;
        default: return 0;
    }
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void cp_async_cg_16(
    void* smem_ptr,
    const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
    const uint32_t smem_u32 =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :
        : "r"(smem_u32), "l"(gmem_ptr), "n"(16));
#else
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_group_1() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

struct GateStageDesc {
    int valid;
    int slab_i;
    int stage_idx;
    int bits;
    int rank_off;
    int pack_off;
    int scale_off;
    int n_valid;
    int n_chunks;
    int words_per_row;
    int stage_row_base;
    int rows_cur;
};

__device__ __forceinline__ GateStageDesc make_invalid_gate_stage_desc() {
    GateStageDesc desc{};
    desc.valid = 0;
    return desc;
}

__device__ __forceinline__ GateStageDesc find_next_gate_stage_desc(
    const int32_t* __restrict__ slab_meta,
    const int32_t* __restrict__ tile_meta,
    int num_slabs,
    int num_tiles_total,
    int tile_local,
    int slab_begin,
    int stage_begin) {
    for (int slab_i = slab_begin; slab_i < num_slabs; ++slab_i) {
        const int bits = slab_meta[slab_i * 4 + 0];
        const int k_slab = slab_meta[slab_i * 4 + 1];
        const int rank_off = slab_meta[slab_i * 4 + 2];
        const int tile_off = slab_meta[slab_i * 4 + 3];
        const int tile_end = (slab_i + 1 < num_slabs)
            ? slab_meta[(slab_i + 1) * 4 + 3]
            : num_tiles_total;

        const int t = tile_off + tile_local;
        if (t >= tile_end) {
            continue;
        }

        const int pack_off = tile_meta[t * 3 + 0];
        const int scale_off = tile_meta[t * 3 + 1];
        const int n_valid = tile_meta[t * 3 + 2];
        const int n_chunks = (n_valid + GATE_TC_N - 1) / GATE_TC_N;
        if (n_chunks <= 0) {
            continue;
        }

        const int stage_count = (k_slab + K_STAGE - 1) / K_STAGE;
        if (stage_count <= 0) {
            continue;
        }

        const int stage_idx = (slab_i == slab_begin) ? stage_begin : 0;
        if (stage_idx >= stage_count) {
            continue;
        }

        GateStageDesc desc{};
        desc.valid = 1;
        desc.slab_i = slab_i;
        desc.stage_idx = stage_idx;
        desc.bits = bits;
        desc.rank_off = rank_off;
        desc.pack_off = pack_off;
        desc.scale_off = scale_off;
        desc.n_valid = n_valid;
        desc.n_chunks = n_chunks;
        desc.words_per_row = (bits == 16) ? 16 : bits;
        desc.stage_row_base = stage_idx * K_STAGE;
        desc.rows_cur = min(K_STAGE, k_slab - desc.stage_row_base);
        return desc;
    }
    return make_invalid_gate_stage_desc();
}

__device__ __forceinline__ int stage_prefix_words(
    int k_slab,
    int stage,
    int words_per_row) {
    int off = 0;
    for (int s = 0; s < stage; ++s) {
        const int rows = min(K_STAGE, k_slab - s * K_STAGE);
        off += rows * MICROS_PER_TILE * words_per_row;
    }
    return off;
}

__device__ __forceinline__ float decode_tile_weight(
    const uint32_t* payload,
    int64_t pack_off,
    int bits,
    int k_slab,
    int row,
    int col_in_tile) {
    const int stage = row / K_STAGE;
    const int row_in_stage = row - stage * K_STAGE;
    const int rows_cur = min(K_STAGE, k_slab - stage * K_STAGE);
    const int micro = col_in_tile / N_MICRO;
    const int lane = col_in_tile - micro * N_MICRO;

    if (bits == 16) {
        constexpr int words_per_row = 16;
        const int base =
            stage_prefix_words(k_slab, stage, words_per_row)
            + micro * rows_cur * words_per_row
            + row_in_stage * words_per_row;
        const int word_idx = lane >> 1;
        const uint32_t word = payload[pack_off + base + word_idx];
        const uint16_t hbits = (lane & 1)
            ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
            : static_cast<uint16_t>(word & 0xFFFFu);
        const half hv = __ushort_as_half(hbits);
        return __half2float(hv);
    }

    const int words_per_row = bits;
    const int base =
        stage_prefix_words(k_slab, stage, words_per_row)
        + micro * rows_cur * words_per_row
        + row_in_stage * words_per_row;

    int value_u = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= bits) {
            break;
        }
        const uint32_t word = payload[pack_off + base + p];
        const int bit = static_cast<int>((word >> lane) & 1u);
        value_u |= (bit << p);
    }

    if (bits == 1) {
        return static_cast<float>((value_u << 1) - 1);
    }
    const int q = value_u - bits_offset(bits);
    return static_cast<float>(q);
}

__device__ __forceinline__ float decode_tile_weight_stage(
    const uint32_t* payload,
    int64_t pack_off,
    int bits,
    int rows_cur,
    int stage_prefix,
    int row_in_stage,
    int col_in_tile) {
    const int micro = col_in_tile / N_MICRO;
    const int lane = col_in_tile - micro * N_MICRO;

    if (bits == 16) {
        constexpr int words_per_row = 16;
        const int base =
            stage_prefix
            + micro * rows_cur * words_per_row
            + row_in_stage * words_per_row;
        const int word_idx = lane >> 1;
        const uint32_t word = payload[pack_off + base + word_idx];
        const uint16_t hbits = (lane & 1)
            ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
            : static_cast<uint16_t>(word & 0xFFFFu);
        return __half2float(__ushort_as_half(hbits));
    }

    const int words_per_row = bits;
    const int base =
        stage_prefix
        + micro * rows_cur * words_per_row
        + row_in_stage * words_per_row;
    int value_u = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= bits) {
            break;
        }
        const uint32_t word = payload[pack_off + base + p];
        const int bit = static_cast<int>((word >> lane) & 1u);
        value_u |= (bit << p);
    }
    if (bits == 1) {
        return static_cast<float>((value_u << 1) - 1);
    }
    const int q = value_u - bits_offset(bits);
    return static_cast<float>(q);
}

__device__ __forceinline__ int find_expert_for_assign(
    const int32_t* expert_offsets,
    int32_t num_experts,
    int32_t assign_idx) {
    int lo = 0;
    int hi = num_experts - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (expert_offsets[mid + 1] <= assign_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void build_gate_task_count_kernel(
    const int32_t* __restrict__ expert_offsets,
    int32_t num_experts,
    int32_t* __restrict__ task_counts) {
    const int e = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= num_experts) {
        return;
    }
    const int32_t begin = expert_offsets[e];
    const int32_t end = expert_offsets[e + 1];
    const int32_t assign_count = end - begin;
    task_counts[e] = (assign_count > 0)
        ? static_cast<int32_t>((assign_count + GATE_TC_M - 1) / GATE_TC_M)
        : 0;
}

__global__ void build_gate_task_table_kernel(
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ task_counts,
    const int32_t* __restrict__ task_prefix,
    int32_t num_experts,
    int32_t* __restrict__ task_expert,
    int32_t* __restrict__ task_assign_base,
    int32_t* __restrict__ task_assign_count) {
    const int e = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= num_experts) {
        return;
    }
    const int32_t count = task_counts[e];
    if (count <= 0) {
        return;
    }
    const int32_t begin = expert_offsets[e];
    const int32_t end = expert_offsets[e + 1];
    const int32_t out_base = task_prefix[e] - count;
    for (int32_t i = 0; i < count; ++i) {
        const int32_t assign_base = begin + i * GATE_TC_M;
        const int32_t rows = min(GATE_TC_M, end - assign_base);
        const int32_t out_idx = out_base + i;
        task_expert[out_idx] = e;
        task_assign_base[out_idx] = assign_base;
        task_assign_count[out_idx] = rows;
    }
}

__device__ __forceinline__ float decode_tile_weight_stage_f16_warp(
    const uint32_t* payload,
    int64_t pack_off,
    int rows_cur,
    int stage_prefix,
    int row_in_stage,
    int col_in_tile) {
    constexpr int words_per_row = 16;
    const int micro = col_in_tile / N_MICRO;
    const int lane = col_in_tile - micro * N_MICRO;
    const int base =
        stage_prefix
        + micro * rows_cur * words_per_row
        + row_in_stage * words_per_row;
    const int src_lane = lane & ~1;
    uint32_t word = 0;
    if ((lane & 1) == 0) {
        word = payload[pack_off + base + (lane >> 1)];
    }
    const unsigned mask = __activemask();
    word = __shfl_sync(mask, word, src_lane);
    const uint16_t hbits = (lane & 1)
        ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
        : static_cast<uint16_t>(word & 0xFFFFu);
    return __half2float(__ushort_as_half(hbits));
}

__device__ __forceinline__ float decode_tile_weight_stage_lowbit_warp(
    const uint32_t* payload,
    int64_t pack_off,
    int bits,
    int rows_cur,
    int stage_prefix,
    int row_in_stage,
    int col_in_tile) {
    const int micro = col_in_tile / N_MICRO;
    const int lane = col_in_tile - micro * N_MICRO;
    const int words_per_row = bits;
    const int base =
        stage_prefix
        + micro * rows_cur * words_per_row
        + row_in_stage * words_per_row;
    const unsigned mask = __activemask();
    uint32_t plane_word = 0;
    if (lane < bits) {
        plane_word = payload[pack_off + base + lane];
    }
    int value_u = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= bits) {
            break;
        }
        const uint32_t word = __shfl_sync(mask, plane_word, p);
        const int bit = static_cast<int>((word >> lane) & 1u);
        value_u |= (bit << p);
    }
    if (bits == 1) {
        return static_cast<float>((value_u << 1) - 1);
    }
    const int q = value_u - bits_offset(bits);
    return static_cast<float>(q);
}

__device__ __forceinline__ float decode_tile_weight_stage_warp(
    const uint32_t* payload,
    int64_t pack_off,
    int bits,
    int rows_cur,
    int stage_prefix,
    int row_in_stage,
    int col_in_tile) {
    if (bits == 16) {
        return decode_tile_weight_stage_f16_warp(
            payload,
            pack_off,
            rows_cur,
            stage_prefix,
            row_in_stage,
            col_in_tile);
    }
    return decode_tile_weight_stage_lowbit_warp(
        payload,
        pack_off,
        bits,
        rows_cur,
        stage_prefix,
        row_in_stage,
        col_in_tile);
}

__device__ __forceinline__ float warp_sum(float v) {
    const unsigned mask = __activemask();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

__device__ __forceinline__ float decode_row_weight_from_pack_warp_runtime(
    const uint32_t* row_pack,
    int bits,
    int col_in_tile) {
    const int micro = col_in_tile / N_MICRO;
    const int lane = col_in_tile - micro * N_MICRO;
    const unsigned mask = __activemask();
    if (bits == 16) {
        const int base = micro * 16;
        const int src_lane = lane & ~1;
        uint32_t word = 0;
        if ((lane & 1) == 0) {
            word = row_pack[base + (lane >> 1)];
        }
        word = __shfl_sync(mask, word, src_lane);
        const uint16_t hbits = (lane & 1)
            ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
            : static_cast<uint16_t>(word & 0xFFFFu);
        return __half2float(__ushort_as_half(hbits));
    }

    const int base = micro * bits;
    uint32_t plane_word = 0;
    if (lane < bits) {
        plane_word = row_pack[base + lane];
    }
    int value_u = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= bits) {
            break;
        }
        const uint32_t word = __shfl_sync(mask, plane_word, p);
        value_u |= (static_cast<int>((word >> lane) & 1u) << p);
    }
    if (bits == 1) {
        return static_cast<float>((value_u << 1) - 1);
    }
    return static_cast<float>(value_u - bits_offset(bits));
}

__device__ __forceinline__ float decode_tile_weight_stage_from_pack_warp(
    const uint32_t* pack_stage,
    int bits,
    int rows_cur,
    int row_in_stage,
    int col_in_tile) {
    const int micro = col_in_tile / N_MICRO;
    const int lane = col_in_tile - micro * N_MICRO;
    const unsigned mask = __activemask();
    if (bits == 16) {
        constexpr int words_per_row = 16;
        const int base =
            micro * rows_cur * words_per_row
            + row_in_stage * words_per_row;
        const int src_lane = lane & ~1;
        uint32_t word = 0;
        if ((lane & 1) == 0) {
            word = pack_stage[base + (lane >> 1)];
        }
        word = __shfl_sync(mask, word, src_lane);
        const uint16_t hbits = (lane & 1)
            ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
            : static_cast<uint16_t>(word & 0xFFFFu);
        return __half2float(__ushort_as_half(hbits));
    }

    const int words_per_row = bits;
    const int base =
        micro * rows_cur * words_per_row
        + row_in_stage * words_per_row;
    uint32_t plane_word = 0;
    if (lane < bits) {
        plane_word = pack_stage[base + lane];
    }
    int value_u = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= bits) {
            break;
        }
        const uint32_t word = __shfl_sync(mask, plane_word, p);
        value_u |= (static_cast<int>((word >> lane) & 1u) << p);
    }
    if (bits == 1) {
        return static_cast<float>((value_u << 1) - 1);
    }
    return static_cast<float>(value_u - bits_offset(bits));
}

__device__ __forceinline__ void prefetch_gate_pack_stage_async(
    int lane_tid,
    int lane_stride,
    uint32_t* __restrict__ pack_stage,
    const uint32_t* __restrict__ payload,
    int32_t pack_off,
    int32_t stage_prefix_words,
    int32_t stage_words) {
    const int vec_words = 4;
    const int vec_count = stage_words / vec_words;
    for (int v = lane_tid; v < vec_count; v += lane_stride) {
        const int idx = v * vec_words;
        cp_async_cg_16(
            &pack_stage[idx],
            &payload[pack_off + stage_prefix_words + idx]);
    }
    for (int idx = vec_count * vec_words + lane_tid; idx < stage_words; idx += lane_stride) {
        pack_stage[idx] = payload[pack_off + stage_prefix_words + idx];
    }
    if (vec_count > 0) {
        cp_async_commit_group();
    }
}

__device__ __forceinline__ void prefetch_gate_pack_desc_async(
    int lane_tid,
    int lane_stride,
    uint32_t* __restrict__ pack_stage,
    const uint32_t* __restrict__ payload,
    const GateStageDesc& desc) {
    if (!desc.valid) {
        return;
    }
    const int stage_prefix_words = desc.stage_row_base * MICROS_PER_TILE * desc.words_per_row;
    const int stage_words = desc.rows_cur * MICROS_PER_TILE * desc.words_per_row;
    prefetch_gate_pack_stage_async(
        lane_tid,
        lane_stride,
        pack_stage,
        payload,
        desc.pack_off,
        stage_prefix_words,
        stage_words);
}

__device__ __forceinline__ void prefetch_gate_pack_micro_stage_async(
    int lane,
    uint32_t* __restrict__ pack_stage,
    const uint32_t* __restrict__ payload,
    int32_t pack_off,
    int32_t stage_prefix_words,
    int32_t rows_cur,
    int32_t words_per_row,
    int32_t subtile) {
    if (rows_cur <= 0) {
        return;
    }
    const int micro_stage_words = rows_cur * words_per_row;
    const int src_off = pack_off + stage_prefix_words + subtile * micro_stage_words;
    const bool can_cp_async = ((src_off & 3) == 0);
    const int vec_words = 4;
    int vec_tail_begin = 0;
    bool cp_async_issued = false;
    if (can_cp_async) {
        const int vec_count = micro_stage_words / vec_words;
        vec_tail_begin = vec_count * vec_words;
        for (int v = lane; v < vec_count; v += 32) {
            const int idx = v * vec_words;
            cp_async_cg_16(
                &pack_stage[idx],
                &payload[src_off + idx]);
            cp_async_issued = true;
        }
    }
    const int scalar_begin = can_cp_async ? vec_tail_begin : 0;
    for (int idx = scalar_begin + lane; idx < micro_stage_words; idx += 32) {
        pack_stage[idx] = payload[src_off + idx];
    }
    if (cp_async_issued) {
        cp_async_commit_group();
    }
}

__device__ __forceinline__ float decode_tile_weight_stage_from_pack_micro_warp(
    const uint32_t* pack_stage,
    int bits,
    int row_in_stage) {
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31;
    if (bits == 16) {
        constexpr int words_per_row = 16;
        const int base = row_in_stage * words_per_row;
        const int src_lane = lane & ~1;
        uint32_t word = 0;
        if ((lane & 1) == 0) {
            word = pack_stage[base + (lane >> 1)];
        }
        word = __shfl_sync(mask, word, src_lane);
        const uint16_t hbits = (lane & 1)
            ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
            : static_cast<uint16_t>(word & 0xFFFFu);
        return __half2float(__ushort_as_half(hbits));
    }

    const int words_per_row = bits;
    const int base = row_in_stage * words_per_row;
    uint32_t plane_word = 0;
    if (lane < bits) {
        plane_word = pack_stage[base + lane];
    }
    int value_u = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= bits) {
            break;
        }
        const uint32_t word = __shfl_sync(mask, plane_word, p);
        value_u |= (static_cast<int>((word >> lane) & 1u) << p);
    }
    if (bits == 1) {
        return static_cast<float>((value_u << 1) - 1);
    }
    return static_cast<float>(value_u - bits_offset(bits));
}

__device__ __forceinline__ void materialize_gate_stage_from_pack(
    int lane_tid,
    int lane_stride,
    half* __restrict__ a_stage,
    half* __restrict__ b_stage,
    const half* __restrict__ h,
    int32_t h_rank,
    const int32_t* __restrict__ token_indices,
    int32_t assign_base,
    int32_t assign_block_row_base,
    int32_t valid_rows,
    const uint32_t* __restrict__ pack_stage,
    const int32_t* __restrict__ rank_idx,
    const half* __restrict__ scale,
    const half* __restrict__ s_vals,
    int32_t bits,
    int32_t rank_off,
    int32_t scale_off,
    int32_t n_valid,
    int32_t stage_row_base,
    int32_t rows_cur) {
    for (int idx = lane_tid; idx < GATE_TC_M * GATE_TC_K; idx += lane_stride) {
        a_stage[idx] = __float2half(0.0f);
    }
    for (int idx = lane_tid; idx < GATE_TC_K * N_TILE; idx += lane_stride) {
        b_stage[idx] = __float2half(0.0f);
    }
    if (rows_cur <= 0) {
        return;
    }

    for (int idx = lane_tid; idx < valid_rows * rows_cur; idx += lane_stride) {
        const int row = idx / rows_cur;
        const int r = idx - row * rows_cur;
        const int a_local = assign_block_row_base + row;
        const int a = assign_base + a_local;
        const int token = token_indices[a];
        const int rk = rank_idx[rank_off + stage_row_base + r];
        const float hs = __half2float(h[static_cast<int64_t>(token) * h_rank + rk])
            * __half2float(s_vals[rank_off + stage_row_base + r]);
        a_stage[row * GATE_TC_K + r] = __float2half(hs);
    }

    for (int idx = lane_tid; idx < rows_cur * N_TILE; idx += lane_stride) {
        const int r = idx / N_TILE;
        const int c = idx - r * N_TILE;
        if (c >= n_valid) {
            continue;
        }
        float w = decode_tile_weight_stage_from_pack_warp(
            pack_stage,
            bits,
            rows_cur,
            r,
            c);
        if (bits != 16) {
            w *= __half2float(scale[scale_off + stage_row_base + r]);
        }
        b_stage[r * N_TILE + c] = __float2half(w);
    }
}

__device__ __forceinline__ void materialize_gate_stage_from_desc(
    int lane_tid,
    int lane_stride,
    half* __restrict__ a_stage,
    half* __restrict__ b_stage,
    const half* __restrict__ h,
    int32_t h_rank,
    const int32_t* __restrict__ token_indices,
    int32_t assign_base,
    int32_t assign_block_row_base,
    int32_t valid_rows,
    const uint32_t* __restrict__ pack_stage,
    const int32_t* __restrict__ rank_idx,
    const half* __restrict__ scale,
    const half* __restrict__ s_vals,
    const GateStageDesc& desc) {
    materialize_gate_stage_from_pack(
        lane_tid,
        lane_stride,
        a_stage,
        b_stage,
        h,
        h_rank,
        token_indices,
        assign_base,
        assign_block_row_base,
        valid_rows,
        pack_stage,
        rank_idx,
        scale,
        s_vals,
        desc.bits,
        desc.rank_off,
        desc.scale_off,
        desc.n_valid,
        desc.stage_row_base,
        desc.rows_cur);
}

__global__ void gate_up_tc_kernel(
    const half* __restrict__ h,
    int32_t h_rank,
    const int32_t* __restrict__ token_indices,
    const int64_t* __restrict__ payload_ptrs,
    const int64_t* __restrict__ rank_idx_ptrs,
    const int64_t* __restrict__ tile_meta_ptrs,
    const int64_t* __restrict__ slab_meta_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    const int64_t* __restrict__ s_ptrs,
    const int32_t* __restrict__ num_slabs_ptr,
    const int32_t* __restrict__ num_tiles_ptr,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_assign_base,
    const int32_t* __restrict__ task_assign_count,
    int32_t num_tasks,
    int32_t intermediate_size,
    float* __restrict__ out) {
    const int task = blockIdx.x;
    const int tile_local = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;

    if (task >= num_tasks) {
        return;
    }

    const int expert = task_expert[task];
    const int assign_base = task_assign_base[task];
    const int valid_rows = task_assign_count[task];
    if (valid_rows <= 0) {
        return;
    }
    const int col_tile_base = tile_local * N_TILE;
    if (col_tile_base >= intermediate_size) {
        return;
    }

    const auto* payload =
        reinterpret_cast<const uint32_t*>(payload_ptrs[expert]);
    const auto* rank_idx =
        reinterpret_cast<const int32_t*>(rank_idx_ptrs[expert]);
    const auto* tile_meta =
        reinterpret_cast<const int32_t*>(tile_meta_ptrs[expert]);
    const auto* slab_meta =
        reinterpret_cast<const int32_t*>(slab_meta_ptrs[expert]);
    const auto* scale =
        reinterpret_cast<const half*>(scale_ptrs[expert]);
    const auto* s_vals =
        reinterpret_cast<const half*>(s_ptrs[expert]);

    const int num_slabs = num_slabs_ptr[expert];
    const int num_tiles_total = num_tiles_ptr[expert];

    __shared__ __align__(16) half a_stage_sh[2][GATE_TC_M * GATE_TC_K];
    __shared__ __align__(16) half b_stage_sh[2][GATE_TC_K * N_TILE];
    __shared__ __align__(16) uint32_t b_pack_stage_sh[3][GATE_PACK_WORDS_MAX];
    __shared__ GateStageDesc gate_mat_stage_desc_sh[2];
    __shared__ GateStageDesc gate_prefetch_stage_desc_sh;
    __shared__ __align__(16) float c_store_sh[8][GATE_TC_M * GATE_TC_N];

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, GATE_TC_M, GATE_TC_N, GATE_TC_K, float> acc_frag0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, GATE_TC_M, GATE_TC_N, GATE_TC_K, float> acc_frag1;
    const bool is_loader_warp = warp_id >= GATE_TC_COMPUTE_WARPS;
    const int loader_tid = tid - GATE_TC_COMPUTE_WARPS * 32;
    const int loader_stride = GATE_TC_BLOCK_THREADS - GATE_TC_COMPUTE_WARPS * 32;
    constexpr int first_loader_tid = GATE_TC_COMPUTE_WARPS * 32;
    if (warp_id < GATE_TC_COMPUTE_WARPS) {
        nvcuda::wmma::fill_fragment(acc_frag0, 0.0f);
        nvcuda::wmma::fill_fragment(acc_frag1, 0.0f);
    }
    if (tid == 0) {
        gate_mat_stage_desc_sh[0] = find_next_gate_stage_desc(
            slab_meta,
            tile_meta,
            num_slabs,
            num_tiles_total,
            tile_local,
            0,
            0);
        gate_mat_stage_desc_sh[1] = make_invalid_gate_stage_desc();
        gate_prefetch_stage_desc_sh = gate_mat_stage_desc_sh[0].valid
            ? find_next_gate_stage_desc(
                slab_meta,
                tile_meta,
                num_slabs,
                num_tiles_total,
                tile_local,
                gate_mat_stage_desc_sh[0].slab_i,
                gate_mat_stage_desc_sh[0].stage_idx + 1)
            : make_invalid_gate_stage_desc();
    }
    __syncthreads();

    if (!gate_mat_stage_desc_sh[0].valid) {
        return;
    }

    GateStageDesc prefetch_desc = gate_prefetch_stage_desc_sh;
    if (is_loader_warp) {
        prefetch_gate_pack_desc_async(
            loader_tid,
            loader_stride,
            b_pack_stage_sh[0],
            payload,
            gate_mat_stage_desc_sh[0]);
        if (prefetch_desc.valid) {
            prefetch_gate_pack_desc_async(
                loader_tid,
                loader_stride,
                b_pack_stage_sh[1],
                payload,
                prefetch_desc);
            cp_async_wait_group_1();
        } else {
            cp_async_wait_group_0();
        }
        materialize_gate_stage_from_desc(
            loader_tid,
            loader_stride,
            a_stage_sh[0],
            b_stage_sh[0],
            h,
            h_rank,
            token_indices,
            assign_base,
            0,
            valid_rows,
            b_pack_stage_sh[0],
            rank_idx,
            scale,
            s_vals,
            gate_mat_stage_desc_sh[0]);
    }
    __syncthreads();

    for (int stage_ordinal = 0; ; ++stage_ordinal) {
        const int cur = stage_ordinal & 1;
        const int nxt = cur ^ 1;
        const GateStageDesc cur_desc = gate_mat_stage_desc_sh[cur];
        if (!cur_desc.valid) {
            break;
        }

        if (warp_id < GATE_TC_COMPUTE_WARPS) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, GATE_TC_M, GATE_TC_N, GATE_TC_K, half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::load_matrix_sync(a_frag, a_stage_sh[cur], GATE_TC_K);

            const int chunk0 = warp_id;
            if (chunk0 < cur_desc.n_chunks) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, GATE_TC_M, GATE_TC_N, GATE_TC_K, half, nvcuda::wmma::row_major> b_frag0;
                nvcuda::wmma::load_matrix_sync(
                    b_frag0,
                    &b_stage_sh[cur][chunk0 * GATE_TC_N],
                    N_TILE);
                nvcuda::wmma::mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
            }

            const int chunk1 = warp_id + GATE_TC_COMPUTE_WARPS;
            if (chunk1 < cur_desc.n_chunks) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, GATE_TC_M, GATE_TC_N, GATE_TC_K, half, nvcuda::wmma::row_major> b_frag1;
                nvcuda::wmma::load_matrix_sync(
                    b_frag1,
                    &b_stage_sh[cur][chunk1 * GATE_TC_N],
                    N_TILE);
                nvcuda::wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
            }
        } else if (is_loader_warp) {
            if (prefetch_desc.valid) {
                const GateStageDesc future_desc = find_next_gate_stage_desc(
                    slab_meta,
                    tile_meta,
                    num_slabs,
                    num_tiles_total,
                    tile_local,
                    prefetch_desc.slab_i,
                    prefetch_desc.stage_idx + 1);
                if (future_desc.valid) {
                    prefetch_gate_pack_desc_async(
                        loader_tid,
                        loader_stride,
                        b_pack_stage_sh[(stage_ordinal + 2) % 3],
                        payload,
                        future_desc);
                    cp_async_wait_group_1();
                } else {
                    cp_async_wait_group_0();
                }
                materialize_gate_stage_from_desc(
                    loader_tid,
                    loader_stride,
                    a_stage_sh[nxt],
                    b_stage_sh[nxt],
                    h,
                    h_rank,
                    token_indices,
                    assign_base,
                    0,
                    valid_rows,
                    b_pack_stage_sh[(stage_ordinal + 1) % 3],
                    rank_idx,
                    scale,
                    s_vals,
                    prefetch_desc);
                if (tid == first_loader_tid) {
                    gate_mat_stage_desc_sh[nxt] = prefetch_desc;
                }
                prefetch_desc = future_desc;
            } else if (tid == first_loader_tid) {
                gate_mat_stage_desc_sh[nxt] = make_invalid_gate_stage_desc();
            }
        }

        __syncthreads();
    }

    if (warp_id < GATE_TC_COMPUTE_WARPS) {
        const int chunk0 = warp_id;
        if (chunk0 < 8) {
            nvcuda::wmma::store_matrix_sync(
                c_store_sh[chunk0],
                acc_frag0,
                GATE_TC_N,
                nvcuda::wmma::mem_row_major);
        }
        const int chunk1 = warp_id + GATE_TC_COMPUTE_WARPS;
        if (chunk1 < 8) {
            nvcuda::wmma::store_matrix_sync(
                c_store_sh[chunk1],
                acc_frag1,
                GATE_TC_N,
                nvcuda::wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < valid_rows * N_TILE; idx += blockDim.x) {
        const int row = idx / N_TILE;
        const int col_in_tile = idx - row * N_TILE;
        const int col = col_tile_base + col_in_tile;
        if (col >= intermediate_size) {
            continue;
        }
        const int chunk = col_in_tile / GATE_TC_N;
        const int col_local = col_in_tile - chunk * GATE_TC_N;
        const float acc = c_store_sh[chunk][row * GATE_TC_N + col_local];
        const int a = assign_base + row;
        out[static_cast<int64_t>(a) * intermediate_size + col] = acc;
    }
}

template<int STATIC_ROWS>
__device__ __forceinline__ void gate_up_small_warp_accumulate(
    const half* __restrict__ h,
    int32_t h_rank,
    const int32_t* __restrict__ token_indices,
    const uint32_t* __restrict__ payload,
    const int32_t* __restrict__ rank_idx,
    const int32_t* __restrict__ tile_meta,
    const int32_t* __restrict__ slab_meta,
    const half* __restrict__ scale,
    const half* __restrict__ s_vals,
    int32_t num_slabs,
    int32_t num_tiles_total,
    int32_t assign_base,
    int32_t valid_rows,
    int32_t tile_local,
    int32_t subtile,
    int32_t intermediate_size,
    uint32_t pack_stage_sh[2][GATE_SMALL_PACK_WORDS_MAX],
    float (&acc_out)[GATE_SMALL_GROUP],
    int32_t& col,
    bool& col_in_bounds) {
    constexpr int MAX_ROWS = GATE_SMALL_GROUP;
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int rows_rt = (STATIC_ROWS > 0) ? STATIC_ROWS : valid_rows;
    const int col_in_tile = subtile * N_MICRO + lane;
    col = tile_local * N_TILE + col_in_tile;
    col_in_bounds = col < intermediate_size;

    const int token_lane = (lane < rows_rt) ? token_indices[assign_base + lane] : 0;
    int tokens[MAX_ROWS] = {0, 0, 0, 0};
    float acc[MAX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int a_local = 0; a_local < MAX_ROWS; ++a_local) {
        if constexpr (STATIC_ROWS > 0) {
            if (a_local < STATIC_ROWS) {
                tokens[a_local] = __shfl_sync(mask, token_lane, a_local);
            }
        } else if (a_local < valid_rows) {
            tokens[a_local] = __shfl_sync(mask, token_lane, a_local);
        }
    }

    for (int slab_i = 0; slab_i < num_slabs; ++slab_i) {
        const int bits = slab_meta[slab_i * 4 + 0];
        const int k_slab = slab_meta[slab_i * 4 + 1];
        const int rank_off = slab_meta[slab_i * 4 + 2];
        const int tile_off = slab_meta[slab_i * 4 + 3];
        const int tile_end = (slab_i + 1 < num_slabs)
            ? slab_meta[(slab_i + 1) * 4 + 3]
            : num_tiles_total;

        const int t = tile_off + tile_local;
        if (t >= tile_end) {
            continue;
        }

        const int pack_off = tile_meta[t * 3 + 0];
        const int scale_off = tile_meta[t * 3 + 1];
        const int n_valid = tile_meta[t * 3 + 2];
        const bool col_active = col_in_bounds && (col_in_tile < n_valid);

        const int words_per_row = (bits == 16) ? 16 : bits;
        const int stage_count = (k_slab + K_STAGE - 1) / K_STAGE;
        if (stage_count <= 0) {
            continue;
        }
        int cur = 0;
        int cur_stage_row_base = 0;
        int cur_rows = min(K_STAGE, k_slab);
        prefetch_gate_pack_micro_stage_async(
            lane,
            pack_stage_sh[cur],
            payload,
            pack_off,
            /*stage_prefix_words=*/0,
            cur_rows,
            words_per_row,
            subtile);
        cp_async_wait_group_0();
        __syncwarp(mask);
        for (int s = 0; s < stage_count; ++s) {
            const int nxt = cur ^ 1;
            const int next_stage_idx = s + 1;
            int next_stage_row_base = 0;
            int next_rows = 0;
            if (next_stage_idx < stage_count) {
                next_stage_row_base = next_stage_idx * K_STAGE;
                next_rows = min(K_STAGE, k_slab - next_stage_row_base);
                const int next_stage_prefix_words =
                    next_stage_row_base * MICROS_PER_TILE * words_per_row;
                prefetch_gate_pack_micro_stage_async(
                    lane,
                    pack_stage_sh[nxt],
                    payload,
                    pack_off,
                    next_stage_prefix_words,
                    next_rows,
                    words_per_row,
                    subtile);
            }

            int rk_lane = 0;
            float s_lane = 0.0f;
            float sc_lane = 1.0f;
            if (lane < cur_rows) {
                const int lr = rank_off + cur_stage_row_base + lane;
                rk_lane = rank_idx[lr];
                s_lane = __half2float(s_vals[lr]);
                sc_lane = (bits == 16)
                    ? 1.0f
                    : __half2float(scale[scale_off + cur_stage_row_base + lane]);
            }

            #pragma unroll
            for (int r = 0; r < K_STAGE; ++r) {
                if (r >= cur_rows) {
                    break;
                }
                const int rk = __shfl_sync(mask, rk_lane, r);
                const float s_row = __shfl_sync(mask, s_lane, r);

                float hs_lane = 0.0f;
                if (lane < rows_rt) {
                    hs_lane =
                        __half2float(h[static_cast<int64_t>(tokens[lane]) * h_rank + rk]) * s_row;
                }

                float w = decode_tile_weight_stage_from_pack_micro_warp(
                    pack_stage_sh[cur],
                    bits,
                    r);
                if (bits != 16) {
                    w *= __shfl_sync(mask, sc_lane, r);
                }
                if (!col_active) {
                    w = 0.0f;
                }

                #pragma unroll
                for (int a_local = 0; a_local < MAX_ROWS; ++a_local) {
                    if (a_local >= rows_rt) {
                        break;
                    }
                    acc[a_local] += __shfl_sync(mask, hs_lane, a_local) * w;
                }
            }
            if (next_stage_idx < stage_count) {
                cp_async_wait_group_0();
                __syncwarp(mask);
                cur = nxt;
                cur_stage_row_base = next_stage_row_base;
                cur_rows = next_rows;
            }
        }
    }

    #pragma unroll
    for (int a_local = 0; a_local < MAX_ROWS; ++a_local) {
        acc_out[a_local] = acc[a_local];
    }
}

struct GateUpSmallDeviceMeta {
    const int64_t* payload_ptrs;
    const int64_t* rank_idx_ptrs;
    const int64_t* tile_meta_ptrs;
    const int64_t* slab_meta_ptrs;
    const int64_t* scale_ptrs;
    const int64_t* s_ptrs;
    const int32_t* num_slabs;
    const int32_t* num_tiles;
};

template<int CTA_SUBTILES>
__device__ __forceinline__ void gate_up_small_cta(
    const half* __restrict__ h,
    int32_t h_rank,
    const int32_t* __restrict__ token_indices,
    GateUpSmallDeviceMeta meta,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_assign_base,
    const int32_t* __restrict__ task_assign_count,
    int32_t num_tasks,
    int32_t intermediate_size,
    int32_t subtile_base,
    uint32_t (&pack_stage_sh)[CTA_SUBTILES][2][GATE_SMALL_PACK_WORDS_MAX],
    float* __restrict__ out) {
    const int task = blockIdx.x;
    if (task >= num_tasks) {
        return;
    }

    const int warp_id = threadIdx.x >> 5;
    if (warp_id >= CTA_SUBTILES) {
        return;
    }

    const int tile_local = blockIdx.y;
    const int subtile = subtile_base + warp_id;
    if (subtile >= MICROS_PER_TILE) {
        return;
    }
    const int expert = task_expert[task];
    const int assign_base = task_assign_base[task];
    const int valid_rows = task_assign_count[task];
    if (valid_rows <= 0 || valid_rows > GATE_SMALL_GROUP) {
        return;
    }

    const auto* payload =
        reinterpret_cast<const uint32_t*>(meta.payload_ptrs[expert]);
    const auto* rank_idx =
        reinterpret_cast<const int32_t*>(meta.rank_idx_ptrs[expert]);
    const auto* tile_meta =
        reinterpret_cast<const int32_t*>(meta.tile_meta_ptrs[expert]);
    const auto* slab_meta =
        reinterpret_cast<const int32_t*>(meta.slab_meta_ptrs[expert]);
    const auto* scale =
        reinterpret_cast<const half*>(meta.scale_ptrs[expert]);
    const auto* s_vals =
        reinterpret_cast<const half*>(meta.s_ptrs[expert]);

    const int num_slabs = meta.num_slabs[expert];
    const int num_tiles_total = meta.num_tiles[expert];

    float acc[GATE_SMALL_GROUP];
    int32_t col = 0;
    bool col_in_bounds = false;
    gate_up_small_warp_accumulate<0>(
        h,
        h_rank,
        token_indices,
        payload,
        rank_idx,
        tile_meta,
        slab_meta,
        scale,
        s_vals,
        num_slabs,
        num_tiles_total,
        assign_base,
        valid_rows,
        tile_local,
        subtile,
        intermediate_size,
        pack_stage_sh[warp_id],
        acc,
        col,
        col_in_bounds);

    if (col_in_bounds) {
        #pragma unroll
        for (int a_local = 0; a_local < GATE_SMALL_GROUP; ++a_local) {
            if (a_local >= valid_rows) {
                break;
            }
            out[static_cast<int64_t>(assign_base + a_local) * intermediate_size + col] =
                acc[a_local];
        }
    }
}

__global__ void gate_up_small_kernel(
    const half* __restrict__ h,
    int32_t h_rank,
    const int32_t* __restrict__ token_indices,
    GateUpSmallDeviceMeta meta,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_assign_base,
    const int32_t* __restrict__ task_assign_count,
    int32_t num_tasks,
    int32_t intermediate_size,
    float* __restrict__ out) {
    __shared__ __align__(16)
        uint32_t pack_stage_sh[GATE_SMALL_SUBTILES][2][GATE_SMALL_PACK_WORDS_MAX];

    gate_up_small_cta<GATE_SMALL_SUBTILES>(
        h,
        h_rank,
        token_indices,
        meta,
        task_expert,
        task_assign_base,
        task_assign_count,
        num_tasks,
        intermediate_size,
        /*subtile_base=*/0,
        pack_stage_sh,
        out);
}

__global__ void gate_up_small_dual_kernel(
    const half* __restrict__ gate_h,
    int32_t gate_h_rank,
    GateUpSmallDeviceMeta gate_meta,
    const half* __restrict__ up_h,
    int32_t up_h_rank,
    GateUpSmallDeviceMeta up_meta,
    const int32_t* __restrict__ token_indices,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_assign_base,
    const int32_t* __restrict__ task_assign_count,
    int32_t num_tasks,
    int32_t intermediate_size,
    float* __restrict__ gate_out,
    float* __restrict__ up_out) {
    const int z = static_cast<int>(blockIdx.z);
    if (z > 3) {
        return;
    }

    const int mtype = z >> 1;
    const int subtile_group = z & 1;
    const int subtile_base = subtile_group * GATE_SMALL_MICROS_PER_CTA;
    const bool do_up = (mtype == 1);
    const half* h = do_up ? up_h : gate_h;
    const int32_t h_rank = do_up ? up_h_rank : gate_h_rank;
    const GateUpSmallDeviceMeta meta = do_up ? up_meta : gate_meta;
    float* out = do_up ? up_out : gate_out;

    __shared__ __align__(16)
        uint32_t pack_stage_sh[GATE_SMALL_MICROS_PER_CTA][2][GATE_SMALL_PACK_WORDS_MAX];

    gate_up_small_cta<GATE_SMALL_MICROS_PER_CTA>(
        h,
        h_rank,
        token_indices,
        meta,
        task_expert,
        task_assign_base,
        task_assign_count,
        num_tasks,
        intermediate_size,
        subtile_base,
        pack_stage_sh,
        out);
}

__global__ void gate_up_small_dual_kernel_legacy(
    const half* __restrict__ gate_h,
    int32_t gate_h_rank,
    GateUpSmallDeviceMeta gate_meta,
    const half* __restrict__ up_h,
    int32_t up_h_rank,
    GateUpSmallDeviceMeta up_meta,
    const int32_t* __restrict__ token_indices,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_assign_base,
    const int32_t* __restrict__ task_assign_count,
    int32_t num_tasks,
    int32_t intermediate_size,
    float* __restrict__ gate_out,
    float* __restrict__ up_out) {
    if (blockIdx.z > 1) {
        return;
    }

    const bool do_up = (blockIdx.z == 1);
    const half* h = do_up ? up_h : gate_h;
    const int32_t h_rank = do_up ? up_h_rank : gate_h_rank;
    const GateUpSmallDeviceMeta meta = do_up ? up_meta : gate_meta;
    float* out = do_up ? up_out : gate_out;

    __shared__ __align__(16)
        uint32_t pack_stage_sh[GATE_SMALL_SUBTILES][2][GATE_SMALL_PACK_WORDS_MAX];

    gate_up_small_cta<GATE_SMALL_SUBTILES>(
        h,
        h_rank,
        token_indices,
        meta,
        task_expert,
        task_assign_base,
        task_assign_count,
        num_tasks,
        intermediate_size,
        /*subtile_base=*/0,
        pack_stage_sh,
        out);
}

struct DownTileDesc {
    int valid;
    int pack_off;
    int scale_off;
    int n_valid;
    int col_base;
};

struct alignas(16) DownSharedStorage {
    // Double-buffered Z/W MMA staging.
    half z_mma_stage[2][N_TILE * DOWN_TC_A_BATCH];
    uint32_t w_pack_raw[2][DOWN_TC_COMPUTE_WARPS][GATE_PACK_WORDS_MAX];
    half w_scale_raw[2][DOWN_TC_COMPUTE_WARPS][DOWN_TC_ROWS_PER_WARP];
    half w_mma_stage[2][DOWN_TC_COMPUTE_WARPS][DOWN_TC_ROWS_PER_WARP * N_TILE];
    // Per-warp temporary C tile and batch-owner accumulation (no CTA-wide acc pool).
    float c_store[DOWN_TC_COMPUTE_WARPS][DOWN_TC_M * DOWN_TC_N];
    float batch_acc[DOWN_TC_COMPUTE_WARPS][DOWN_TC_A_GROUP_BATCHES][DOWN_TC_M * DOWN_TC_N];
    // Row metadata reused across tile/batch traversal.
    int logical_rank[N_TILE];
    float s_val[N_TILE];
    int row_valid[N_TILE];
};

static inline size_t down_kernel_shared_bytes() {
    return sizeof(DownSharedStorage);
}

__device__ __forceinline__ DownTileDesc make_invalid_down_tile_desc() {
    DownTileDesc desc{};
    desc.valid = 0;
    return desc;
}

__device__ __forceinline__ DownTileDesc make_down_tile_desc(
    const int32_t* __restrict__ tile_meta,
    int tile_off,
    int tile_count,
    int tile_idx) {
    if (tile_idx < 0 || tile_idx >= tile_count) {
        return make_invalid_down_tile_desc();
    }
    const int t = tile_off + tile_idx;
    DownTileDesc desc{};
    desc.valid = 1;
    desc.pack_off = tile_meta[t * 3 + 0];
    desc.scale_off = tile_meta[t * 3 + 1];
    desc.n_valid = tile_meta[t * 3 + 2];
    desc.col_base = tile_idx * N_TILE;
    return desc;
}

__device__ __forceinline__ void materialize_down_z_stage_from_gmem(
    int lane_tid,
    int lane_stride,
    half* __restrict__ z_stage,
    const float* __restrict__ gate_out,
    const float* __restrict__ up_out,
    int32_t assign_base,
    int32_t assign_count,
    int32_t intermediate_size,
    int32_t col_base,
    int32_t n_valid) {
    for (int idx = lane_tid; idx < N_TILE * DOWN_TC_A_BATCH; idx += lane_stride) {
        const int k = idx / DOWN_TC_A_BATCH;
        const int n = idx - k * DOWN_TC_A_BATCH;
        half v = __float2half(0.0f);
        if (n < assign_count && k < n_valid) {
            const int64_t base_idx =
                static_cast<int64_t>(assign_base + n) * intermediate_size + col_base + k;
            const float z = silu_f32(gate_out[base_idx]) * up_out[base_idx];
            v = __float2half(z);
        }
        z_stage[idx] = v;
    }
}

__device__ __forceinline__ void prefetch_down_weight_stage_async(
    int lane_tid,
    int lane_stride,
    uint32_t* __restrict__ pack_stage,
    half* __restrict__ scale_stage,
    const uint32_t* __restrict__ payload,
    const half* __restrict__ scale,
    int32_t bits,
    int32_t pack_off,
    int32_t scale_off,
    int32_t row_batch_base,
    int32_t row_count,
    int32_t local_row_base,
    int32_t rank_off) {
    const int words_per_row = (bits == 16) ? 16 : bits;
    bool issued = false;
    for (int w = 0; w < DOWN_TC_COMPUTE_WARPS; ++w) {
        uint32_t* warp_pack =
            pack_stage + static_cast<int64_t>(w) * GATE_PACK_WORDS_MAX;
        half* warp_scale =
            scale_stage + static_cast<int64_t>(w) * DOWN_TC_ROWS_PER_WARP;
        const int row_block_base = row_batch_base + w * DOWN_TC_ROWS_PER_WARP;
        const int rows_cur = max(0, min(DOWN_TC_ROWS_PER_WARP, row_count - row_block_base));
        if (rows_cur <= 0) {
            continue;
        }
        const int row_in_slab_base = local_row_base + row_block_base - rank_off;
        const int stage_prefix_words =
            row_in_slab_base * MICROS_PER_TILE * words_per_row;
        const int stage_words = rows_cur * MICROS_PER_TILE * words_per_row;
        const int vec_words = 4;
        const int vec_count = stage_words / vec_words;
        for (int v = lane_tid; v < vec_count; v += lane_stride) {
            const int idx = v * vec_words;
            cp_async_cg_16(
                &warp_pack[idx],
                &payload[pack_off + stage_prefix_words + idx]);
            issued = true;
        }
        for (int idx = vec_count * vec_words + lane_tid; idx < stage_words; idx += lane_stride) {
            warp_pack[idx] = payload[pack_off + stage_prefix_words + idx];
        }
        for (int r = lane_tid; r < rows_cur; r += lane_stride) {
            warp_scale[r] = scale[scale_off + row_in_slab_base + r];
        }
    }
    if (issued) {
        cp_async_commit_group();
    }
}

template<int BITS>
__device__ __forceinline__ void materialize_down_weight_stage_warp_bits_from_pack(
    int lane,
    int compute_warp_id,
    int row_batch_base,
    int row_count,
    int n_valid,
    const uint32_t* __restrict__ pack_stage,
    const half* __restrict__ scale_stage,
    const int* __restrict__ row_valid_sh,
    half* __restrict__ w_stage) {
    half* warp_w =
        w_stage + static_cast<int64_t>(compute_warp_id) * DOWN_TC_ROWS_PER_WARP * N_TILE;
    const uint32_t* warp_pack =
        pack_stage + static_cast<int64_t>(compute_warp_id) * GATE_PACK_WORDS_MAX;
    const half* warp_scale =
        scale_stage + static_cast<int64_t>(compute_warp_id) * DOWN_TC_ROWS_PER_WARP;
    const int row_block_base = row_batch_base + compute_warp_id * DOWN_TC_ROWS_PER_WARP;
    const int rows_cur = max(0, min(DOWN_TC_ROWS_PER_WARP, row_count - row_block_base));

    #pragma unroll
    for (int r = 0; r < DOWN_TC_ROWS_PER_WARP; ++r) {
        const int row_rel = row_block_base + r;
        const bool row_ok = r < rows_cur && row_rel < row_count && row_valid_sh[row_rel];
        const float scale_row = (BITS == 16 || !row_ok)
            ? 1.0f
            : __half2float(warp_scale[r]);
        #pragma unroll
        for (int micro = 0; micro < MICROS_PER_TILE; ++micro) {
            const int c = micro * N_MICRO + lane;
            float w = 0.0f;
            if (row_ok && c < n_valid) {
                w = decode_tile_weight_stage_from_pack_warp(
                    warp_pack,
                    BITS,
                    rows_cur,
                    r,
                    c);
                if constexpr (BITS != 16) {
                    w *= scale_row;
                }
            }
            warp_w[r * N_TILE + c] = __float2half(w);
        }
    }
}

__device__ __forceinline__ void materialize_down_weight_stage_warp_runtime_from_pack(
    int bits,
    int lane,
    int compute_warp_id,
    int row_batch_base,
    int row_count,
    int n_valid,
    const uint32_t* __restrict__ pack_stage,
    const half* __restrict__ scale_stage,
    const int* __restrict__ row_valid_sh,
    half* __restrict__ w_stage) {
    switch (bits) {
        case 1:
            materialize_down_weight_stage_warp_bits_from_pack<1>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        case 2:
            materialize_down_weight_stage_warp_bits_from_pack<2>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        case 3:
            materialize_down_weight_stage_warp_bits_from_pack<3>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        case 4:
            materialize_down_weight_stage_warp_bits_from_pack<4>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        case 6:
            materialize_down_weight_stage_warp_bits_from_pack<6>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        case 8:
            materialize_down_weight_stage_warp_bits_from_pack<8>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        case 16:
            materialize_down_weight_stage_warp_bits_from_pack<16>(
                lane, compute_warp_id, row_batch_base, row_count, n_valid,
                pack_stage, scale_stage, row_valid_sh, w_stage);
            break;
        default: {
            half* warp_w =
                w_stage + static_cast<int64_t>(compute_warp_id) * DOWN_TC_ROWS_PER_WARP * N_TILE;
            #pragma unroll
            for (int r = 0; r < DOWN_TC_ROWS_PER_WARP; ++r) {
                #pragma unroll
                for (int micro = 0; micro < MICROS_PER_TILE; ++micro) {
                    warp_w[r * N_TILE + micro * N_MICRO + lane] = __float2half(0.0f);
                }
            }
            break;
        }
    }
}

__device__ __forceinline__ void materialize_down_weight_stage_runtime_from_pack_by_loader(
    int bits,
    int lane,
    int loader_warp_id,
    int row_batch_base,
    int row_count,
    int n_valid,
    const uint32_t* __restrict__ pack_stage,
    const half* __restrict__ scale_stage,
    const int* __restrict__ row_valid_sh,
    half* __restrict__ w_stage) {
    const int loader_groups = max(1, DOWN_TC_LOADER_WARPS);
    for (int compute_warp_id = loader_warp_id;
         compute_warp_id < DOWN_TC_COMPUTE_WARPS;
         compute_warp_id += loader_groups) {
        const int row_block_base = row_batch_base + compute_warp_id * DOWN_TC_ROWS_PER_WARP;
        if (row_block_base >= row_count) {
            continue;
        }
        materialize_down_weight_stage_warp_runtime_from_pack(
            bits,
            lane,
            compute_warp_id,
            row_batch_base,
            row_count,
            n_valid,
            pack_stage,
            scale_stage,
            row_valid_sh,
            w_stage);
    }
}

__global__ void down_kernel(
    const float* __restrict__ gate_out,
    const float* __restrict__ up_out,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ token_indices,
    const int32_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ payload_ptrs,
    const int64_t* __restrict__ rank_idx_ptrs,
    const int64_t* __restrict__ tile_meta_ptrs,
    const int64_t* __restrict__ slab_meta_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    const int64_t* __restrict__ s_ptrs,
    const int32_t* __restrict__ num_slabs_ptr,
    const int32_t* __restrict__ num_tiles_ptr,
    const int32_t* __restrict__ rank_lens_ptr,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_row_start,
    const int32_t* __restrict__ task_row_count,
    const int32_t* __restrict__ task_rank_off,
    const int32_t* __restrict__ task_k_slab,
    const int32_t* __restrict__ task_tile_off,
    const int32_t* __restrict__ task_tile_end,
    const int32_t* __restrict__ task_bits,
    int32_t num_tasks,
    int32_t intermediate_size,
    int32_t rank_out,
    float* __restrict__ rank_accum) {
    const int cta = blockIdx.x;
    if (cta >= num_tasks) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const bool is_compute_warp = warp_id < DOWN_TC_COMPUTE_WARPS;
    const bool is_loader_warp = warp_id >= DOWN_TC_COMPUTE_WARPS;
    const int loader_warp_id = warp_id - DOWN_TC_COMPUTE_WARPS;
    const int loader_tid = tid - DOWN_TC_COMPUTE_WARPS * 32;
    const int loader_stride = max(1, DOWN_TC_LOADER_WARPS * 32);
    extern __shared__ __align__(16) unsigned char down_shared_raw[];
    auto& sh = *reinterpret_cast<DownSharedStorage*>(down_shared_raw);

    const int expert = task_expert[cta];
    const int local_row_base = task_row_start[cta];
    const int row_count = task_row_count[cta];
    const int rank_off = task_rank_off[cta];
    const int k_slab = task_k_slab[cta];
    const int tile_off = task_tile_off[cta];
    const int tile_end = task_tile_end[cta];
    const int bits = task_bits[cta];

    const auto* payload =
        reinterpret_cast<const uint32_t*>(payload_ptrs[expert]);
    const auto* rank_idx =
        reinterpret_cast<const int32_t*>(rank_idx_ptrs[expert]);
    const auto* tile_meta =
        reinterpret_cast<const int32_t*>(tile_meta_ptrs[expert]);
    const auto* scale =
        reinterpret_cast<const half*>(scale_ptrs[expert]);
    const auto* s_vals =
        reinterpret_cast<const half*>(s_ptrs[expert]);

    // Down tasks are already slab-scoped row chunks, so lookahead is kept inside the
    // owning slab's tile sequence; there is no same-CTA cross-slab consumer to prefetch for.
    (void)slab_meta_ptrs;
    (void)num_slabs_ptr;
    (void)num_tiles_ptr;
    (void)k_slab;

    const int local_total = rank_lens_ptr[expert];
    if (local_row_base >= local_total) {
        return;
    }

    const int assign_begin = expert_offsets[expert];
    const int assign_end = expert_offsets[expert + 1];
    const int assign_total = assign_end - assign_begin;
    if (assign_total <= 0) {
        return;
    }

    if (tid < row_count) {
        const int local_row = local_row_base + tid;
        const int logical_rank = rank_idx[local_row];
        sh.logical_rank[tid] = logical_rank;
        sh.row_valid[tid] = (logical_rank >= 0 && logical_rank < rank_out) ? 1 : 0;
        sh.s_val[tid] = __half2float(s_vals[local_row]);
    } else if (tid < N_TILE) {
        sh.row_valid[tid] = 0;
    }
    __syncthreads();

    const int tile_count = tile_end - tile_off;
    if (tile_count <= 0) {
        return;
    }

    const int a_group_span =
        (assign_total < DOWN_TC_SMALL_ASSIGN_THRESHOLD) ? DOWN_TC_A_BATCH : DOWN_TC_A_GROUP;

    for (int row_batch_base = 0; row_batch_base < row_count; row_batch_base += DOWN_TC_ROW_BATCH) {
        // `a_group` only caps the local accumulation working set. Inside each group the
        // actual execution order is tile-major and assignment-batched.
        for (int a_group_begin = assign_begin; a_group_begin < assign_end; a_group_begin += a_group_span) {
            const int group_assign_count = min(a_group_span, assign_end - a_group_begin);
            const int group_batch_count =
                (group_assign_count + DOWN_TC_A_BATCH - 1) / DOWN_TC_A_BATCH;

            const int compute_warp_id = warp_id;
            const int row_block_base =
                row_batch_base + compute_warp_id * DOWN_TC_ROWS_PER_WARP;
            const bool row_block_active =
                is_compute_warp && row_block_base < row_count;

            if (row_block_active && lane < DOWN_TC_A_BATCH) {
                #pragma unroll
                for (int b = 0; b < DOWN_TC_A_GROUP_BATCHES; ++b) {
                    #pragma unroll
                    for (int r = 0; r < DOWN_TC_ROWS_PER_WARP; ++r) {
                        sh.batch_acc[compute_warp_id][b][r * DOWN_TC_A_BATCH + lane] = 0.0f;
                    }
                }
            }

            const DownTileDesc first_tile_desc =
                make_down_tile_desc(tile_meta, tile_off, tile_count, 0);
            const int first_batch_count = min(DOWN_TC_A_BATCH, group_assign_count);

            if (is_loader_warp) {
                prefetch_down_weight_stage_async(
                    loader_tid,
                    loader_stride,
                    &sh.w_pack_raw[0][0][0],
                    &sh.w_scale_raw[0][0][0],
                    payload,
                    scale,
                    bits,
                    first_tile_desc.pack_off,
                    first_tile_desc.scale_off,
                    row_batch_base,
                    row_count,
                    local_row_base,
                    rank_off);
                materialize_down_z_stage_from_gmem(
                    loader_tid,
                    loader_stride,
                    sh.z_mma_stage[0],
                    gate_out,
                    up_out,
                    a_group_begin,
                    first_batch_count,
                    intermediate_size,
                    first_tile_desc.col_base,
                    first_tile_desc.n_valid);
                cp_async_wait_group_0();
            }
            // cp.async wait is per-thread; CTA sync is required before any warp decodes
            // pack/scale data that may have been fetched by other loader warps.
            __syncthreads();
            if (is_loader_warp) {
                materialize_down_weight_stage_runtime_from_pack_by_loader(
                    bits,
                    lane,
                    loader_warp_id,
                    row_batch_base,
                    row_count,
                    first_tile_desc.n_valid,
                    &sh.w_pack_raw[0][0][0],
                    &sh.w_scale_raw[0][0][0],
                    sh.row_valid,
                    &sh.w_mma_stage[0][0][0]);
            }
            __syncthreads();

            // Work is traversed as:
            //   tile 0 -> batch 0..B-1
            //   tile 1 -> batch 0..B-1
            // which preserves the requested `row_batch -> tile -> a_batch` reuse order
            // within the current accumulation working set.
            const int total_works = tile_count * group_batch_count;
            int cur_z_stage = 0;
            int cur_w_stage = 0;

            for (int work = 0; work < total_works; ++work) {
                const int tile_idx = work / group_batch_count;
                const int batch_idx = work - tile_idx * group_batch_count;
                const DownTileDesc cur_tile_desc =
                    make_down_tile_desc(tile_meta, tile_off, tile_count, tile_idx);
                const int batch_local_base = batch_idx * DOWN_TC_A_BATCH;
                const int batch_assign_count =
                    min(DOWN_TC_A_BATCH, group_assign_count - batch_local_base);

                const int next_work = work + 1;
                const bool has_next = next_work < total_works;
                int next_tile_idx = -1;
                int next_batch_idx = -1;
                int next_batch_local_base = 0;
                int next_batch_assign_count = 0;
                DownTileDesc next_tile_desc = make_invalid_down_tile_desc();
                const int next_z_stage = cur_z_stage ^ 1;
                const int next_w_stage = cur_w_stage ^ 1;
                bool next_needs_weight = false;

                if (has_next) {
                    next_tile_idx = next_work / group_batch_count;
                    next_batch_idx = next_work - next_tile_idx * group_batch_count;
                    next_batch_local_base = next_batch_idx * DOWN_TC_A_BATCH;
                    next_batch_assign_count =
                        min(DOWN_TC_A_BATCH, group_assign_count - next_batch_local_base);
                    next_tile_desc =
                        make_down_tile_desc(tile_meta, tile_off, tile_count, next_tile_idx);
                    next_needs_weight = next_tile_idx != tile_idx;
                    if (is_loader_warp) {
                        // producer-materialize: z path consumes gmem directly and writes next stage.
                        materialize_down_z_stage_from_gmem(
                            loader_tid,
                            loader_stride,
                            sh.z_mma_stage[next_z_stage],
                            gate_out,
                            up_out,
                            a_group_begin + next_batch_local_base,
                            next_batch_assign_count,
                            intermediate_size,
                            next_tile_desc.col_base,
                            next_tile_desc.n_valid);
                        if (next_needs_weight) {
                            // producer-load: only tile transitions need a new down-weight stage.
                            prefetch_down_weight_stage_async(
                                loader_tid,
                                loader_stride,
                                &sh.w_pack_raw[next_w_stage][0][0],
                                &sh.w_scale_raw[next_w_stage][0][0],
                                payload,
                                scale,
                                bits,
                                next_tile_desc.pack_off,
                                next_tile_desc.scale_off,
                                row_batch_base,
                                row_count,
                                local_row_base,
                                rank_off);
                        }
                    }
                }

                if (row_block_active) {
                    // consumer-compute: consume current z/w stage while producer prepares next.
                    nvcuda::wmma::fragment<
                        nvcuda::wmma::accumulator,
                        DOWN_TC_M,
                        DOWN_TC_N,
                        DOWN_TC_K,
                        float> acc_frag;
                    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
                    const int n_chunks =
                        (cur_tile_desc.n_valid + DOWN_TC_K - 1) / DOWN_TC_K;
                    for (int chunk = 0; chunk < n_chunks; ++chunk) {
                        nvcuda::wmma::fragment<
                            nvcuda::wmma::matrix_a,
                            DOWN_TC_M,
                            DOWN_TC_N,
                            DOWN_TC_K,
                            half,
                            nvcuda::wmma::row_major> a_frag;
                        nvcuda::wmma::fragment<
                            nvcuda::wmma::matrix_b,
                            DOWN_TC_M,
                            DOWN_TC_N,
                            DOWN_TC_K,
                            half,
                            nvcuda::wmma::row_major> b_frag;
                        nvcuda::wmma::load_matrix_sync(
                            a_frag,
                            &sh.w_mma_stage[cur_w_stage][compute_warp_id][chunk * DOWN_TC_K],
                            N_TILE);
                        nvcuda::wmma::load_matrix_sync(
                            b_frag,
                            &sh.z_mma_stage[cur_z_stage][chunk * DOWN_TC_K * DOWN_TC_A_BATCH],
                            DOWN_TC_A_BATCH);
                        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                    }

                    nvcuda::wmma::store_matrix_sync(
                        sh.c_store[compute_warp_id],
                        acc_frag,
                        DOWN_TC_A_BATCH,
                        nvcuda::wmma::mem_row_major);
                    __syncwarp();
                    if (lane < DOWN_TC_A_BATCH) {
                        const int a_local = batch_local_base + lane;
                        if (a_local < group_assign_count) {
                            #pragma unroll
                            for (int r = 0; r < DOWN_TC_ROWS_PER_WARP; ++r) {
                                const int row_rel = row_block_base + r;
                                if (row_rel < row_count && sh.row_valid[row_rel]) {
                                    sh.batch_acc[compute_warp_id][batch_idx]
                                        [r * DOWN_TC_A_BATCH + lane] +=
                                        sh.c_store[compute_warp_id][r * DOWN_TC_A_BATCH + lane];
                                }
                            }
                        }
                    }
                }

                if (has_next) {
                    if (is_loader_warp) {
                        if (next_needs_weight) {
                            // Delay wait until right before weight materialization/consumption.
                            cp_async_wait_group_0();
                        }
                    }
                    if (next_needs_weight) {
                        // Same reason as initial stage: ensure all loader-warp async writes
                        // are visible before any loader warp starts weight materialization.
                        __syncthreads();
                    }
                    if (is_loader_warp) {
                        if (next_needs_weight) {
                            materialize_down_weight_stage_runtime_from_pack_by_loader(
                                bits,
                                lane,
                                loader_warp_id,
                                row_batch_base,
                                row_count,
                                next_tile_desc.n_valid,
                                &sh.w_pack_raw[next_w_stage][0][0],
                                &sh.w_scale_raw[next_w_stage][0][0],
                                sh.row_valid,
                                &sh.w_mma_stage[next_w_stage][0][0]);
                        }
                    }
                    __syncthreads();
                    cur_z_stage = next_z_stage;
                    if (next_needs_weight) {
                        cur_w_stage = next_w_stage;
                    }
                }
            }

            if (row_block_active && lane < DOWN_TC_A_BATCH) {
                #pragma unroll
                for (int b = 0; b < DOWN_TC_A_GROUP_BATCHES; ++b) {
                    if (b >= group_batch_count) {
                        break;
                    }
                    const int a_local = b * DOWN_TC_A_BATCH + lane;
                    if (a_local >= group_assign_count) {
                        continue;
                    }
                    const int a = a_group_begin + a_local;
                    const int token = token_indices[a];
                    const float rw = route_weights[a];
                    #pragma unroll
                    for (int r = 0; r < DOWN_TC_ROWS_PER_WARP; ++r) {
                        const int row_rel = row_block_base + r;
                        if (row_rel >= row_count || !sh.row_valid[row_rel]) {
                            continue;
                        }
                        const float acc =
                            sh.batch_acc[compute_warp_id][b][r * DOWN_TC_A_BATCH + lane];
                        if (acc == 0.0f) {
                            continue;
                        }
                        const int logical_rank = sh.logical_rank[row_rel];
                        const float s_val = sh.s_val[row_rel];
                        atomicAdd(
                            &rank_accum[static_cast<int64_t>(token) * rank_out + logical_rank],
                            acc * s_val * rw);
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void materialize_down_small_z_stage(
    int lane_tid,
    int lane_stride,
    float* __restrict__ z_stage,
    const float* __restrict__ gate_out,
    const float* __restrict__ up_out,
    int32_t assign_base,
    int32_t assign_count,
    int32_t intermediate_size,
    int32_t col_base,
    int32_t n_valid) {
    for (int idx = lane_tid; idx < DOWN_SMALL_A_GROUP * N_TILE; idx += lane_stride) {
        const int a_local = idx / N_TILE;
        const int k = idx - a_local * N_TILE;
        float z = 0.0f;
        if (a_local < assign_count && k < n_valid) {
            const int64_t z_idx =
                static_cast<int64_t>(assign_base + a_local) * intermediate_size + col_base + k;
            z = silu_f32(gate_out[z_idx]) * up_out[z_idx];
        }
        z_stage[idx] = z;
    }
}

__device__ __forceinline__ void prefetch_down_small_weight_row_stage(
    int lane,
    uint32_t* __restrict__ row_pack_stage,
    float* __restrict__ row_scale_stage,
    int* __restrict__ row_decode_valid_stage,
    const uint32_t* __restrict__ payload,
    const half* __restrict__ scale,
    int32_t bits,
    int32_t k_slab,
    int32_t pack_off,
    int32_t scale_off,
    int32_t row_in_slab) {
    if (lane == 0) {
        *row_decode_valid_stage = 0;
        *row_scale_stage = 1.0f;
    }
    if (row_in_slab < 0 || row_in_slab >= k_slab) {
        return;
    }

    const int words_per_row = (bits == 16) ? 16 : bits;
    const int stage = row_in_slab / K_STAGE;
    const int row_in_stage = row_in_slab - stage * K_STAGE;
    const int rows_cur = min(K_STAGE, k_slab - stage * K_STAGE);
    if (rows_cur <= 0) {
        return;
    }

    const int stage_prefix = stage_prefix_words(k_slab, stage, words_per_row);
    for (int micro = 0; micro < MICROS_PER_TILE; ++micro) {
        const int src_base =
            pack_off
            + stage_prefix
            + micro * rows_cur * words_per_row
            + row_in_stage * words_per_row;
        const int dst_base = micro * words_per_row;
        for (int i = lane; i < words_per_row; i += 32) {
            row_pack_stage[dst_base + i] = payload[src_base + i];
        }
    }
    if (bits != 16 && lane == 0) {
        *row_scale_stage = __half2float(scale[scale_off + row_in_slab]);
    }
    if (lane == 0) {
        *row_decode_valid_stage = 1;
    }
}

__global__ void down_small_kernel(
    const float* __restrict__ gate_out,
    const float* __restrict__ up_out,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ token_indices,
    const int32_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ payload_ptrs,
    const int64_t* __restrict__ rank_idx_ptrs,
    const int64_t* __restrict__ tile_meta_ptrs,
    const int64_t* __restrict__ slab_meta_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    const int64_t* __restrict__ s_ptrs,
    const int32_t* __restrict__ num_slabs_ptr,
    const int32_t* __restrict__ num_tiles_ptr,
    const int32_t* __restrict__ rank_lens_ptr,
    const int32_t* __restrict__ task_expert,
    const int32_t* __restrict__ task_row_start,
    const int32_t* __restrict__ task_row_count,
    const int32_t* __restrict__ task_rank_off,
    const int32_t* __restrict__ task_k_slab,
    const int32_t* __restrict__ task_tile_off,
    const int32_t* __restrict__ task_tile_end,
    const int32_t* __restrict__ task_bits,
    int32_t num_tasks,
    int32_t intermediate_size,
    int32_t rank_out,
    float* __restrict__ rank_accum) {
    const int cta = blockIdx.x;
    if (cta >= num_tasks) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    const int expert = task_expert[cta];
    const int local_row_base = task_row_start[cta];
    const int row_count = task_row_count[cta];
    const int rank_off = task_rank_off[cta];
    const int k_slab = task_k_slab[cta];
    const int bits = task_bits[cta];
    const int tile_off = task_tile_off[cta];
    const int tile_end = task_tile_end[cta];

    const auto* payload =
        reinterpret_cast<const uint32_t*>(payload_ptrs[expert]);
    const auto* rank_idx =
        reinterpret_cast<const int32_t*>(rank_idx_ptrs[expert]);
    const auto* tile_meta =
        reinterpret_cast<const int32_t*>(tile_meta_ptrs[expert]);
    const auto* scale =
        reinterpret_cast<const half*>(scale_ptrs[expert]);
    const auto* s_vals =
        reinterpret_cast<const half*>(s_ptrs[expert]);

    (void)slab_meta_ptrs;
    (void)num_slabs_ptr;
    (void)num_tiles_ptr;
    const int local_total = rank_lens_ptr[expert];
    if (local_row_base >= local_total) {
        return;
    }

    const int assign_begin = expert_offsets[expert];
    const int assign_end = expert_offsets[expert + 1];
    const int assign_total = assign_end - assign_begin;
    if (assign_total <= 0) {
        return;
    }

    const int tile_count = tile_end - tile_off;
    if (tile_count <= 0) {
        return;
    }

    __shared__ int logical_rank_sh[N_TILE];
    __shared__ int row_valid_sh[N_TILE];
    __shared__ float row_s_sh[N_TILE];
    __shared__ int token_sh[DOWN_SMALL_A_GROUP];
    __shared__ float route_w_sh[DOWN_SMALL_A_GROUP];
    __shared__ __align__(16) float z_sh[2][DOWN_SMALL_A_GROUP][N_TILE];
    __shared__ uint32_t w_pack_sh[2][DOWN_SMALL_WARPS][DOWN_SMALL_PACK_WORDS_MAX];
    __shared__ float w_scale_sh[2][DOWN_SMALL_WARPS];
    __shared__ int w_decode_valid_sh[2][DOWN_SMALL_WARPS];
    __shared__ float row_acc_sh[DOWN_SMALL_A_GROUP][N_TILE];

    if (tid < row_count) {
        const int local_row = local_row_base + tid;
        const int logical_rank = rank_idx[local_row];
        logical_rank_sh[tid] = logical_rank;
        row_valid_sh[tid] = (logical_rank >= 0 && logical_rank < rank_out) ? 1 : 0;
        row_s_sh[tid] = __half2float(s_vals[local_row]);
    } else if (tid < N_TILE) {
        row_valid_sh[tid] = 0;
    }
    __syncthreads();

    for (int a_group_begin = assign_begin; a_group_begin < assign_end; a_group_begin += DOWN_SMALL_A_GROUP) {
        const int group_count = min(DOWN_SMALL_A_GROUP, assign_end - a_group_begin);
        for (int idx = tid; idx < group_count * row_count; idx += blockDim.x) {
            const int a_local = idx / row_count;
            const int row_local = idx - a_local * row_count;
            row_acc_sh[a_local][row_local] = 0.0f;
        }
        if (tid < DOWN_SMALL_A_GROUP) {
            if (tid < group_count) {
                const int a = a_group_begin + tid;
                token_sh[tid] = token_indices[a];
                route_w_sh[tid] = route_weights[a];
            }
        }
        __syncthreads();

        int cur_stage = 0;
        DownTileDesc cur_tile_desc =
            make_down_tile_desc(tile_meta, tile_off, tile_count, /*tile_idx=*/0);
        materialize_down_small_z_stage(
            tid,
            blockDim.x,
            &z_sh[cur_stage][0][0],
            gate_out,
            up_out,
            a_group_begin,
            group_count,
            intermediate_size,
            cur_tile_desc.col_base,
            cur_tile_desc.n_valid);
        __syncthreads();

        for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
            const int next_tile_idx = tile_idx + 1;
            const int next_stage = cur_stage ^ 1;
            DownTileDesc next_tile_desc = make_invalid_down_tile_desc();
            if (next_tile_idx < tile_count) {
                next_tile_desc =
                    make_down_tile_desc(tile_meta, tile_off, tile_count, next_tile_idx);
            }

            int row_stage = 0;
            prefetch_down_small_weight_row_stage(
                lane,
                &w_pack_sh[row_stage][warp_id][0],
                &w_scale_sh[row_stage][warp_id],
                &w_decode_valid_sh[row_stage][warp_id],
                payload,
                scale,
                bits,
                k_slab,
                cur_tile_desc.pack_off,
                cur_tile_desc.scale_off,
                local_row_base + warp_id - rank_off);
            __syncwarp();

            for (int row_block_base = 0; row_block_base < row_count; row_block_base += DOWN_SMALL_WARPS) {
                const int next_row_block_base = row_block_base + DOWN_SMALL_WARPS;
                const int next_row_stage = row_stage ^ 1;
                if (next_row_block_base < row_count) {
                    prefetch_down_small_weight_row_stage(
                        lane,
                        &w_pack_sh[next_row_stage][warp_id][0],
                        &w_scale_sh[next_row_stage][warp_id],
                        &w_decode_valid_sh[next_row_stage][warp_id],
                        payload,
                        scale,
                        bits,
                        k_slab,
                        cur_tile_desc.pack_off,
                        cur_tile_desc.scale_off,
                        local_row_base + next_row_block_base + warp_id - rank_off);
                }

                const int row_local = row_block_base + warp_id;
                const bool row_active =
                    row_local < row_count
                    && row_valid_sh[row_local]
                    && w_decode_valid_sh[row_stage][warp_id];
                float acc[DOWN_SMALL_A_GROUP] = {};

                if (row_active) {
                    const float scale_row = w_scale_sh[row_stage][warp_id];
                    #pragma unroll
                    for (int micro = 0; micro < MICROS_PER_TILE; ++micro) {
                        const int c = micro * N_MICRO + lane;
                        if (c >= cur_tile_desc.n_valid) {
                            continue;
                        }
                        float w = decode_row_weight_from_pack_warp_runtime(
                            &w_pack_sh[row_stage][warp_id][0],
                            bits,
                            c);
                        if (bits != 16) {
                            w *= scale_row;
                        }
                        #pragma unroll
                        for (int a_local = 0; a_local < DOWN_SMALL_A_GROUP; ++a_local) {
                            if (a_local >= group_count) {
                                break;
                            }
                            acc[a_local] += w * z_sh[cur_stage][a_local][c];
                        }
                    }
                }

                if (row_active) {
                    #pragma unroll
                    for (int a_local = 0; a_local < DOWN_SMALL_A_GROUP; ++a_local) {
                        if (a_local >= group_count) {
                            break;
                        }
                        const float sum = warp_sum(acc[a_local]);
                        if (lane == 0) {
                            row_acc_sh[a_local][row_local] += sum;
                        }
                    }
                }

                if (next_row_block_base < row_count) {
                    __syncwarp();
                    row_stage = next_row_stage;
                }
            }

            if (next_tile_idx < tile_count) {
                materialize_down_small_z_stage(
                    tid,
                    blockDim.x,
                    &z_sh[next_stage][0][0],
                    gate_out,
                    up_out,
                    a_group_begin,
                    group_count,
                    intermediate_size,
                    next_tile_desc.col_base,
                    next_tile_desc.n_valid);
                __syncthreads();
                cur_stage = next_stage;
                cur_tile_desc = next_tile_desc;
            } else {
                __syncthreads();
            }
        }

        for (int idx = tid; idx < group_count * row_count; idx += blockDim.x) {
            const int a_local = idx / row_count;
            const int row_local = idx - a_local * row_count;
            if (!row_valid_sh[row_local]) {
                continue;
            }
            const float acc = row_acc_sh[a_local][row_local];
            if (acc == 0.0f) {
                continue;
            }
            atomicAdd(
                &rank_accum[static_cast<int64_t>(token_sh[a_local]) * rank_out + logical_rank_sh[row_local]],
                acc * row_s_sh[row_local] * route_w_sh[a_local]);
        }
        __syncthreads();
    }
}

Tensor i64_vector_to_cuda(const std::vector<int64_t>& host, const c10::Device& device) {
    auto cpu = torch::empty(
        {static_cast<long>(host.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto* dst = cpu.data_ptr<int64_t>();
    std::copy(host.begin(), host.end(), dst);
    return cpu.to(device, /*non_blocking=*/true, /*copy=*/true);
}

Tensor i32_vector_to_cuda(const std::vector<int32_t>& host, const c10::Device& device) {
    auto cpu = torch::empty(
        {static_cast<long>(host.size())},
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
    auto* dst = cpu.data_ptr<int32_t>();
    std::copy(host.begin(), host.end(), dst);
    return cpu.to(device, /*non_blocking=*/true, /*copy=*/true);
}

struct PackedMeta {
    Tensor payload_ptrs;
    Tensor rank_idx_ptrs;
    Tensor tile_meta_ptrs;
    Tensor slab_meta_ptrs;
    Tensor scale_ptrs;
    Tensor s_ptrs;
    Tensor num_slabs;
    Tensor num_tiles;
    Tensor rank_lens;
    Tensor cta_expert;
    Tensor cta_row_start;
    Tensor cta_row_count;
    Tensor cta_rank_off;
    Tensor cta_k_slab;
    Tensor cta_tile_off;
    Tensor cta_tile_end;
    Tensor cta_bits;
    int32_t total_cta = 0;
    std::vector<int32_t> rank_lens_host;
};

static inline void check_i32_cuda_contig(const Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kInt, name, " must be int32");
}

static inline void check_u32_cuda_contig(const Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kUInt32, name, " must be uint32");
}

static inline void check_f16_cuda_contig(const Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kHalf, name, " must be float16");
}

struct PackedMetaKey {
    int device_index = -1;
    std::vector<int64_t> ptrs;
    std::vector<int32_t> shape_info;
};

bool operator==(const PackedMetaKey& a, const PackedMetaKey& b) {
    return a.device_index == b.device_index
        && a.ptrs == b.ptrs
        && a.shape_info == b.shape_info;
}

size_t hash_packed_meta_key(const PackedMetaKey& key) {
    size_t h = static_cast<size_t>(key.device_index + 1);
    auto mix = [&h](uint64_t v) {
        h ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    };
    for (auto v : key.ptrs) {
        mix(static_cast<uint64_t>(v));
    }
    for (auto v : key.shape_info) {
        mix(static_cast<uint64_t>(static_cast<uint32_t>(v)));
    }
    return h;
}

struct PackedMetaCacheEntry {
    PackedMetaKey key;
    PackedMeta value;
};

std::unordered_map<size_t, std::vector<PackedMetaCacheEntry>>& packed_meta_cache() {
    static std::unordered_map<size_t, std::vector<PackedMetaCacheEntry>> cache;
    return cache;
}

std::mutex& packed_meta_cache_mutex() {
    static std::mutex m;
    return m;
}

#if BITSMOE_SMALL_GATE_UP_SEPARATE_LAUNCH
struct SmallPathLaunchRuntime {
    at::cuda::CUDAStream gate_stream;
    at::cuda::CUDAStream up_stream;
    at::cuda::CUDAEvent gate_done;
    at::cuda::CUDAEvent up_done;

    explicit SmallPathLaunchRuntime(int device_index)
        : gate_stream(at::cuda::getStreamFromPool(false, device_index)),
          up_stream(at::cuda::getStreamFromPool(false, device_index)),
          gate_done(),
          up_done() {}
};

SmallPathLaunchRuntime& get_small_path_launch_runtime(int device_index) {
    thread_local std::unordered_map<int, std::unique_ptr<SmallPathLaunchRuntime>> runtimes;
    auto& slot = runtimes[device_index];
    if (!slot) {
        slot = std::make_unique<SmallPathLaunchRuntime>(device_index);
    }
    return *slot;
}
#endif

PackedMeta build_packed_meta_cached(
    const std::vector<Tensor>& payload_list,
    const std::vector<Tensor>& rank_idx_list,
    const std::vector<Tensor>& tile_meta_list,
    const std::vector<Tensor>& slab_meta_list,
    const std::vector<Tensor>& scale_list,
    const std::vector<Tensor>& s_list,
    const c10::Device& device) {
    const int64_t n = static_cast<int64_t>(payload_list.size());

    std::vector<int64_t> payload_ptrs;
    std::vector<int64_t> rank_idx_ptrs;
    std::vector<int64_t> tile_meta_ptrs;
    std::vector<int64_t> slab_meta_ptrs;
    std::vector<int64_t> scale_ptrs;
    std::vector<int64_t> s_ptrs;
    std::vector<int32_t> num_slabs_host;
    std::vector<int32_t> num_tiles_host;
    std::vector<int32_t> rank_lens_host;

    payload_ptrs.reserve(n);
    rank_idx_ptrs.reserve(n);
    tile_meta_ptrs.reserve(n);
    slab_meta_ptrs.reserve(n);
    scale_ptrs.reserve(n);
    s_ptrs.reserve(n);
    num_slabs_host.reserve(n);
    num_tiles_host.reserve(n);
    rank_lens_host.reserve(n);

    for (int64_t i = 0; i < n; ++i) {
        const auto& payload = payload_list[i];
        const auto& rank_idx = rank_idx_list[i];
        const auto& tile_meta = tile_meta_list[i];
        const auto& slab_meta = slab_meta_list[i];
        const auto& scale = scale_list[i];
        const auto& s_buf = s_list[i];

        check_u32_cuda_contig(payload, "payload");
        check_i32_cuda_contig(rank_idx, "rank_idx");
        check_i32_cuda_contig(tile_meta, "tile_meta");
        check_i32_cuda_contig(slab_meta, "slab_meta");
        check_f16_cuda_contig(scale, "scale");
        check_f16_cuda_contig(s_buf, "s_buffer");

        TORCH_CHECK(rank_idx.dim() == 1, "rank_idx must be 1D");
        TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
        TORCH_CHECK(s_buf.dim() == 1, "s_buffer must be 1D");
        TORCH_CHECK(tile_meta.dim() == 2 && tile_meta.size(1) == 3, "tile_meta must be [N,3]");
        TORCH_CHECK(slab_meta.dim() == 2 && slab_meta.size(1) == 4, "slab_meta must be [N,4]");

        payload_ptrs.push_back(reinterpret_cast<int64_t>(payload.data_ptr<uint32_t>()));
        rank_idx_ptrs.push_back(reinterpret_cast<int64_t>(rank_idx.data_ptr<int32_t>()));
        tile_meta_ptrs.push_back(reinterpret_cast<int64_t>(tile_meta.data_ptr<int32_t>()));
        slab_meta_ptrs.push_back(reinterpret_cast<int64_t>(slab_meta.data_ptr<int32_t>()));
        scale_ptrs.push_back(reinterpret_cast<int64_t>(scale.data_ptr<at::Half>()));
        s_ptrs.push_back(reinterpret_cast<int64_t>(s_buf.data_ptr<at::Half>()));

        num_slabs_host.push_back(static_cast<int32_t>(slab_meta.size(0)));
        num_tiles_host.push_back(static_cast<int32_t>(tile_meta.size(0)));
        rank_lens_host.push_back(static_cast<int32_t>(rank_idx.numel()));
    }

    PackedMetaKey key;
    key.device_index = device.index();
    key.ptrs.reserve(static_cast<size_t>(6 * n));
    key.shape_info.reserve(static_cast<size_t>(3 * n));
    for (int64_t i = 0; i < n; ++i) {
        key.ptrs.push_back(payload_ptrs[i]);
        key.ptrs.push_back(rank_idx_ptrs[i]);
        key.ptrs.push_back(tile_meta_ptrs[i]);
        key.ptrs.push_back(slab_meta_ptrs[i]);
        key.ptrs.push_back(scale_ptrs[i]);
        key.ptrs.push_back(s_ptrs[i]);
        key.shape_info.push_back(num_slabs_host[i]);
        key.shape_info.push_back(num_tiles_host[i]);
        key.shape_info.push_back(rank_lens_host[i]);
    }

    const size_t h = hash_packed_meta_key(key);
    {
        std::lock_guard<std::mutex> lock(packed_meta_cache_mutex());
        auto it = packed_meta_cache().find(h);
        if (it != packed_meta_cache().end()) {
            for (const auto& ent : it->second) {
                if (ent.key == key) {
                    return ent.value;
                }
            }
        }
    }

    PackedMeta out;
    out.payload_ptrs = i64_vector_to_cuda(payload_ptrs, device);
    out.rank_idx_ptrs = i64_vector_to_cuda(rank_idx_ptrs, device);
    out.tile_meta_ptrs = i64_vector_to_cuda(tile_meta_ptrs, device);
    out.slab_meta_ptrs = i64_vector_to_cuda(slab_meta_ptrs, device);
    out.scale_ptrs = i64_vector_to_cuda(scale_ptrs, device);
    out.s_ptrs = i64_vector_to_cuda(s_ptrs, device);
    out.num_slabs = i32_vector_to_cuda(num_slabs_host, device);
    out.num_tiles = i32_vector_to_cuda(num_tiles_host, device);
    out.rank_lens = i32_vector_to_cuda(rank_lens_host, device);
    out.rank_lens_host = rank_lens_host;
    std::vector<int32_t> cta_expert_host;
    std::vector<int32_t> cta_row_start_host;
    std::vector<int32_t> cta_row_count_host;
    std::vector<int32_t> cta_rank_off_host;
    std::vector<int32_t> cta_k_slab_host;
    std::vector<int32_t> cta_tile_off_host;
    std::vector<int32_t> cta_tile_end_host;
    std::vector<int32_t> cta_bits_host;

    for (int e = 0; e < static_cast<int>(n); ++e) {
        const auto slab_cpu = slab_meta_list[e].cpu().contiguous();
        const auto* slab_ptr = slab_cpu.data_ptr<int32_t>();
        const int num_slabs = num_slabs_host[e];
        const int num_tiles_total = num_tiles_host[e];

        for (int s = 0; s < num_slabs; ++s) {
            const int bits = slab_ptr[s * 4 + 0];
            const int k_slab = slab_ptr[s * 4 + 1];
            const int rank_off = slab_ptr[s * 4 + 2];
            const int tile_off = slab_ptr[s * 4 + 3];
            const int tile_end = (s + 1 < num_slabs)
                ? slab_ptr[(s + 1) * 4 + 3]
                : num_tiles_total;

            const int row_end = rank_off + k_slab;
            for (int row_start = rank_off; row_start < row_end; row_start += N_TILE) {
                const int row_count = std::min(N_TILE, row_end - row_start);
                cta_expert_host.push_back(e);
                cta_row_start_host.push_back(row_start);
                cta_row_count_host.push_back(row_count);
                cta_rank_off_host.push_back(rank_off);
                cta_k_slab_host.push_back(k_slab);
                cta_tile_off_host.push_back(tile_off);
                cta_tile_end_host.push_back(tile_end);
                cta_bits_host.push_back(bits);
            }
        }
    }

    out.total_cta = static_cast<int32_t>(cta_expert_host.size());
    if (out.total_cta > 0) {
        out.cta_expert = i32_vector_to_cuda(cta_expert_host, device);
        out.cta_row_start = i32_vector_to_cuda(cta_row_start_host, device);
        out.cta_row_count = i32_vector_to_cuda(cta_row_count_host, device);
        out.cta_rank_off = i32_vector_to_cuda(cta_rank_off_host, device);
        out.cta_k_slab = i32_vector_to_cuda(cta_k_slab_host, device);
        out.cta_tile_off = i32_vector_to_cuda(cta_tile_off_host, device);
        out.cta_tile_end = i32_vector_to_cuda(cta_tile_end_host, device);
        out.cta_bits = i32_vector_to_cuda(cta_bits_host, device);
    } else {
        out.cta_expert = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_row_start = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_row_count = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_rank_off = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_k_slab = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_tile_off = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_tile_end = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
        out.cta_bits = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt).device(device));
    }

    {
        std::lock_guard<std::mutex> lock(packed_meta_cache_mutex());
        packed_meta_cache()[h].push_back(PackedMetaCacheEntry{std::move(key), out});
    }
    return out;
}

} // namespace

Tensor moe_packed_forward_cuda(
    const Tensor& h_gate,
    const Tensor& h_up,
    const Tensor& token_indices,
    const Tensor& expert_offsets,
    const Tensor& route_weights,
    const std::vector<Tensor>& gate_payload,
    const std::vector<Tensor>& gate_rank_idx,
    const std::vector<Tensor>& gate_tile_meta,
    const std::vector<Tensor>& gate_slab_meta,
    const std::vector<Tensor>& gate_scale,
    const std::vector<Tensor>& gate_s,
    const std::vector<Tensor>& up_payload,
    const std::vector<Tensor>& up_rank_idx,
    const std::vector<Tensor>& up_tile_meta,
    const std::vector<Tensor>& up_slab_meta,
    const std::vector<Tensor>& up_scale,
    const std::vector<Tensor>& up_s,
    const std::vector<Tensor>& down_payload,
    const std::vector<Tensor>& down_rank_idx,
    const std::vector<Tensor>& down_tile_meta,
    const std::vector<Tensor>& down_slab_meta,
    const std::vector<Tensor>& down_scale,
    const std::vector<Tensor>& down_s,
    int64_t rank_out,
    int64_t intermediate_size,
    int64_t /*act_type*/) {
    c10::cuda::CUDAGuard guard(h_gate.device());

    const int64_t token_count = h_gate.size(0);
    const int64_t num_assign = token_indices.numel();
    const int64_t num_experts = expert_offsets.numel() - 1;
    if (num_assign == 0 || num_experts == 0) {
        return torch::zeros(
            {token_count, rank_out},
            torch::TensorOptions().device(h_gate.device()).dtype(torch::kFloat));
    }

    TORCH_CHECK(h_gate.is_contiguous(), "h_gate must be contiguous");
    TORCH_CHECK(h_up.is_contiguous(), "h_up must be contiguous");
    TORCH_CHECK(token_indices.is_contiguous(), "token_indices must be contiguous");
    TORCH_CHECK(expert_offsets.is_contiguous(), "expert_offsets must be contiguous");
    TORCH_CHECK(route_weights.is_contiguous(), "route_weights must be contiguous");
    TORCH_CHECK(h_gate.scalar_type() == torch::kHalf, "h_gate must be float16");
    TORCH_CHECK(h_up.scalar_type() == torch::kHalf, "h_up must be float16");
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt, "token_indices must be int32");
    TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt, "expert_offsets must be int32");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat, "route_weights must be float32");

    auto gate_meta = build_packed_meta_cached(
        gate_payload,
        gate_rank_idx,
        gate_tile_meta,
        gate_slab_meta,
        gate_scale,
        gate_s,
        h_gate.device());
    auto up_meta = build_packed_meta_cached(
        up_payload,
        up_rank_idx,
        up_tile_meta,
        up_slab_meta,
        up_scale,
        up_s,
        h_gate.device());
    auto down_meta = build_packed_meta_cached(
        down_payload,
        down_rank_idx,
        down_tile_meta,
        down_slab_meta,
        down_scale,
        down_s,
        h_gate.device());

    const auto main_stream = at::cuda::getCurrentCUDAStream(h_gate.get_device());
    const int32_t num_experts_i32 = static_cast<int32_t>(num_experts);
    const auto i32_cuda_opts =
        torch::TensorOptions().dtype(torch::kInt).device(h_gate.device());
    auto gate_task_counts = torch::empty({num_experts}, i32_cuda_opts);
    constexpr int kGateTaskBuildThreads = 256;
    const int gate_task_build_blocks =
        static_cast<int>((num_experts_i32 + kGateTaskBuildThreads - 1) / kGateTaskBuildThreads);
    build_gate_task_count_kernel<<<
        gate_task_build_blocks,
        kGateTaskBuildThreads,
        0,
        main_stream.stream()>>>(
            expert_offsets.data_ptr<int32_t>(),
            num_experts_i32,
            gate_task_counts.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gate_task_prefix = at::cumsum(gate_task_counts, /*dim=*/0).to(torch::kInt);
    const int32_t gate_num_tasks = static_cast<int32_t>(
        gate_task_counts.sum().item<int64_t>());
    const int32_t active_experts = static_cast<int32_t>(
        gate_task_counts.gt(0).sum().item<int64_t>());
    auto gate_task_expert = torch::empty({gate_num_tasks}, i32_cuda_opts);
    auto gate_task_assign_base = torch::empty({gate_num_tasks}, i32_cuda_opts);
    auto gate_task_assign_count = torch::empty({gate_num_tasks}, i32_cuda_opts);
    if (gate_num_tasks > 0) {
        build_gate_task_table_kernel<<<
            gate_task_build_blocks,
            kGateTaskBuildThreads,
            0,
            main_stream.stream()>>>(
                expert_offsets.data_ptr<int32_t>(),
                gate_task_counts.data_ptr<int32_t>(),
                gate_task_prefix.data_ptr<int32_t>(),
                num_experts_i32,
                gate_task_expert.data_ptr<int32_t>(),
                gate_task_assign_base.data_ptr<int32_t>(),
                gate_task_assign_count.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int32_t max_gate_task_rows = (gate_num_tasks > 0)
        ? gate_task_assign_count.max().item<int32_t>()
        : 0;

    const int n_tiles = static_cast<int>((intermediate_size + N_TILE - 1) / N_TILE);
    const float avg_assign_per_active = (active_experts > 0)
        ? static_cast<float>(num_assign) / static_cast<float>(active_experts)
        : 0.0f;
    const bool use_gate_small =
        gate_num_tasks > 0
        && (
            token_count == 1
            || (
                num_assign <= DECODE_SMALL_MAX_NUM_ASSIGN
                && max_gate_task_rows <= GATE_SMALL_GROUP
                && avg_assign_per_active <= DECODE_SMALL_AVG_ASSIGN_THRESHOLD));
    // Keep a single small-path policy: decode-sized workloads use gate/up small + down_small.
    const bool use_down_small =
        down_meta.total_cta > 0
        && use_gate_small;
    const bool use_small_gate_up = use_gate_small;

    Tensor gate_out;
    Tensor up_out;
    const auto out_options =
        torch::TensorOptions().device(h_gate.device()).dtype(torch::kFloat);
    if (use_small_gate_up) {
        gate_out = torch::empty({num_assign, intermediate_size}, out_options);
        up_out = torch::empty_like(gate_out);
    } else {
        gate_out = torch::zeros({num_assign, intermediate_size}, out_options);
        up_out = torch::zeros_like(gate_out);
    }

    auto launch_gate_up_tc = [&](const Tensor& h_in, const PackedMeta& meta, const Tensor& out_tensor, const at::cuda::CUDAStream& stream) {
        if (gate_num_tasks <= 0) {
            return;
        }
        dim3 grid(static_cast<unsigned int>(gate_num_tasks), static_cast<unsigned int>(n_tiles), 1);
        gate_up_tc_kernel<<<grid, GATE_TC_BLOCK_THREADS, 0, stream.stream()>>>(
            reinterpret_cast<const half*>(h_in.data_ptr<at::Half>()),
            static_cast<int32_t>(h_in.size(1)),
            token_indices.data_ptr<int32_t>(),
            meta.payload_ptrs.data_ptr<int64_t>(),
            meta.rank_idx_ptrs.data_ptr<int64_t>(),
            meta.tile_meta_ptrs.data_ptr<int64_t>(),
            meta.slab_meta_ptrs.data_ptr<int64_t>(),
            meta.scale_ptrs.data_ptr<int64_t>(),
            meta.s_ptrs.data_ptr<int64_t>(),
            meta.num_slabs.data_ptr<int32_t>(),
            meta.num_tiles.data_ptr<int32_t>(),
            gate_task_expert.data_ptr<int32_t>(),
            gate_task_assign_base.data_ptr<int32_t>(),
            gate_task_assign_count.data_ptr<int32_t>(),
            gate_num_tasks,
            static_cast<int32_t>(intermediate_size),
            out_tensor.data_ptr<float>());
    };
    auto make_gate_up_small_device_meta = [](const PackedMeta& meta) {
        GateUpSmallDeviceMeta device_meta{};
        device_meta.payload_ptrs = meta.payload_ptrs.data_ptr<int64_t>();
        device_meta.rank_idx_ptrs = meta.rank_idx_ptrs.data_ptr<int64_t>();
        device_meta.tile_meta_ptrs = meta.tile_meta_ptrs.data_ptr<int64_t>();
        device_meta.slab_meta_ptrs = meta.slab_meta_ptrs.data_ptr<int64_t>();
        device_meta.scale_ptrs = meta.scale_ptrs.data_ptr<int64_t>();
        device_meta.s_ptrs = meta.s_ptrs.data_ptr<int64_t>();
        device_meta.num_slabs = meta.num_slabs.data_ptr<int32_t>();
        device_meta.num_tiles = meta.num_tiles.data_ptr<int32_t>();
        return device_meta;
    };
#if BITSMOE_SMALL_GATE_UP_SEPARATE_LAUNCH
    auto launch_gate_up_small = [&](const Tensor& h_in, const PackedMeta& meta, const Tensor& out_tensor, const at::cuda::CUDAStream& stream) {
        if (gate_num_tasks <= 0) {
            return;
        }
        const GateUpSmallDeviceMeta device_meta = make_gate_up_small_device_meta(meta);
        dim3 grid(
            static_cast<unsigned int>(gate_num_tasks),
            static_cast<unsigned int>(n_tiles),
            1);
        gate_up_small_kernel<<<grid, GATE_SMALL_FULL_THREADS, 0, stream.stream()>>>(
            reinterpret_cast<const half*>(h_in.data_ptr<at::Half>()),
            static_cast<int32_t>(h_in.size(1)),
            token_indices.data_ptr<int32_t>(),
            device_meta,
            gate_task_expert.data_ptr<int32_t>(),
            gate_task_assign_base.data_ptr<int32_t>(),
            gate_task_assign_count.data_ptr<int32_t>(),
            gate_num_tasks,
            static_cast<int32_t>(intermediate_size),
            out_tensor.data_ptr<float>());
    };
#endif
    auto launch_gate_up_small_dual = [&](const at::cuda::CUDAStream& stream) {
        if (gate_num_tasks <= 0) {
            return;
        }
        const GateUpSmallDeviceMeta gate_device_meta = make_gate_up_small_device_meta(gate_meta);
        const GateUpSmallDeviceMeta up_device_meta = make_gate_up_small_device_meta(up_meta);
#if BITSMOE_SMALL_GATE_UP_DUAL_FALLBACK_128T
        dim3 grid(
            static_cast<unsigned int>(gate_num_tasks),
            static_cast<unsigned int>(n_tiles),
            2);
        gate_up_small_dual_kernel_legacy<<<grid, GATE_SMALL_FULL_THREADS, 0, stream.stream()>>>(
            reinterpret_cast<const half*>(h_gate.data_ptr<at::Half>()),
            static_cast<int32_t>(h_gate.size(1)),
            gate_device_meta,
            reinterpret_cast<const half*>(h_up.data_ptr<at::Half>()),
            static_cast<int32_t>(h_up.size(1)),
            up_device_meta,
            token_indices.data_ptr<int32_t>(),
            gate_task_expert.data_ptr<int32_t>(),
            gate_task_assign_base.data_ptr<int32_t>(),
            gate_task_assign_count.data_ptr<int32_t>(),
            gate_num_tasks,
            static_cast<int32_t>(intermediate_size),
            gate_out.data_ptr<float>(),
            up_out.data_ptr<float>());
#else
        dim3 grid(
            static_cast<unsigned int>(gate_num_tasks),
            static_cast<unsigned int>(n_tiles),
            4);
        gate_up_small_dual_kernel<<<grid, GATE_SMALL_THREADS, 0, stream.stream()>>>(
            reinterpret_cast<const half*>(h_gate.data_ptr<at::Half>()),
            static_cast<int32_t>(h_gate.size(1)),
            gate_device_meta,
            reinterpret_cast<const half*>(h_up.data_ptr<at::Half>()),
            static_cast<int32_t>(h_up.size(1)),
            up_device_meta,
            token_indices.data_ptr<int32_t>(),
            gate_task_expert.data_ptr<int32_t>(),
            gate_task_assign_base.data_ptr<int32_t>(),
            gate_task_assign_count.data_ptr<int32_t>(),
            gate_num_tasks,
            static_cast<int32_t>(intermediate_size),
            gate_out.data_ptr<float>(),
            up_out.data_ptr<float>());
#endif
    };
    if (gate_num_tasks > 0 && n_tiles > 0) {
        if (use_small_gate_up) {
#if BITSMOE_SMALL_GATE_UP_SEPARATE_LAUNCH
            auto& small_runtime = get_small_path_launch_runtime(h_gate.get_device());
            // Ensure producer work on main_stream (task tables/meta) is visible
            // before side-stream gate/up kernels consume these buffers.
            at::cuda::CUDAEvent gate_up_ready;
            gate_up_ready.record(main_stream);
            gate_up_ready.block(small_runtime.gate_stream);
            gate_up_ready.block(small_runtime.up_stream);
            launch_gate_up_small(h_gate, gate_meta, gate_out, small_runtime.gate_stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            launch_gate_up_small(h_up, up_meta, up_out, small_runtime.up_stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            small_runtime.gate_done.record(small_runtime.gate_stream);
            small_runtime.up_done.record(small_runtime.up_stream);
            small_runtime.gate_done.block(main_stream);
            small_runtime.up_done.block(main_stream);
#else
            launch_gate_up_small_dual(main_stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
        } else {
            const auto gate_stream = at::cuda::getStreamFromPool(false, h_gate.get_device());
            const auto up_stream = at::cuda::getStreamFromPool(false, h_gate.get_device());
            at::cuda::CUDAEvent gate_done;
            at::cuda::CUDAEvent up_done;
            // Enforce producer(main_stream) -> consumer(gate/up streams) ordering.
            at::cuda::CUDAEvent gate_up_ready;
            gate_up_ready.record(main_stream);
            gate_up_ready.block(gate_stream);
            gate_up_ready.block(up_stream);
            launch_gate_up_tc(h_gate, gate_meta, gate_out, gate_stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            launch_gate_up_tc(h_up, up_meta, up_out, up_stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            gate_done.record(gate_stream);
            up_done.record(up_stream);
            gate_done.block(main_stream);
            up_done.block(main_stream);
        }
    }

    auto rank_accum = torch::zeros(
        {token_count, rank_out},
        torch::TensorOptions().device(h_gate.device()).dtype(torch::kFloat));

    if (down_meta.total_cta > 0) {
        if (use_down_small) {
            down_small_kernel<<<down_meta.total_cta, DOWN_SMALL_THREADS, 0, main_stream.stream()>>>(
                gate_out.data_ptr<float>(),
                up_out.data_ptr<float>(),
                route_weights.data_ptr<float>(),
                token_indices.data_ptr<int32_t>(),
                expert_offsets.data_ptr<int32_t>(),
                down_meta.payload_ptrs.data_ptr<int64_t>(),
                down_meta.rank_idx_ptrs.data_ptr<int64_t>(),
                down_meta.tile_meta_ptrs.data_ptr<int64_t>(),
                down_meta.slab_meta_ptrs.data_ptr<int64_t>(),
                down_meta.scale_ptrs.data_ptr<int64_t>(),
                down_meta.s_ptrs.data_ptr<int64_t>(),
                down_meta.num_slabs.data_ptr<int32_t>(),
                down_meta.num_tiles.data_ptr<int32_t>(),
                down_meta.rank_lens.data_ptr<int32_t>(),
                down_meta.cta_expert.data_ptr<int32_t>(),
                down_meta.cta_row_start.data_ptr<int32_t>(),
                down_meta.cta_row_count.data_ptr<int32_t>(),
                down_meta.cta_rank_off.data_ptr<int32_t>(),
                down_meta.cta_k_slab.data_ptr<int32_t>(),
                down_meta.cta_tile_off.data_ptr<int32_t>(),
                down_meta.cta_tile_end.data_ptr<int32_t>(),
                down_meta.cta_bits.data_ptr<int32_t>(),
                down_meta.total_cta,
                static_cast<int32_t>(intermediate_size),
                static_cast<int32_t>(rank_out),
                rank_accum.data_ptr<float>());
        } else {
            const size_t down_smem_bytes = down_kernel_shared_bytes();
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                down_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(down_smem_bytes)));
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                down_kernel,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100));
            down_kernel<<<down_meta.total_cta, N_TILE, down_smem_bytes, main_stream.stream()>>>(
                gate_out.data_ptr<float>(),
                up_out.data_ptr<float>(),
                route_weights.data_ptr<float>(),
                token_indices.data_ptr<int32_t>(),
                expert_offsets.data_ptr<int32_t>(),
                down_meta.payload_ptrs.data_ptr<int64_t>(),
                down_meta.rank_idx_ptrs.data_ptr<int64_t>(),
                down_meta.tile_meta_ptrs.data_ptr<int64_t>(),
                down_meta.slab_meta_ptrs.data_ptr<int64_t>(),
                down_meta.scale_ptrs.data_ptr<int64_t>(),
                down_meta.s_ptrs.data_ptr<int64_t>(),
                down_meta.num_slabs.data_ptr<int32_t>(),
                down_meta.num_tiles.data_ptr<int32_t>(),
                down_meta.rank_lens.data_ptr<int32_t>(),
                down_meta.cta_expert.data_ptr<int32_t>(),
                down_meta.cta_row_start.data_ptr<int32_t>(),
                down_meta.cta_row_count.data_ptr<int32_t>(),
                down_meta.cta_rank_off.data_ptr<int32_t>(),
                down_meta.cta_k_slab.data_ptr<int32_t>(),
                down_meta.cta_tile_off.data_ptr<int32_t>(),
                down_meta.cta_tile_end.data_ptr<int32_t>(),
                down_meta.cta_bits.data_ptr<int32_t>(),
                down_meta.total_cta,
                static_cast<int32_t>(intermediate_size),
                static_cast<int32_t>(rank_out),
                rank_accum.data_ptr<float>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return rank_accum;
}
