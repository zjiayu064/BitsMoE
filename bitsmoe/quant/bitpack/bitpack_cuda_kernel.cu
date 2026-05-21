#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

using torch::Tensor;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

inline dim3 get_grid_from_total(int64_t total_elements) {
    int64_t blocks = (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > kMaxBlocks) {
        blocks = kMaxBlocks;
    }
    return dim3(static_cast<unsigned int>(blocks));
}

inline dim3 get_grid_from_warps(int64_t total_warps) {
    int64_t blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > kMaxBlocks) {
        blocks = kMaxBlocks;
    }
    return dim3(static_cast<unsigned int>(blocks));
}

__global__ void pack_bitplanes_u8_warp_kernel(
    const uint8_t* __restrict__ in,
    uint32_t* __restrict__ out,
    int rows,
    int cols,
    int groups,
    int bits
) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_local = threadIdx.x / kWarpSize;
    int64_t warp_id = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_local;
    const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    const int64_t total_words = static_cast<int64_t>(rows) * groups * bits;
    while (warp_id < total_words) {
        const int plane = static_cast<int>(warp_id % bits);
        const int64_t tmp = warp_id / bits;
        const int group = static_cast<int>(tmp % groups);
        const int row = static_cast<int>(tmp / groups);

        const int col = group * kWarpSize + lane;
        uint8_t value = 0;
        if (col < cols) {
            value = in[static_cast<int64_t>(row) * cols + col];
        }

        const unsigned mask = __ballot_sync(
            0xFFFFFFFFu, ((static_cast<unsigned>(value) >> plane) & 0x1u) != 0u
        );
        if (lane == 0) {
            out[warp_id] = static_cast<uint32_t>(mask);
        }
        warp_id += warp_stride;
    }
}

__global__ void pack_bitplanes_i8_warp_kernel(
    const int8_t* __restrict__ in,
    uint32_t* __restrict__ out,
    int rows,
    int cols,
    int groups,
    int bits,
    int offset
) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_local = threadIdx.x / kWarpSize;
    int64_t warp_id = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_local;
    const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    const int64_t total_words = static_cast<int64_t>(rows) * groups * bits;
    while (warp_id < total_words) {
        const int plane = static_cast<int>(warp_id % bits);
        const int64_t tmp = warp_id / bits;
        const int group = static_cast<int>(tmp % groups);
        const int row = static_cast<int>(tmp / groups);

        const int col = group * kWarpSize + lane;
        uint8_t value = 0;
        if (col < cols) {
            const int8_t q = in[static_cast<int64_t>(row) * cols + col];
            if (offset == 1) {
                value = static_cast<uint8_t>((static_cast<int16_t>(q) + 1) >> 1);
            } else {
                value = static_cast<uint8_t>(static_cast<int16_t>(q) + static_cast<int16_t>(offset));
            }
        }

        const unsigned mask = __ballot_sync(
            0xFFFFFFFFu, ((static_cast<unsigned>(value) >> plane) & 0x1u) != 0u
        );
        if (lane == 0) {
            out[warp_id] = static_cast<uint32_t>(mask);
        }
        warp_id += warp_stride;
    }
}

__global__ void unpack_bitplanes_to_u8_warp_kernel(
    const uint32_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int rows,
    int groups,
    int bits
) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_local = threadIdx.x / kWarpSize;
    int64_t warp_id = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_local;
    const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    const int out_cols = groups * kWarpSize;
    const int64_t total_row_groups = static_cast<int64_t>(rows) * groups;

    while (warp_id < total_row_groups) {
        const int row = static_cast<int>(warp_id / groups);
        const int group = static_cast<int>(warp_id % groups);
        const int64_t in_base = warp_id * bits;

        uint8_t value = 0;
        #pragma unroll
        for (int plane = 0; plane < 8; ++plane) {
            if (plane < bits) {
                const uint32_t word = in[in_base + plane];
                value |= static_cast<uint8_t>(((word >> lane) & 0x1u) << plane);
            }
        }

        out[static_cast<int64_t>(row) * out_cols + group * kWarpSize + lane] = value;
        warp_id += warp_stride;
    }
}

__global__ void unpack_bitplanes_to_i8_warp_kernel(
    const uint32_t* __restrict__ in,
    int8_t* __restrict__ out,
    int rows,
    int groups,
    int bits,
    int offset
) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_local = threadIdx.x / kWarpSize;
    int64_t warp_id = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_local;
    const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    const int out_cols = groups * kWarpSize;
    const int64_t total_row_groups = static_cast<int64_t>(rows) * groups;

    while (warp_id < total_row_groups) {
        const int row = static_cast<int>(warp_id / groups);
        const int group = static_cast<int>(warp_id % groups);
        const int64_t in_base = warp_id * bits;

        uint8_t value_u = 0;
        #pragma unroll
        for (int plane = 0; plane < 8; ++plane) {
            if (plane < bits) {
                const uint32_t word = in[in_base + plane];
                value_u |= static_cast<uint8_t>(((word >> lane) & 0x1u) << plane);
            }
        }

        int8_t value_s;
        if (offset == 1) {
            value_s = static_cast<int8_t>((static_cast<int16_t>(value_u) << 1) - 1);
        } else {
            value_s = static_cast<int8_t>(static_cast<int16_t>(value_u) - static_cast<int16_t>(offset));
        }
        out[static_cast<int64_t>(row) * out_cols + group * kWarpSize + lane] = value_s;
        warp_id += warp_stride;
    }
}

__global__ void int8_to_uint8_kernel(
    const int8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int64_t total,
    int offset
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    while (idx < total) {
        const int8_t q = in[idx];

        if (offset == 1) {
            out[idx] = static_cast<uint8_t>((static_cast<int16_t>(q) + 1) >> 1);
        } else {
            const int16_t v = static_cast<int16_t>(q) + static_cast<int16_t>(offset);
            out[idx] = static_cast<uint8_t>(v);
        }

        idx += stride;
    }
}

__global__ void uint8_to_int8_kernel(
    const uint8_t* __restrict__ in,
    int8_t* __restrict__ out,
    int64_t total,
    int offset
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    while (idx < total) {
        const uint8_t u = in[idx];

        if (offset == 1) {
            out[idx] = static_cast<int8_t>((static_cast<int16_t>(u) << 1) - 1);
        } else {
            const int16_t v = static_cast<int16_t>(u) - static_cast<int16_t>(offset);
            out[idx] = static_cast<int8_t>(v);
        }

        idx += stride;
    }
}

inline Tensor pack_bits_cuda_impl(const Tensor& input, int bits) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "pack_bits: input must be 2D [rows, cols]");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "pack_bits expects uint8 input");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    TORCH_CHECK(rows >= 0 && cols >= 0, "pack_bits: invalid input shape");

    const int64_t groups = (cols + 31) / 32;
    const int64_t out_cols = groups * bits;

    Tensor out = torch::empty({rows, out_cols}, input.options().dtype(torch::kUInt32));

    const int64_t total_words = rows * out_cols;
    const dim3 grid = get_grid_from_warps(total_words);
    const dim3 block(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream();

    pack_bitplanes_u8_warp_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        out.data_ptr<uint32_t>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(groups),
        bits
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}

inline Tensor unpack_bits_cuda_impl(const Tensor& input, int bits) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "unpack_bits: input must be 2D [rows, packed_cols]");
    TORCH_CHECK(input.dtype() == torch::kUInt32, "unpack_bits expects uint32 input");

    const int64_t rows = input.size(0);
    const int64_t packed_cols = input.size(1);
    TORCH_CHECK(packed_cols % bits == 0,
        "unpack_bits: packed_cols must be divisible by bits");

    const int64_t groups = packed_cols / bits;
    const int64_t out_cols = groups * 32;

    Tensor out = torch::empty({rows, out_cols}, input.options().dtype(torch::kUInt8));

    const int64_t total_row_groups = rows * groups;
    const dim3 grid = get_grid_from_warps(total_row_groups);
    const dim3 block(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream();

    unpack_bitplanes_to_u8_warp_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<uint32_t>(),
        out.data_ptr<uint8_t>(),
        static_cast<int>(rows),
        static_cast<int>(groups),
        bits
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}

inline Tensor pack_int8_bits_cuda_impl(const Tensor& input, int bits, int offset) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "pack_int8_bits: input must be 2D [rows, cols]");
    TORCH_CHECK(input.dtype() == torch::kInt8, "pack_int8_bits expects int8 input");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    TORCH_CHECK(rows >= 0 && cols >= 0, "pack_int8_bits: invalid input shape");

    const int64_t groups = (cols + 31) / 32;
    const int64_t out_cols = groups * bits;
    Tensor out = torch::empty({rows, out_cols}, input.options().dtype(torch::kUInt32));

    const int64_t total_words = rows * out_cols;
    const dim3 grid = get_grid_from_warps(total_words);
    const dim3 block(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream();

    pack_bitplanes_i8_warp_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<int8_t>(),
        out.data_ptr<uint32_t>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(groups),
        bits,
        offset
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}

inline Tensor unpack_int8_bits_cuda_impl(const Tensor& input, int bits, int offset) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "unpack_int8_bits: input must be 2D [rows, packed_cols]");
    TORCH_CHECK(input.dtype() == torch::kUInt32, "unpack_int8_bits expects uint32 input");

    const int64_t rows = input.size(0);
    const int64_t packed_cols = input.size(1);
    TORCH_CHECK(packed_cols % bits == 0, "unpack_int8_bits: packed_cols must be divisible by bits");

    const int64_t groups = packed_cols / bits;
    const int64_t out_cols = groups * 32;
    Tensor out = torch::empty({rows, out_cols}, input.options().dtype(torch::kInt8));

    const int64_t total_row_groups = rows * groups;
    const dim3 grid = get_grid_from_warps(total_row_groups);
    const dim3 block(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream();

    unpack_bitplanes_to_i8_warp_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<uint32_t>(),
        out.data_ptr<int8_t>(),
        static_cast<int>(rows),
        static_cast<int>(groups),
        bits,
        offset
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}

} // namespace

Tensor pack_1bit_cuda(const Tensor& input) {
    return pack_bits_cuda_impl(input, 1);
}

Tensor unpack_1bit_cuda(const Tensor& input) {
    return unpack_bits_cuda_impl(input, 1);
}

Tensor pack_2bit_cuda(const Tensor& input) {
    return pack_bits_cuda_impl(input, 2);
}

Tensor unpack_2bit_cuda(const Tensor& input) {
    return unpack_bits_cuda_impl(input, 2);
}

Tensor pack_3bit_cuda(const Tensor& input) {
    return pack_bits_cuda_impl(input, 3);
}

Tensor unpack_3bit_cuda(const Tensor& input) {
    return unpack_bits_cuda_impl(input, 3);
}

Tensor pack_4bit_cuda(const Tensor& input) {
    return pack_bits_cuda_impl(input, 4);
}

Tensor unpack_4bit_cuda(const Tensor& input) {
    return unpack_bits_cuda_impl(input, 4);
}

Tensor pack_8bit_cuda(const Tensor& input) {
    return pack_bits_cuda_impl(input, 8);
}

Tensor unpack_8bit_cuda(const Tensor& input) {
    return unpack_bits_cuda_impl(input, 8);
}

Tensor pack_6bit_cuda(const Tensor& input) {
    return pack_bits_cuda_impl(input, 6);
}

Tensor unpack_6bit_cuda(const Tensor& input) {
    return unpack_bits_cuda_impl(input, 6);
}

Tensor pack_int8_1bit_cuda(const Tensor& input) {
    return pack_int8_bits_cuda_impl(input, 1, 1);
}

Tensor unpack_int8_1bit_cuda(const Tensor& input) {
    return unpack_int8_bits_cuda_impl(input, 1, 1);
}

Tensor pack_int8_2bit_cuda(const Tensor& input) {
    return pack_int8_bits_cuda_impl(input, 2, 2);
}

Tensor unpack_int8_2bit_cuda(const Tensor& input) {
    return unpack_int8_bits_cuda_impl(input, 2, 2);
}

Tensor pack_int8_3bit_cuda(const Tensor& input) {
    return pack_int8_bits_cuda_impl(input, 3, 4);
}

Tensor unpack_int8_3bit_cuda(const Tensor& input) {
    return unpack_int8_bits_cuda_impl(input, 3, 4);
}

Tensor pack_int8_4bit_cuda(const Tensor& input) {
    return pack_int8_bits_cuda_impl(input, 4, 8);
}

Tensor unpack_int8_4bit_cuda(const Tensor& input) {
    return unpack_int8_bits_cuda_impl(input, 4, 8);
}

Tensor pack_int8_6bit_cuda(const Tensor& input) {
    return pack_int8_bits_cuda_impl(input, 6, 32);
}

Tensor unpack_int8_6bit_cuda(const Tensor& input) {
    return unpack_int8_bits_cuda_impl(input, 6, 32);
}

Tensor pack_int8_8bit_cuda(const Tensor& input) {
    return pack_int8_bits_cuda_impl(input, 8, 128);
}

Tensor unpack_int8_8bit_cuda(const Tensor& input) {
    return unpack_int8_bits_cuda_impl(input, 8, 128);
}

Tensor int8_to_uint8_cuda(const Tensor& input, int offset) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kInt8, "int8_to_uint8 expects int8 input");
    const int64_t total = input.numel();

    Tensor out = torch::empty_like(input, input.options().dtype(torch::kUInt8));

    const dim3 grid = get_grid_from_total(total);
    const dim3 block(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream();

    int8_to_uint8_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<int8_t>(),
        out.data_ptr<uint8_t>(),
        total,
        offset
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}

Tensor uint8_to_int8_cuda(const Tensor& input, int offset) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kUInt8, "uint8_to_int8 expects uint8 input");
    const int64_t total = input.numel();

    Tensor out = torch::empty_like(input, input.options().dtype(torch::kInt8));

    const dim3 grid = get_grid_from_total(total);
    const dim3 block(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream();

    uint8_to_int8_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        out.data_ptr<int8_t>(),
        total,
        offset
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return out;
}
