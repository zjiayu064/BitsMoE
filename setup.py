import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_torch_lib_dir():
    torch_dir = os.path.dirname(torch.__file__)
    torch_lib = os.path.join(torch_dir, "lib")
    assert os.path.exists(torch_lib), f"torch lib dir not found: {torch_lib}"
    return torch_lib

def get_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return cuda_home
    # fallback
    return "/usr/local/cuda"

CUDA_HOME = get_cuda_home()
TORCH_LIB_DIR = get_torch_lib_dir()
RPATH_FLAG = f"-Wl,-rpath,{TORCH_LIB_DIR}"

CUTLASS_DIR = os.path.join(
    os.path.dirname(__file__),
    "bitsmoe/algorithms/cpp/cutlass",
    "include",
)

CUTLASS_TOOLS_DIR = os.path.join(
    os.path.dirname(__file__),
    "bitsmoe/algorithms/cpp/cutlass",
    "tools/util/include",
)

# Note: Update these NVCC architecture flags to match your actual target GPU architectures.
NVCC_ARCH_FLAGS = [
    "-gencode=arch=compute_80,code=sm_80",          # A100
    "-gencode=arch=compute_86,code=sm_86",          # RTX3090
    "-gencode=arch=compute_89,code=sm_89",          # A6000 Ada
    # "-gencode=arch=compute_90a,code=sm_90a",        # H100 HBM3
    # "-gencode=arch=compute_90a,code=compute_90a",   # PTX fallback
]

COMMON_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
] + NVCC_ARCH_FLAGS

setup(
    name="bitsmoe",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        # Bitpack CUDA extension
        CUDAExtension(
            name="bitsmoe.quant._bitpack_cuda",
            sources=[
                "bitsmoe/quant/bitpack/bitpack_cuda_kernel.cu",
                "bitsmoe/quant/bitpack/bitpack_cuda_binding.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": COMMON_NVCC_FLAGS,
            },
            extra_link_args=[RPATH_FLAG],
        ),

        # Packed MoE forward CUDA extension (gate/up/down kernels + CUTLASS GEMM)
        CUDAExtension(
            name="bitsmoe.algorithms._mlp_forward_cuda",
            sources=[
                "bitsmoe/algorithms/cpp/mlp_forward.cu",
                "bitsmoe/algorithms/cpp/mlp_forward_binding.cpp",
            ],
            include_dirs=[
                CUTLASS_DIR,
                CUTLASS_TOOLS_DIR,
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": COMMON_NVCC_FLAGS,
            },
            extra_link_args=[RPATH_FLAG],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
