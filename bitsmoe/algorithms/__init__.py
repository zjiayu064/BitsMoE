import torch

# Device name
def get_cuda_device_name() -> str:
    """
    Return a human-readable CUDA device name, similar to unsloth style.

    Examples:
        "NVIDIA H100 PCIe 80GB"
        "NVIDIA A100-SXM4-80GB"
        "NVIDIA RTX A6000"
        "NVIDIA GeForce RTX 3090"
        "cpu"
    """
    if not torch.cuda.is_available():
        return "cpu"

    dev = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(dev)

    name = p.name.strip()
    vram_gb = round(p.total_memory / 1024**3)

    return f"{name} ({vram_gb}GB)"


# Packed routed-expert forward backend (gate/up/down kernels)
try:
    from bitsmoe.algorithms import (
        _mlp_forward_cuda as mlp_forward_cuda
    )

    _HAS_MLP_FORWARD_CUDA = True
    _MLP_FORWARD_BACKEND = get_cuda_device_name()

except Exception as e:
    mlp_forward_cuda = None
    _HAS_MLP_FORWARD_CUDA = False
    _MLP_FORWARD_BACKEND = "None"


__all__ = [
    "mlp_forward_cuda",
    "_HAS_MLP_FORWARD_CUDA",
    "_MLP_FORWARD_BACKEND",
    "get_cuda_device_name",
]
