import torch


def marlin_shared_basis_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or x.dtype != torch.float16:
        raise RuntimeError("Marlin shared basis requires CUDA FP16 activations.")
    if weight.dtype != torch.int32 or scales.dtype != torch.float16 or weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("Invalid Marlin shared basis layout.")

    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
    from vllm.scalar_type import scalar_types

    k = scales.shape[0] * 128
    n = scales.shape[1]
    if k == 0 or n == 0 or tuple(weight.shape) != (k // 16, n * 4):
        raise ValueError(f"Invalid W8 Marlin shared basis shapes: {tuple(weight.shape)}, {tuple(scales.shape)}.")
    if x.shape[-1] != k:
        raise ValueError(f"Shared basis input width {x.shape[-1]} does not match K={k}.")

    empty = torch.empty(0, dtype=torch.int32, device=x.device)
    return apply_gptq_marlin_linear(
        input=x,
        weight=weight,
        weight_scale=scales,
        weight_zp=empty,
        g_idx=empty,
        g_idx_sort_indices=empty,
        workspace=workspace,
        wtype=scalar_types.uint8b128,
        output_size_per_partition=n,
        input_size_per_partition=k,
        is_k_full=True,
    )
