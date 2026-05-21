import torch
import torch.nn as nn

from bitsmoe.utils.logger import setup_logger

logger = setup_logger(__name__)

def drop_dense_linear(linear: nn.Linear, name: str = ""):
    """
    Safely remove weight & bias Parameters from an nn.Linear module.

    After this:
        - linear.weight / linear.bias are no longer nn.Parameter
        - They will NOT appear in model.parameters()
        - They will NOT be moved by model.to("cuda")
        - This is the canonical way to free dense FP32 MoE weights

    Args:
        linear (nn.Linear): target linear module
        name (str): optional module name for logging
    """

    if not isinstance(linear, nn.Linear):
        logger.warning(f"[BitsMoE] skip drop_dense_linear on non-Linear: {name}")
        return

    # ---- remove weight ----
    if linear.weight is not None:
        linear.register_parameter("weight", None)

    # ---- remove bias ----
    if linear.bias is not None:
        linear.register_parameter("bias", None)

    # ---- optional placeholder buffer (debug-friendly) ----
    if not hasattr(linear, "_bitsmoe_dense_cleared"):
        linear.register_buffer("_bitsmoe_dense_cleared", torch.empty(0))


# For debug
def report_model_param_memory(model):
    total_bytes = 0
    print("=" * 80)
    print("[Model Parameter Memory Breakdown]")
    print("=" * 80)

    for name, p in model.named_parameters():
        if p is None:
            continue
        numel = p.numel()
        bytes_ = numel * p.element_size()
        total_bytes += bytes_

        print(
            f"{name:<70} | "
            f"shape={tuple(p.shape)!s:<20} | "
            f"dtype={str(p.dtype):<15} | "
            f"size={bytes_ / 1024**2:8.2f} MB"
        )

    print("-" * 80)
    print(f"[Total Parameter Memory] = {total_bytes / 1024**3:.2f} GB")
    print("=" * 80)

