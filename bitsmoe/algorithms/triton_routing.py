"""Normalize routing weights with a torch sum, exact division, and dtype casts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_exact(Weights, Denominator, Output, ELEMENTS,
                     TOPK: tl.constexpr, RENORM: tl.constexpr,
                     ROUND_HALF: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Weights + offset, offset < ELEMENTS, 0)
    if RENORM:
        denominator = tl.load(Denominator + offset // TOPK, offset < ELEMENTS, 1)
        # Triton 3.4 lowers tl.div_rn to div.rn.ftz.f32. Omit FTZ explicitly to preserve torch's behavior for subnormal inputs and denominators.
        values = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[values, denominator], dtype=tl.float32, is_pure=True, pack=1)
    if ROUND_HALF:
        values = values.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(Output + offset, values, offset < ELEMENTS)


def _normalize_weights(weights, renormalize, round_half):
    if not renormalize and not round_half:
        return weights
    # Compute the denominator with torch FP32 summation.
    denominator = weights.sum(dim=-1, keepdim=True) if renormalize else weights
    output = torch.empty_like(weights)
    if weights.numel():
        _normalize_exact[(triton.cdiv(weights.numel(), 256),)](
            weights, denominator, output, weights.numel(), weights.shape[-1],
            renormalize, round_half, 256, num_warps=4)
    return output


def route(logits, topk, renormalize, round_half=True, sigmoid=False):
    probabilities = logits.float().sigmoid() if sigmoid else torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, selected = torch.topk(probabilities, topk, dim=-1)
    return _normalize_weights(weights, renormalize, round_half), selected
