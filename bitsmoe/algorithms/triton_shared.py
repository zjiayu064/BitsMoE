"""Prefill shared-basis multiplication from the existing W8 Marlin layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_w8(W, S, Out, K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r, n = offset // N, offset % N
    row, col = r % 16, n % 16
    lane = (col % 8) * 4 + (row % 8) // 2
    entry = (col // 8) * 4 + (row // 8) * 2 + row % 2
    # Invert the 16x64 Marlin tile permutation and its byte interleave.
    slot = lane * 32 + (n % 64 // 16) * 8 + entry
    permuted = slot // 4 * 4 + (slot % 4) // 2 + (slot % 2) * 2
    byte = (r // 16) * N * 16 + (n // 64) * 1024 + permuted
    word = tl.load(W + byte // 4, offset < K * N, 0).to(tl.uint32)
    quant = ((word >> ((byte % 4) * 8)) & 255).to(tl.int32) - 128
    if K > 128:
        scale_col = (n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8
    else:
        scale_col = (n // 32) * 32 + (n % 8 // 2) * 8 + (n % 32 // 8) * 2 + n % 2
    scale = tl.load(S + (r // 128) * N + scale_col, offset < K * N, 0)
    tl.store(Out + offset, quant.to(tl.float32) * scale.to(tl.float32), offset < K * N)


@triton.jit
def _unpack_w8_tiles(W, S, Out, K: tl.constexpr, N: tl.constexpr):
    # One CTA owns a complete 16x64 packed tile. Keep neighboring packed bytes in the same CTA instead of revisiting the tile for each output row.
    tiles_n: tl.constexpr = triton.cdiv(N, 64)
    r = (tl.program_id(0) // tiles_n) * 16 + tl.arange(0, 16)
    n = (tl.program_id(0) % tiles_n) * 64 + tl.arange(0, 64)
    row, col = r[:, None] % 16, n[None, :] % 16
    lane = (col % 8) * 4 + (row % 8) // 2
    entry = (col // 8) * 4 + (row // 8) * 2 + row % 2
    slot = lane * 32 + (n[None, :] % 64 // 16) * 8 + entry
    permuted = slot // 4 * 4 + (slot % 4) // 2 + (slot % 2) * 2
    byte = (r[:, None] // 16) * N * 16 + (n[None, :] // 64) * 1024 + permuted
    valid = (r[:, None] < K) & (n[None, :] < N)
    word = tl.load(W + byte // 4, valid, 0).to(tl.uint32)
    quant = ((word >> ((byte % 4) * 8)) & 255).to(tl.int32) - 128
    if K > 128:
        scale_col = (n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8
    else:
        scale_col = (n // 32) * 32 + (n % 8 // 2) * 8 + (n % 32 // 8) * 2 + n % 2
    scale = tl.load(S + (r[:, None] // 128) * N + scale_col[None, :], valid, 0)
    tl.store(Out + r[:, None] * N + n[None, :], quant.to(tl.float32) * scale.to(tl.float32), valid)


def unpack_w8(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    k, n = scales.shape[0] * 128, scales.shape[1]
    output = torch.empty((k, n), device=weight.device, dtype=torch.float16)
    # Use flat unpacking for small matrices and complete packed tiles for larger matrices.
    if k * n >= 4 * 1024 * 1024:
        _unpack_w8_tiles[(triton.cdiv(k, 16) * triton.cdiv(n, 64),)](weight, scales, output, k, n)
    else:
        _unpack_w8[(triton.cdiv(k * n, 1024),)](weight, scales, output, k, n, 1024)
    return output


def prefill_shared_linear(x: torch.Tensor, weight: torch.Tensor,
                          scales: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16) @ unpack_w8(weight, scales)
