"""Restore contiguous gate/up activations after gathering basis partitions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_gathered(Input, Gate, Up, TOKENS: tl.constexpr,
                    GATE: tl.constexpr, UP: tl.constexpr, TP: tl.constexpr,
                    BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate_element = offset < TOKENS * GATE
    local_offset = tl.where(gate_element, offset, offset - TOKENS * GATE)
    width = tl.where(gate_element, GATE, UP)
    token, column = local_offset // width, local_offset % width
    local_width = width // TP
    owner, local_column = column // local_width, column % local_width
    source = ((owner * TOKENS + token) * ((GATE + UP) // TP)
              + tl.where(gate_element, 0, GATE // TP) + local_column)
    valid = offset < TOKENS * (GATE + UP)
    value = tl.load(Input + source, valid, 0)
    tl.store(Gate + local_offset, value, valid & gate_element)
    tl.store(Up + local_offset, value, valid & ~gate_element)


def split_gathered(projected: torch.Tensor, tokens: int,
                   gate_rank: int, up_rank: int, tp: int):
    # Use one allocation and one copy kernel to produce contiguous gate/up matrices for the expert kernels.
    storage = torch.empty(tokens * (gate_rank + up_rank), dtype=projected.dtype,
                          device=projected.device)
    gate = storage[:tokens * gate_rank].view(tokens, gate_rank)
    up = storage[tokens * gate_rank:].view(tokens, up_rank)
    if storage.numel():
        _split_gathered[(triton.cdiv(storage.numel(), 1024),)](
            projected, gate, up, tokens, gate_rank, up_rank, tp, 1024)
    return gate, up
