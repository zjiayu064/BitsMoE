"""Packed expert matrix-vector products for a single routed token."""

import torch
import triton
import triton.language as tl
from .triton_moe import _decode_rows, _activate


@triton.jit
def _direct_gate_up(HG, HU, PG, PU, RG, RU, LG, LU, Selected, Map, Gate, Up,
                    GH: tl.constexpr, UH: tl.constexpr, GR: tl.constexpr, UR: tl.constexpr,
                    N: tl.constexpr, ASSIGNMENTS: tl.constexpr, TOP_K: tl.constexpr,
                    BR: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    assignment = tl.program_id(0)
    expert = tl.load(Map + tl.load(Selected + assignment))
    if expert < 0:
        return
    projection = tl.program_id(2) // SPLIT
    split = tl.program_id(2) % SPLIT
    H = tl.where(projection == 0, HG, HU)
    P = tl.where(projection == 0, PG, PU)
    R = tl.where(projection == 0, RG, RU)
    L = tl.where(projection == 0, LG, LU)
    Out = tl.where(projection == 0, Gate, Up)
    stride = tl.where(projection == 0, GR, UR)
    hstride = tl.where(projection == 0, GH, UH)
    rank = tl.load(L + expert)
    indices = tl.load(P + expert * 6 + 1).to(tl.pointer_type(tl.int32))
    singular = tl.load(P + expert * 6 + 5).to(tl.pointer_type(tl.float16))
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    accum = tl.zeros((BR, BN), tl.float32)
    blocks = tl.cdiv(tl.cdiv(rank, BR), SPLIT)
    for block in range(split * blocks, (split + 1) * blocks):
        r = block * BR + tl.arange(0, BR)
        logical = tl.load(indices + r, r < rank, 0)
        singulars = tl.load(singular + r, r < rank, 0).to(tl.float32)
        h = tl.load(H + assignment // TOP_K * hstride + logical, r < rank, 0)
        a = (h.to(tl.float32) * singulars).to(tl.float16)
        b = _decode_rows(P, R, expert, r, n, tl.program_id(1) * BN // 128,
                         r < rank, stride, N)
        accum += a[:, None].to(tl.float32) * b.to(tl.float32)
    value = tl.sum(accum, axis=0)
    tl.store(Out + (split * ASSIGNMENTS + assignment) * N + n, value, n < N)


@triton.jit
def _direct_down(Z, P, Rows, Lengths, Selected, Map, Weights, Out,
                  RANK_OUT: tl.constexpr, N: tl.constexpr, ASSIGNMENTS: tl.constexpr,
                  TOP_K: tl.constexpr, STRIDE: tl.constexpr, BR: tl.constexpr,
                  BN: tl.constexpr, SHARDS: tl.constexpr):
    assignment = tl.program_id(0)
    expert = tl.load(Map + tl.load(Selected + assignment))
    if expert < 0:
        return
    rank = tl.load(Lengths + expert)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    if tl.program_id(1) * BR >= rank:
        return
    accum = tl.zeros((BR, BN), tl.float32)
    for block in range(tl.cdiv(N, BN)):
        n = block * BN + tl.arange(0, BN)
        z = tl.load(Z + assignment * N + n, n < N, 0)
        w = _decode_rows(P, Rows, expert, r, n, block * BN // 128, r < rank, STRIDE, N)
        accum += w.to(tl.float32) * z[None, :].to(tl.float32)
    result = tl.sum(accum, axis=1)
    indices = tl.load(P + expert * 6 + 1).to(tl.pointer_type(tl.int32))
    singular = tl.load(P + expert * 6 + 5).to(tl.pointer_type(tl.float16))
    logical = tl.load(indices + r, r < rank, 0)
    s = tl.load(singular + r, r < rank, 0).to(tl.float32)
    weight = tl.load(Weights + assignment)
    result *= s * weight
    token = assignment // TOP_K
    if SHARDS == 1:
        address = token * RANK_OUT + logical
    else:
        width: tl.constexpr = RANK_OUT // SHARDS
        address = (logical // width * (ASSIGNMENTS // TOP_K) + token) * width + logical % width
    tl.atomic_add(Out + address, result, r < rank, sem="relaxed")


def single_token_forward(self, h_gate, h_up, selected, weights, output_shards=1):
    split, gate_br, gate_bn, down_br, down_bn, warps = 8, 32, 128, 16, 128, 4
    tokens, topk = selected.shape
    assignments = selected.numel()
    shape = ((tokens, self.ranks[2]) if output_shards == 1 else
             (output_shards, tokens, self.ranks[2] // output_shards))
    output = torch.zeros(shape, dtype=torch.float32, device=h_gate.device)
    if assignments == 0 or not self.buffers[0]:
        return output
    gate = torch.empty((split, assignments, self.intermediate), dtype=torch.float32, device=h_gate.device)
    up = torch.empty_like(gate)
    gp, gr, gl, gm = self.metadata[0]
    up_, ur, ul, um = self.metadata[1]
    _direct_gate_up[(assignments, triton.cdiv(self.intermediate, gate_bn), 2 * split)](
        h_gate, h_up, gp, up_, gr, ur, gl, ul, selected, self.expert_map, gate, up,
        self.ranks[0], self.ranks[1], gm, um, self.intermediate, assignments, topk,
        gate_br, gate_bn, split, num_warps=warps, num_stages=1)
    z = torch.empty((assignments, self.intermediate), dtype=torch.float16, device=h_gate.device)
    _activate[(triton.cdiv(assignments * self.intermediate, 256),)](
        gate, up, z, selected, self.expert_map, assignments, self.intermediate, 256, split)
    dp, dr, dl, dm = self.metadata[2]
    _direct_down[(assignments, triton.cdiv(dm, down_br))](
        z, dp, dr, dl, selected, self.expert_map, weights, output,
        self.ranks[2], self.intermediate, assignments, topk, dm,
        down_br, down_bn, output_shards, num_warps=warps, num_stages=1)
    return output
