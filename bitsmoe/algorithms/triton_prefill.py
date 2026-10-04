"""Reuse unpacked weights and gathered activations within large prefills."""

import torch
import triton
import triton.language as tl

from .vllm_compat import moe_align_block_size

from .triton_moe import _decode_rows


@triton.jit
def _mark_active(Experts, Padded, Active, TASKS: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr):
    task = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (task < TASKS) & (task * BM < tl.load(Padded))
    expert = tl.load(Experts + task, valid, -1)
    tl.atomic_xchg(Active + expert, 1, valid & (expert >= 0), sem="relaxed")


@triton.jit(do_not_specialize=["ROW_STRIDE"])
def _unpack(Pointers, Rows, Lengths, Decoded, Active, ROW_STRIDE,
            N: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr):
    expert = tl.program_id(0)
    if tl.load(Active + expert) == 0:
        return
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    rank = tl.load(Lengths + expert)
    if tl.program_id(1) * BR >= rank:
        return
    w = _decode_rows(Pointers, Rows, expert, r, n, tl.program_id(2) * BN // 128,
                     r < rank, ROW_STRIDE, N)
    tl.store(Decoded + (expert * ROW_STRIDE + r[:, None]) * N + n[None, :],
             w, (r[:, None] < rank) & (n[None, :] < N))


@triton.jit(do_not_specialize=["ROW_STRIDE", "ASSIGNMENTS"])
def _gather(H, Pointers, Lengths, Sorted, Experts, Padded, Prepared, ROW_STRIDE, ASSIGNMENTS,
            H_RANK: tl.constexpr, TOP_K: tl.constexpr, BM: tl.constexpr, BR: tl.constexpr):
    task = tl.program_id(0)
    if task * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + task)
    if expert < 0:
        return
    rank = tl.load(Lengths + expert)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    assignments = tl.load(Sorted + task * BM + tl.arange(0, BM))
    indices = tl.load(Pointers + expert * 6 + 1).to(tl.pointer_type(tl.int32))
    singular = tl.load(Pointers + expert * 6 + 5).to(tl.pointer_type(tl.float16))
    logical = tl.load(indices + r, r < rank, 0)
    scale = tl.load(singular + r, r < rank, 0)
    h = tl.load(H + (assignments[:, None] // TOP_K) * H_RANK + logical[None, :],
                (assignments[:, None] < ASSIGNMENTS) & (r[None, :] < rank), 0)
    position = task * BM + tl.arange(0, BM)
    tl.store(Prepared + position[:, None] * ROW_STRIDE + r[None, :],
             h.to(tl.float32) * scale[None, :].to(tl.float32), r[None, :] < ROW_STRIDE)


@triton.jit(do_not_specialize=["ASSIGNMENTS", "ROW_STRIDE"])
def _prepared_gate_up(H, Pointers, Rows, Lengths, Sorted, Experts, Padded, Output, Decoded,
             H_RANK: tl.constexpr, N: tl.constexpr, ASSIGNMENTS,
             TOP_K: tl.constexpr, ROW_STRIDE, Gate, ACTIVATE: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             SPLIT_K: tl.constexpr):
    task = tl.program_id(0)
    if task * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + task)
    if expert < 0:
        return
    assignments = tl.load(Sorted + task * BM + tl.arange(0, BM))
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rank = tl.load(Lengths + expert)
    accum = tl.zeros((BM, BN), tl.float32)
    blocks = tl.cdiv(tl.cdiv(rank, BK), SPLIT_K)
    for block in range(tl.program_id(2) * blocks, (tl.program_id(2) + 1) * blocks):
        r = block * BK + tl.arange(0, BK)
        position = task * BM + tl.arange(0, BM)
        a = tl.load(H + position[:, None] * ROW_STRIDE + r[None, :],
                    r[None, :] < rank, 0)
        b = tl.load(Decoded + (expert * ROW_STRIDE + r[:, None]) * N + n[None, :],
                    (r[:, None] < rank) & (n[None, :] < N), 0)
        accum += tl.dot(a, b)
    offset = (task * BM + tl.arange(0, BM))[:, None] * N + n[None, :]
    if ACTIVATE:
        gate = tl.load(Gate + offset, n[None, :] < N, 0)
        accum = gate * tl.sigmoid(gate) * accum
    tl.store(Output + offset, accum, n[None, :] < N)


@triton.jit(do_not_specialize=["ASSIGNMENTS", "ROW_STRIDE"])
def _decoded_down(Z, Pointers, Rows, Lengths, Sorted, Experts, Padded, Weights, Output, Decoded,
          RANK_OUT: tl.constexpr, N: tl.constexpr, ASSIGNMENTS,
          TOP_K: tl.constexpr, ROW_STRIDE,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          OUTPUT_SHARDS: tl.constexpr = 1):
    task = tl.program_id(0)
    if task * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + task)
    if expert < 0:
        return
    rank = tl.load(Lengths + expert)
    r = tl.program_id(1) * BN + tl.arange(0, BN)
    if tl.program_id(1) * BN >= rank:
        return
    assignments = tl.load(Sorted + task * BM + tl.arange(0, BM))
    accum = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(N, BK)):
        n = block * BK + tl.arange(0, BK)
        offset = (task * BM + tl.arange(0, BM))[:, None] * N + n[None, :]
        mask = (assignments[:, None] < ASSIGNMENTS) & (n[None, :] < N)
        z = tl.load(Z + offset, mask, 0)
        w = tl.load(Decoded + (expert * ROW_STRIDE + r[:, None]) * N + n[None, :],
                    (r[:, None] < rank) & (n[None, :] < N), 0)
        accum += tl.dot(z, tl.trans(w))
    indices = tl.load(Pointers + expert * 6 + 1).to(tl.pointer_type(tl.int32))
    singular = tl.load(Pointers + expert * 6 + 5).to(tl.pointer_type(tl.float16))
    logical = tl.load(indices + r, r < rank, 0)
    s = tl.load(singular + r, r < rank, 0).to(tl.float32)
    weights = tl.load(Weights + assignments, assignments < ASSIGNMENTS, 0)
    result = accum * s[None, :] * weights[:, None]
    token = assignments[:, None] // TOP_K
    if OUTPUT_SHARDS == 1:
        address = token * RANK_OUT + logical[None, :]
    else:
        width = RANK_OUT // OUTPUT_SHARDS
        address = ((logical[None, :] // width) * (ASSIGNMENTS // TOP_K) + token) * width + logical[None, :] % width
    tl.atomic_add(Output + address,
                  result, (assignments[:, None] < ASSIGNMENTS) & (r[None, :] < rank),
                  sem="relaxed")


def prefill_forward(self, h_gate, h_up, selected, weights, output_shards=1):
    tokens, top_k = selected.shape
    shape = ((tokens, self.ranks[2]) if output_shards == 1 else (output_shards, tokens, self.ranks[2] // output_shards))
    output = torch.zeros(shape, dtype=torch.float32, device=h_gate.device)
    if tokens == 0 or not self.buffers[0]:
        return output
    bm, split_k = (32 if tokens < 2048 else 64), 1
    gate_n, gate_k = 128, 64
    down_n, down_k = (64 if tokens < 2048 else 128), 64
    sorted_ids, experts, padded = moe_align_block_size(
        selected, bm, self.expert_map.numel(), self.expert_map, pad_sorted_ids=True)
    tasks = experts.numel()
    active = torch.zeros(len(self.buffers[0]), device=h_gate.device, dtype=torch.int32)
    _mark_active[(triton.cdiv(tasks, 256),)](experts, padded, active, tasks, bm, 256)
    assignments = selected.numel()
    gate = torch.empty((tasks * bm, self.intermediate),
                       dtype=torch.float32, device=h_gate.device)
    up = torch.empty_like(gate, dtype=torch.float16)
    max_rows = max(m[3] for m in self.metadata)
    decoded = torch.empty((len(self.buffers[0]), max_rows, self.intermediate),
                          dtype=torch.float16, device=h_gate.device)
    prepared = torch.empty((tasks * bm, max_rows), dtype=torch.float16, device=h_gate.device)
    for i, (h, out) in enumerate(((h_gate, gate), (h_up, up))):
        pointers, rows, lengths, max_rank = self.metadata[i]
        _unpack[(len(self.buffers[0]), triton.cdiv(max_rank, 32), triton.cdiv(self.intermediate, 128))](
            pointers, rows, lengths, decoded, active, max_rank, self.intermediate, 32, 128,
            num_warps=4)
        _gather[(tasks, triton.cdiv(max_rank, 128))](
            h, pointers, lengths, sorted_ids, experts, padded, prepared, max_rank,
            assignments, self.ranks[i], top_k, bm, 128)
        _prepared_gate_up[(tasks, triton.cdiv(self.intermediate, gate_n), split_k)](
            prepared, pointers, rows, lengths, sorted_ids, experts, padded, out, decoded,
            self.ranks[i], self.intermediate, assignments, top_k, max_rank, gate, i == 1,
            bm, gate_n, gate_k, split_k, num_warps=4, num_stages=3)
    pointers, rows, lengths, max_rank = self.metadata[2]
    z = up
    _unpack[(len(self.buffers[0]), triton.cdiv(max_rank, 32), triton.cdiv(self.intermediate, 128))](
        pointers, rows, lengths, decoded, active, max_rank, self.intermediate, 32, 128,
        num_warps=4)
    _decoded_down[(tasks, triton.cdiv(max_rank, down_n))](
        z, pointers, rows, lengths, sorted_ids, experts, padded, weights, output, decoded,
        self.ranks[2], self.intermediate, assignments, top_k, max_rank,
        bm, down_n, down_k, output_shards, num_warps=4, num_stages=3)
    return output
