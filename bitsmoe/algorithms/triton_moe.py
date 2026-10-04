"""Grouped matrix multiplication over the BitsMoE packed weight layout."""

import torch
import triton
import triton.language as tl

from .vllm_compat import moe_align_block_size


@triton.jit
def _decode_rows(Pointers, Rows, expert, row, col, tile_col, valid_rows,
                 ROW_STRIDE, WIDTH: tl.constexpr):
    payload = tl.load(Pointers + expert * 6).to(tl.pointer_type(tl.uint32))
    tiles = tl.load(Pointers + expert * 6 + 2).to(tl.pointer_type(tl.int32))
    scales = tl.load(Pointers + expert * 6 + 4).to(tl.pointer_type(tl.float16))
    desc = Rows + (expert * ROW_STRIDE + row) * 4
    bits = tl.load(desc, valid_rows, 0)
    slab_row = tl.load(desc + 1, valid_rows, 0)
    slab_rows = tl.load(desc + 2, valid_rows, 0)
    tile = tl.load(desc + 3, valid_rows, 0)
    tile_ids = tile + tile_col
    mask = valid_rows[:, None] & (col[None, :] < WIDTH)
    packed_offset = tl.load(tiles + tile_ids * 3, valid_rows, 0)
    scale_offset = tl.load(tiles + tile_ids * 3 + 1, valid_rows, 0)
    stage = slab_row // 16
    stage_rows = tl.minimum(16, slab_rows - stage * 16)
    lane = col % 32
    base = (packed_offset[:, None] + stage[:, None] * 64 * bits[:, None]
            + (col[None, :] % 128 // 32) * stage_rows[:, None] * bits[:, None]
            + (slab_row[:, None] % 16) * bits[:, None])
    value = tl.full((row.shape[0], col.shape[0]), 0, tl.int32)
    for plane in range(tl.max(tl.where(bits < 16, bits, 0), axis=0)):
        word = tl.load(payload + base + plane,
                       mask & (plane < bits[:, None]) & (bits[:, None] < 16), 0)
        value |= ((word >> lane[None, :]) & 1).to(tl.int32) << plane
    quant = tl.where(bits[:, None] == 1, value * 2 - 1,
                     value - (1 << (tl.maximum(bits[:, None], 1) - 1)))
    scale = tl.load(scales + scale_offset + slab_row,
                    valid_rows & (bits < 16), 0).to(tl.float32)
    word = tl.load(payload + base + lane[None, :] // 2,
                   mask & (bits[:, None] == 16), 0)
    half_bits = ((word >> ((lane[None, :] % 2) * 16)) & 65535).to(tl.uint16)
    value_fp16 = half_bits.to(tl.float16, bitcast=True)
    return tl.where(bits[:, None] == 16, value_fp16,
                    (quant.to(tl.float32) * scale[:, None]).to(tl.float16))


@triton.jit(do_not_specialize=["ASSIGNMENTS", "ROW_STRIDE"])
def _gate_up(H, Pointers, Rows, Lengths, Sorted, Experts, Padded, Output,
             H_RANK: tl.constexpr, N: tl.constexpr, ASSIGNMENTS,
             TOP_K: tl.constexpr, ROW_STRIDE,
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
    indices = tl.load(Pointers + expert * 6 + 1).to(tl.pointer_type(tl.int32))
    singular = tl.load(Pointers + expert * 6 + 5).to(tl.pointer_type(tl.float16))
    accum = tl.zeros((BM, BN), tl.float32)
    blocks = tl.cdiv(tl.cdiv(rank, BK), SPLIT_K)
    for block in range(tl.program_id(2) * blocks, (tl.program_id(2) + 1) * blocks):
        r = block * BK + tl.arange(0, BK)
        logical = tl.load(indices + r, r < rank, 0)
        s = tl.load(singular + r, r < rank, 0).to(tl.float32)
        h = tl.load(H + (assignments[:, None] // TOP_K) * H_RANK + logical[None, :],
                    (assignments[:, None] < ASSIGNMENTS) & (r[None, :] < rank), 0)
        a = (h.to(tl.float32) * s[None, :]).to(tl.float16)
        b = _decode_rows(Pointers, Rows, expert, r, n, tl.program_id(1) * BN // 128,
                         r < rank, ROW_STRIDE, N)
        accum += tl.dot(a, b)
    tl.store(Output + (tl.program_id(2) * ASSIGNMENTS + assignments[:, None]) * N + n[None, :], accum,
             (assignments[:, None] < ASSIGNMENTS) & (n[None, :] < N))


@triton.jit(do_not_specialize=["ASSIGNMENTS"])
def _activate(Gate, Up, Z, Selected, Map,
              ASSIGNMENTS, N: tl.constexpr, BLOCK: tl.constexpr,
              SPLIT_K: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < ASSIGNMENTS * N
    expert = tl.load(Selected + offset // N, valid, 0)
    local = tl.load(Map + expert, valid, -1)
    gate = tl.full((BLOCK,), 0, tl.float32)
    up = tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(SPLIT_K):
        address = split * ASSIGNMENTS * N + offset
        gate += tl.load(Gate + address, valid & (local >= 0), 0)
        up += tl.load(Up + address, valid & (local >= 0), 0)
    tl.store(Z + offset, gate * tl.sigmoid(gate) * up, offset < ASSIGNMENTS * N)


@triton.jit(do_not_specialize=["ASSIGNMENTS", "ROW_STRIDE"])
def _down(Z, Pointers, Rows, Lengths, Sorted, Experts, Padded, Weights, Output,
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
        offset = assignments[:, None] * N + n[None, :]
        mask = (assignments[:, None] < ASSIGNMENTS) & (n[None, :] < N)
        z = tl.load(Z + offset, mask, 0)
        w = _decode_rows(Pointers, Rows, expert, r, n, block * BK // 128,
                         r < rank, ROW_STRIDE, N)
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


class TritonPackedMoE:
    def __init__(self, buffers, expert_map, gate_rank, up_rank, output_rank, intermediate_size):
        self.buffers = buffers
        self.expert_map = expert_map
        self.ranks = (gate_rank, up_rank, output_rank)
        self.intermediate = intermediate_size
        self.metadata = []
        device = expert_map.device
        for projection in range(3):
            group = buffers[projection * 6:(projection + 1) * 6]
            lengths = [x.numel() for x in group[1]]
            max_rank = max(lengths, default=0)
            rows = torch.zeros((len(lengths), max_rank, 4), dtype=torch.int32)
            for e, slabs in enumerate(group[3]):
                for bits, count, offset, tile in slabs.cpu().tolist():
                    rows[e, offset:offset + count, 0] = bits
                    rows[e, offset:offset + count, 1] = torch.arange(count)
                    rows[e, offset:offset + count, 2] = count
                    rows[e, offset:offset + count, 3] = tile
            pointers = torch.tensor([[group[i][e].data_ptr() for i in range(6)]
                                     for e in range(len(lengths))], dtype=torch.int64, device=device)
            self.metadata.append((pointers, rows.to(device),
                                  torch.tensor(lengths, dtype=torch.int32, device=device), max_rank))

    def forward(self, h_gate, h_up, selected, weights, output_shards=1):
        """Return FP32 sums, optionally stored as [shard, token, rank/shards].

        HF uses the default token-major layout. Rank-major storage lets vLLM reduce-scatter spectral sums without a full activation transpose.
        """
        if output_shards < 1 or self.ranks[2] % output_shards:
            raise ValueError("Output rank must be divisible by output_shards")
        tokens, top_k = selected.shape
        if tokens >= 1024 and selected.numel() / self.expert_map.numel() >= 64:
            from .triton_prefill import prefill_forward

            return prefill_forward(self, h_gate, h_up, selected, weights, output_shards)
        shape = ((tokens, self.ranks[2]) if output_shards == 1 else (output_shards, tokens, self.ranks[2] // output_shards))
        output = torch.zeros(shape, dtype=torch.float32, device=h_gate.device)
        if tokens == 0 or not self.buffers[0]:
            return output
        average_routes = selected.numel() / self.expert_map.numel()
        bm = 64 if average_routes >= 32 else 32 if average_routes >= 16 else 16
        split_k = 8 if average_routes <= 0.5 else 2 if average_routes <= 8 else 1
        gate_n, gate_k = 128, 32
        down_n, down_k = 32, 128
        sorted_ids, experts, padded = moe_align_block_size(
            selected, bm, self.expert_map.numel(), self.expert_map, pad_sorted_ids=True)
        # Every nonempty aligned block contains at least one routed assignment. vLLM reserves padding for *all* experts; during decode most of those blocks cannot exist. Bound the grid without reading GPU counts back.
        tasks = min(experts.numel(), selected.numel())
        assignments = selected.numel()
        gate = torch.empty((split_k, assignments, self.intermediate),
                           dtype=torch.float32, device=h_gate.device)
        up = torch.empty_like(gate)
        for i, (h, out) in enumerate(((h_gate, gate), (h_up, up))):
            pointers, rows, lengths, max_rank = self.metadata[i]
            _gate_up[(tasks, triton.cdiv(self.intermediate, gate_n), split_k)](
                h, pointers, rows, lengths, sorted_ids, experts, padded, out,
                self.ranks[i], self.intermediate, assignments, top_k, max_rank,
                bm, gate_n, gate_k, split_k, num_warps=4, num_stages=1)
        pointers, rows, lengths, max_rank = self.metadata[2]
        z = torch.empty((assignments, self.intermediate), dtype=torch.float16, device=h_gate.device)
        _activate[(triton.cdiv(assignments * self.intermediate, 256),)](
            gate, up, z, selected, self.expert_map, assignments, self.intermediate, 256, split_k)
        _down[(tasks, triton.cdiv(max_rank, down_n))](
            z, pointers, rows, lengths, sorted_ids, experts, padded, weights, output,
            self.ranks[2], self.intermediate, assignments, top_k, max_rank,
            bm, down_n, down_k, output_shards, num_warps=4, num_stages=1)
        return output
