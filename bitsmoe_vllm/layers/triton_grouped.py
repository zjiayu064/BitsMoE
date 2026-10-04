"""vLLM grouped MoE with a combined gate/up kernel launch.

Each CTA computes one projection using split-K accumulation and packed decoding. The inherited constructor initializes the packed weights and expert metadata.
"""

import torch
import triton
import triton.language as tl

from bitsmoe.algorithms.triton_moe import TritonPackedMoE, _decode_rows, _activate, _down
from bitsmoe.algorithms.vllm_compat import moe_align_block_size


@triton.jit(do_not_specialize=["ASSIGNMENTS", "G_ROW_STRIDE", "U_ROW_STRIDE"])
def _gate_up_fused(HG, HU, PG, PU, RG, RU, LG, LU,
                   Sorted, Experts, Padded, Gate, Up,
                   G_RANK: tl.constexpr, U_RANK: tl.constexpr,
                   N: tl.constexpr, ASSIGNMENTS, TOP_K: tl.constexpr,
                   G_ROW_STRIDE, U_ROW_STRIDE,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   SPLIT_K: tl.constexpr):
    projection = tl.program_id(2) // SPLIT_K
    split = tl.program_id(2) % SPLIT_K
    H = tl.where(projection == 0, HG, HU)
    Pointers = tl.where(projection == 0, PG, PU)
    Rows = tl.where(projection == 0, RG, RU)
    Lengths = tl.where(projection == 0, LG, LU)
    Output = tl.where(projection == 0, Gate, Up)
    H_RANK = tl.where(projection == 0, G_RANK, U_RANK)
    ROW_STRIDE = tl.where(projection == 0, G_ROW_STRIDE, U_ROW_STRIDE)
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
    for block in range(split * blocks, (split + 1) * blocks):
        r = block * BK + tl.arange(0, BK)
        logical = tl.load(indices + r, r < rank, 0)
        s = tl.load(singular + r, r < rank, 0).to(tl.float32)
        h = tl.load(H + (assignments[:, None] // TOP_K) * H_RANK + logical[None, :],
                    (assignments[:, None] < ASSIGNMENTS) & (r[None, :] < rank), 0)
        a = (h.to(tl.float32) * s[None, :]).to(tl.float16)
        b = _decode_rows(Pointers, Rows, expert, r, n, tl.program_id(1) * BN // 128,
                         r < rank, ROW_STRIDE, N)
        accum += tl.dot(a, b)
    tl.store(Output + (split * ASSIGNMENTS + assignments[:, None]) * N + n[None, :], accum,
             (assignments[:, None] < ASSIGNMENTS) & (n[None, :] < N))


class VllmTritonPackedMoE(TritonPackedMoE):
    def forward(self, h_gate, h_up, selected, weights, output_shards=1):
        """Return FP32 sums, optionally stored as [shard, token, rank/shards].

        HF uses the default token-major layout. Rank-major storage lets vLLM reduce-scatter spectral sums without a full activation transpose.
        """
        if output_shards < 1 or self.ranks[2] % output_shards:
            raise ValueError("Output rank must be divisible by output_shards")
        tokens, top_k = selected.shape
        if tokens >= 1024 and selected.numel() / self.expert_map.numel() >= 64:
            from bitsmoe.algorithms.triton_prefill import prefill_forward

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
        gp, gr, gl, gm = self.metadata[0]
        up_ptr, ur, ul, um = self.metadata[1]
        _gate_up_fused[(tasks, triton.cdiv(self.intermediate, gate_n), 2 * split_k)](
            h_gate, h_up, gp, up_ptr, gr, ur, gl, ul,
            sorted_ids, experts, padded, gate, up,
            self.ranks[0], self.ranks[1], self.intermediate, assignments, top_k,
            gm, um, bm, gate_n, gate_k, split_k, num_warps=4, num_stages=1)
        pointers, rows, lengths, max_rank = self.metadata[2]
        z = torch.empty((assignments, self.intermediate), dtype=torch.float16, device=h_gate.device)
        _activate[(triton.cdiv(assignments * self.intermediate, 256),)](
            gate, up, z, selected, self.expert_map, assignments, self.intermediate, 256, split_k)
        _down[(tasks, triton.cdiv(max_rank, down_n))](
            z, pointers, rows, lengths, sorted_ids, experts, padded, weights, output,
            self.ranks[2], self.intermediate, assignments, top_k, max_rank,
            bm, down_n, down_k, output_shards, num_warps=4, num_stages=1)
        return output
