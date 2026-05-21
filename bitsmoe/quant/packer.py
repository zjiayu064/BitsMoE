import os
import time

import torch
from torch import Tensor
from typing import Dict, List, Tuple
from collections import defaultdict

from .quantizer import quantize_symmetric, pack_quantized, unpack_quantized
from bitsmoe.utils.logger import setup_logger


K_SLAB = 64
N_TILE = 128
K_STAGE = 16
N_MICRO = 32
MICROS_PER_TILE = N_TILE // N_MICRO
_SUPPORTED_BITS = {1, 2, 3, 4, 6, 8, 16}


def _to_compact_int_tensor(rows: List[List[int]]) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), dtype=torch.int16)

    max_abs = 0
    for r in rows:
        for v in r:
            max_abs = max(max_abs, abs(int(v)))

    dtype = torch.int16 if max_abs <= 32767 else torch.int32
    return torch.tensor(rows, dtype=dtype)


def _words_per_row(bits: int) -> int:
    if bits in {1, 2, 3, 4, 6, 8}:
        return bits
    if bits == 16:
        # 32 fp16 values per row-chunk -> 16 uint32 words
        return 16
    raise ValueError(f"Unsupported bits={bits}")


def _pack_lowbit_tile_payload(tile_q: Tensor, bits: int) -> Tensor:
    """
    Pack one tile [k_slab, n_valid] int8 into 1-D uint32 payload.

    Physical order in payload:
      [stage][microtile][row][word]
    with fixed stage/microtile traversal.
    """
    if tile_q.dim() != 2 or tile_q.dtype != torch.int8:
        raise ValueError("tile_q must be 2-D int8")

    k_slab, n_valid = tile_q.shape
    chunks: List[Tensor] = []

    for stage_row in range(0, k_slab, K_STAGE):
        rows_valid = min(K_STAGE, k_slab - stage_row)

        for micro in range(MICROS_PER_TILE):
            c0 = micro * N_MICRO
            c1 = min(c0 + N_MICRO, n_valid)

            block = torch.zeros((rows_valid, N_MICRO), dtype=torch.int8, device=tile_q.device)
            if c1 > c0:
                block[:, : (c1 - c0)] = tile_q[stage_row:stage_row + rows_valid, c0:c1]

            packed, orig_size = pack_quantized(block, bits)
            if int(orig_size) != N_MICRO:
                raise RuntimeError(f"Expected original_size={N_MICRO}, got {orig_size}")

            chunks.append(packed.reshape(-1).to(torch.uint32))

    if not chunks:
        return torch.empty((0,), dtype=torch.uint32, device=tile_q.device)
    return torch.cat(chunks, dim=0)


def _unpack_lowbit_tile_payload(payload_tile: Tensor, bits: int, k_slab: int, device: torch.device) -> Tensor:
    """
    Decode payload of one tile into [k_slab, 128] int8 (padded columns included).
    """
    out = torch.empty((k_slab, N_TILE), dtype=torch.int8, device=device)
    ptr = 0
    words_row = _words_per_row(bits)

    for stage_row in range(0, k_slab, K_STAGE):
        rows_valid = min(K_STAGE, k_slab - stage_row)

        for micro in range(MICROS_PER_TILE):
            words = rows_valid * words_row
            packed = payload_tile[ptr:ptr + words].reshape(rows_valid, words_row)
            ptr += words

            block = unpack_quantized(packed, bits, N_MICRO)
            c0 = micro * N_MICRO
            out[stage_row:stage_row + rows_valid, c0:c0 + N_MICRO] = block

    if ptr != int(payload_tile.numel()):
        raise RuntimeError("Low-bit tile payload decode pointer mismatch")
    return out


def _pack_fp16_tile_payload(tile_fp16: Tensor) -> Tensor:
    """
    Pack one fp16 tile [k_slab, n_valid] to 1-D uint32 payload in [stage][micro][row][word] order.

    Row-chunk [1, 32] -> 16 uint32 words by bit-casting two fp16 into one uint32.
    """
    if tile_fp16.dim() != 2 or tile_fp16.dtype != torch.float16:
        raise ValueError("tile_fp16 must be 2-D float16")

    k_slab, n_valid = tile_fp16.shape
    chunks: List[Tensor] = []

    for stage_row in range(0, k_slab, K_STAGE):
        rows_valid = min(K_STAGE, k_slab - stage_row)

        for micro in range(MICROS_PER_TILE):
            c0 = micro * N_MICRO
            c1 = min(c0 + N_MICRO, n_valid)

            block = torch.zeros((rows_valid, N_MICRO), dtype=torch.float16, device=tile_fp16.device)
            if c1 > c0:
                block[:, : (c1 - c0)] = tile_fp16[stage_row:stage_row + rows_valid, c0:c1]

            # [rows, 32] fp16 -> [rows, 32] uint16
            u16 = block.view(torch.uint16)
            lo64 = u16[:, 0::2].to(torch.int64)
            hi64 = u16[:, 1::2].to(torch.int64)
            words = (lo64 | (hi64 << 16)).to(torch.uint32)
            chunks.append(words.reshape(-1))

    if not chunks:
        return torch.empty((0,), dtype=torch.uint32, device=tile_fp16.device)
    return torch.cat(chunks, dim=0)


def _unpack_fp16_tile_payload(payload_tile: Tensor, k_slab: int, device: torch.device) -> Tensor:
    out = torch.empty((k_slab, N_TILE), dtype=torch.float16, device=device)
    ptr = 0
    words_row = _words_per_row(16)

    for stage_row in range(0, k_slab, K_STAGE):
        rows_valid = min(K_STAGE, k_slab - stage_row)

        for micro in range(MICROS_PER_TILE):
            words = rows_valid * words_row
            flat = payload_tile[ptr:ptr + words].reshape(rows_valid, words_row)
            ptr += words

            flat64 = flat.to(torch.int64)
            lo = (flat64 & 0xFFFF).to(torch.uint16)
            hi = ((flat64 >> 16) & 0xFFFF).to(torch.uint16)

            u16 = torch.empty((rows_valid, N_MICRO), dtype=torch.uint16, device=device)
            u16[:, 0::2] = lo
            u16[:, 1::2] = hi
            block = u16.view(torch.float16)

            c0 = micro * N_MICRO
            out[stage_row:stage_row + rows_valid, c0:c0 + N_MICRO] = block

    if ptr != int(payload_tile.numel()):
        raise RuntimeError("fp16 tile payload decode pointer mismatch")
    return out


def _reorder_packed_rows_to_runtime_layout(
    packed_rows: Tensor,
    k_slab: int,
    words_per_row: int,
) -> Tensor:
    """
    Convert [tile, row, micro, word] to runtime order [tile, stage, micro, row, word].
    """
    if packed_rows.dim() != 4:
        raise ValueError(f"packed_rows must be 4-D, got shape={tuple(packed_rows.shape)}")
    if int(packed_rows.shape[1]) != k_slab:
        raise ValueError("k_slab mismatch in packed_rows")
    if int(packed_rows.shape[2]) != MICROS_PER_TILE:
        raise ValueError("MICROS_PER_TILE mismatch in packed_rows")
    if int(packed_rows.shape[3]) != words_per_row:
        raise ValueError("words_per_row mismatch in packed_rows")

    num_tiles = int(packed_rows.shape[0])
    if num_tiles == 0:
        return torch.empty((0, 0), dtype=torch.uint32, device=packed_rows.device)

    if k_slab % K_STAGE == 0:
        num_stages = k_slab // K_STAGE
        return (
            packed_rows
            .view(num_tiles, num_stages, K_STAGE, MICROS_PER_TILE, words_per_row)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(num_tiles, -1)
        )

    # Tail slab path: keep exact stage traversal semantics.
    chunks: List[Tensor] = []
    for stage_row in range(0, k_slab, K_STAGE):
        rows_valid = min(K_STAGE, k_slab - stage_row)
        chunk = packed_rows[:, stage_row:stage_row + rows_valid, :, :]
        chunk = chunk.permute(0, 2, 1, 3).contiguous().view(num_tiles, -1)
        chunks.append(chunk)
    return torch.cat(chunks, dim=1)


def _pack_lowbit_slab_payload_and_scale(
    slab_rank: Tensor,
    bits: int,
    quantize_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] | None = None,
) -> Tuple[Tensor, Tensor]:
    """
    Pack one slab [k_slab, dim] low-bit quantized payload in tile-major order.

    Returns:
        payload_tiles: [n_tiles, words_per_tile] uint32
        scale_flat:    [n_tiles * k_slab] fp16 in [tile][local_rank] order
    """
    if slab_rank.dim() != 2:
        raise ValueError("slab_rank must be 2-D")

    device = slab_rank.device
    # Upcast to fp32 for quantize_symmetric so that iterative MSE scale
    # refinement (bits in {2,3,4}) and clamp(min=eps) stay numerically stable.
    # fp16 inputs would underflow eps=1e-8 to 0 and increase quantization error.
    if slab_rank.dtype != torch.float32:
        slab_rank = slab_rank.to(torch.float32)
    k_slab, dim = slab_rank.shape
    n_tiles = (dim + N_TILE - 1) // N_TILE
    full_tiles = dim // N_TILE
    rem = dim - full_tiles * N_TILE

    payload_chunks: List[Tensor] = []
    scale_chunks: List[Tensor] = []
    if full_tiles > 0:
        full = slab_rank[:, :full_tiles * N_TILE].contiguous().view(k_slab, full_tiles, N_TILE)
        q_start = None
        q_end = None
        if quantize_events is not None:
            q_start = torch.cuda.Event(enable_timing=True)
            q_end = torch.cuda.Event(enable_timing=True)
            q_start.record()
        q_full, s_full = quantize_symmetric(full, bits=bits, dim=2)
        if q_start is not None and q_end is not None:
            q_end.record()
            quantize_events.append((q_start, q_end))

        # Batch pack all full tiles with a single kernel launch.
        q_pack = q_full.permute(1, 0, 2).contiguous().view(full_tiles * k_slab, N_TILE)
        packed_full, orig_size = pack_quantized(q_pack, bits)
        if int(orig_size) != N_TILE:
            raise RuntimeError(f"Expected original_size={N_TILE}, got {orig_size}")

        packed_full = packed_full.view(full_tiles, k_slab, MICROS_PER_TILE, bits).to(torch.uint32)
        payload_full = _reorder_packed_rows_to_runtime_layout(packed_full, k_slab, bits)
        payload_chunks.append(payload_full)

        # [k_slab, full_tiles] -> [full_tiles, k_slab] tile-major.
        scale_full = s_full.transpose(0, 1).contiguous().reshape(-1).to(torch.float16)
        scale_chunks.append(scale_full)

    if rem > 0:
        tail = slab_rank[:, full_tiles * N_TILE:].contiguous()
        q_start = None
        q_end = None
        if quantize_events is not None:
            q_start = torch.cuda.Event(enable_timing=True)
            q_end = torch.cuda.Event(enable_timing=True)
            q_start.record()
        q_tail, s_tail = quantize_symmetric(tail, bits=bits, dim=1)
        if q_start is not None and q_end is not None:
            q_end.record()
            quantize_events.append((q_start, q_end))

        q_pad = torch.zeros((k_slab, N_TILE), dtype=torch.int8, device=device)
        q_pad[:, :rem] = q_tail
        packed_tail, orig_size = pack_quantized(q_pad, bits)
        if int(orig_size) != N_TILE:
            raise RuntimeError(f"Expected original_size={N_TILE}, got {orig_size}")

        packed_tail = packed_tail.view(1, k_slab, MICROS_PER_TILE, bits).to(torch.uint32)
        payload_tail = _reorder_packed_rows_to_runtime_layout(packed_tail, k_slab, bits)
        payload_chunks.append(payload_tail)
        scale_chunks.append(s_tail.reshape(-1).to(torch.float16))

    if payload_chunks:
        payload_tiles = torch.cat(payload_chunks, dim=0)
        scale_flat = torch.cat(scale_chunks, dim=0)
    else:
        words_per_tile = k_slab * MICROS_PER_TILE * bits
        payload_tiles = torch.empty((0, words_per_tile), dtype=torch.uint32, device=device)
        scale_flat = torch.empty((0,), dtype=torch.float16, device=device)

    if int(payload_tiles.shape[0]) != n_tiles:
        raise RuntimeError(
            f"payload tile count mismatch: expected {n_tiles}, got {int(payload_tiles.shape[0])}"
        )
    if int(scale_flat.numel()) != n_tiles * k_slab:
        raise RuntimeError(
            f"scale size mismatch: expected {n_tiles * k_slab}, got {int(scale_flat.numel())}"
        )

    return payload_tiles, scale_flat


def _pack_fp16_slab_payload_and_scale(
    slab_rank: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Pack one slab [k_slab, dim] fp16 payload in tile-major order.

    Returns:
        payload_tiles: [n_tiles, words_per_tile] uint32
        scale_flat:    [n_tiles * k_slab] fp16 ones
    """
    if slab_rank.dim() != 2:
        raise ValueError("slab_rank must be 2-D")

    device = slab_rank.device
    k_slab, dim = slab_rank.shape
    n_tiles = (dim + N_TILE - 1) // N_TILE
    full_tiles = dim // N_TILE
    rem = dim - full_tiles * N_TILE

    payload_chunks: List[Tensor] = []
    scale_chunks: List[Tensor] = []

    if full_tiles > 0:
        full = slab_rank[:, :full_tiles * N_TILE].to(torch.float16).contiguous().view(k_slab, full_tiles, N_TILE)
        u16 = full.view(torch.uint16).view(k_slab, full_tiles, MICROS_PER_TILE, N_MICRO)
        lo64 = u16[..., 0::2].to(torch.int64)
        hi64 = u16[..., 1::2].to(torch.int64)
        words = (lo64 | (hi64 << 16)).to(torch.uint32)  # [k_slab, full_tiles, 4, 16]

        packed_full = words.permute(1, 0, 2, 3).contiguous()  # [full_tiles, k_slab, 4, 16]
        payload_full = _reorder_packed_rows_to_runtime_layout(packed_full, k_slab, 16)
        payload_chunks.append(payload_full)
        scale_chunks.append(torch.ones((full_tiles * k_slab,), dtype=torch.float16, device=device))

    if rem > 0:
        tail = torch.zeros((k_slab, N_TILE), dtype=torch.float16, device=device)
        tail[:, :rem] = slab_rank[:, full_tiles * N_TILE:].to(torch.float16)

        u16 = tail.view(torch.uint16).view(k_slab, MICROS_PER_TILE, N_MICRO)
        lo64 = u16[..., 0::2].to(torch.int64)
        hi64 = u16[..., 1::2].to(torch.int64)
        words = (lo64 | (hi64 << 16)).to(torch.uint32)  # [k_slab, 4, 16]

        packed_tail = words.unsqueeze(0).contiguous()  # [1, k_slab, 4, 16]
        payload_tail = _reorder_packed_rows_to_runtime_layout(packed_tail, k_slab, 16)
        payload_chunks.append(payload_tail)
        scale_chunks.append(torch.ones((k_slab,), dtype=torch.float16, device=device))

    if payload_chunks:
        payload_tiles = torch.cat(payload_chunks, dim=0)
        scale_flat = torch.cat(scale_chunks, dim=0)
    else:
        words_per_tile = k_slab * MICROS_PER_TILE * 16
        payload_tiles = torch.empty((0, words_per_tile), dtype=torch.uint32, device=device)
        scale_flat = torch.empty((0,), dtype=torch.float16, device=device)

    if int(payload_tiles.shape[0]) != n_tiles:
        raise RuntimeError(
            f"payload tile count mismatch: expected {n_tiles}, got {int(payload_tiles.shape[0])}"
        )
    if int(scale_flat.numel()) != n_tiles * k_slab:
        raise RuntimeError(
            f"scale size mismatch: expected {n_tiles * k_slab}, got {int(scale_flat.numel())}"
        )

    return payload_tiles, scale_flat


def validate_runtime_layout(
    rank_major_ref: Tensor,
    slab_meta_buffer: Tensor,
    tile_meta_buffer: Tensor,
    rank_idx_buffer: Tensor,
    payload_buffer: Tensor,
    scale_buffer: Tensor,
    groupsize: int = N_TILE,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """
    Validate runtime buffers by decoding payload/meta and comparing to direct per-tile quantization.
    """
    if groupsize != N_TILE:
        raise ValueError(f"Validation currently expects groupsize={N_TILE}, got {groupsize}")

    device = rank_major_ref.device

    slab_meta = slab_meta_buffer.to(device=device, dtype=torch.int64)
    tile_meta = tile_meta_buffer.to(device=device, dtype=torch.int64)
    rank_idx = rank_idx_buffer.to(device=device, dtype=torch.int64)
    payload = payload_buffer.to(device=device, dtype=torch.uint32)
    scale = scale_buffer.to(device=device, dtype=torch.float16)

    num_slabs = int(slab_meta.shape[0])
    num_tiles_total = int(tile_meta.shape[0])

    for slab_i in range(num_slabs):
        bits = int(slab_meta[slab_i, 0].item())
        k_slab = int(slab_meta[slab_i, 1].item())
        rank_off = int(slab_meta[slab_i, 2].item())
        tile_off = int(slab_meta[slab_i, 3].item())

        if bits not in _SUPPORTED_BITS:
            raise AssertionError(f"Invalid bits in slab_meta: {bits}")

        next_tile_off = (
            int(slab_meta[slab_i + 1, 3].item()) if slab_i + 1 < num_slabs else num_tiles_total
        )
        slab_rank_idx = rank_idx[rank_off:rank_off + k_slab]
        slab_ref = rank_major_ref.index_select(0, slab_rank_idx)

        words_row = _words_per_row(bits)

        for t in range(tile_off, next_tile_off):
            pack_off = int(tile_meta[t, 0].item())
            scale_off = int(tile_meta[t, 1].item())
            n_valid = int(tile_meta[t, 2].item())
            tile_local_id = t - tile_off
            c0 = tile_local_id * N_TILE

            words_tile = k_slab * MICROS_PER_TILE * words_row
            payload_tile = payload[pack_off:pack_off + words_tile]

            if bits == 16:
                decoded_tile = _unpack_fp16_tile_payload(payload_tile, k_slab, device=device)
                ref_tile = slab_ref[:, c0:c0 + n_valid].to(torch.float16)
                if not torch.allclose(decoded_tile[:, :n_valid], ref_tile, atol=atol, rtol=rtol):
                    max_abs = (decoded_tile[:, :n_valid] - ref_tile).abs().max().item()
                    raise AssertionError(f"fp16 payload validation failed (slab={slab_i}, tile={tile_local_id}, max_abs={max_abs})")
                continue

            decoded_q = _unpack_lowbit_tile_payload(payload_tile, bits, k_slab, device=device)
            decoded_q = decoded_q[:, :n_valid]

            # Match the packer's quantize dtype (fp32) so validation does not
            # drift on fp16 rank_major inputs.
            ref_tile = slab_ref[:, c0:c0 + n_valid].to(torch.float32)
            q_ref, _ = quantize_symmetric(ref_tile, bits=bits, dim=1)
            if not torch.equal(decoded_q, q_ref):
                max_abs = (decoded_q.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item()
                raise AssertionError(
                    f"low-bit payload validation failed (slab={slab_i}, tile={tile_local_id}, bits={bits}, max_abs={max_abs})"
                )

            # Also verify scale buffer corresponds to per-tile quantization scale.
            _, s_ref = quantize_symmetric(ref_tile, bits=bits, dim=1)
            s_buf = scale[scale_off:scale_off + k_slab].to(torch.float32)
            if not torch.allclose(s_buf, s_ref.to(torch.float32), atol=1e-3, rtol=1e-3):
                max_abs = (s_buf - s_ref.to(torch.float32)).abs().max().item()
                raise AssertionError(
                    f"scale validation failed (slab={slab_i}, tile={tile_local_id}, bits={bits}, max_abs={max_abs})"
                )


def _get_rank_major_tensor(
    moe_dict,
    layer_idx: int,
    expert_idx: int,
    mtype: str,
    device: torch.device,
) -> Tuple[Tensor, Tensor, str]:
    """
    Return rank-major matrix and S.

    gate/up input format:
        moe_dict[layer][expert][mtype] = (U_rank_dim, S), U_rank_dim shape [rank, dim]

    down input format:
        moe_dict[layer][expert][mtype] = (Vh_dim_rank, S), Vh_dim_rank shape [dim, rank]
        Converted to rank-major [rank, dim] by transpose.

    Returns:
        rank_major: [rank, dim]
        S: [rank]
        packed_field: "u" for gate/up, "vh" for down
    """
    payload = moe_dict[layer_idx][expert_idx][mtype]
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"Expected tuple(matrix, S) for {mtype}, got type={type(payload)}")

    matrix, s_vec = payload
    if s_vec.dim() != 1:
        raise ValueError(f"S must be 1-D, got shape={tuple(s_vec.shape)}")

    rank = int(s_vec.numel())

    if mtype in {"gate_proj", "up_proj"}:
        if matrix.dim() != 2:
            raise ValueError(f"U_rank_dim must be 2-D, got shape={tuple(matrix.shape)}")
        if int(matrix.shape[0]) != rank:
            raise ValueError(
                f"U_rank_dim shape mismatch: rank={rank}, tensor shape={tuple(matrix.shape)}"
            )
        return matrix.to(device, non_blocking=True).contiguous(), s_vec.to(device, non_blocking=True).contiguous(), "u"

    if mtype == "down_proj":
        if matrix.dim() != 2:
            raise ValueError(f"Vh_dim_rank must be 2-D, got shape={tuple(matrix.shape)}")
        if int(matrix.shape[1]) != rank:
            raise ValueError(
                f"Vh_dim_rank shape mismatch: rank={rank}, tensor shape={tuple(matrix.shape)}"
            )
        rank_major = matrix.to(device, non_blocking=True).transpose(0, 1).contiguous()
        return rank_major, s_vec.to(device, non_blocking=True).contiguous(), "vh"

    raise ValueError(f"Unsupported mtype: {mtype}")


def pack_single_weight(
    moe_dict,
    segments: List[Tuple[int, Tuple[int, int]]],
    state_dict: Dict[str, Tensor],
    layer_idx: int,
    expert_idx: int,
    mtype: str,
    groupsize: int = N_TILE,
    validate_layout: bool = False,
):
    """
    Quantize + pack one expert matrix into runtime-friendly flat buffers.

    Global buffers generated:
      - payload_buffer (uint32, 1-D): [slab][tile][stage][micro][row][word]
      - rank_idx_buffer (int32, 1-D): local rank -> logical rank mapping by slab
      - tile_meta_buffer (int32, 2-D): [pack_offset, scale_offset, n_valid]
      - slab_meta_buffer (int32, 2-D): [bit, k_slab, rank_idx_offset, tile_meta_offset]
      - scale_buffer (fp16, 1-D): [tile][local_rank][group_in_tile], currently group_in_tile=1

    Constraints:
      - slab bit homogeneous
      - fixed rank-major view
      - fixed formulas for stage/micro traversal (no stage/micro metadata)
    """
    if groupsize != N_TILE:
        raise ValueError(
            f"Current runtime layout requires groupsize == N_TILE ({N_TILE}), got {groupsize}"
        )

    device = torch.device("cuda")
    rank_major, s_vec, packed_field = _get_rank_major_tensor(
        moe_dict=moe_dict,
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        mtype=mtype,
        device=device,
    )

    rank, dim = rank_major.shape
    if rank != int(s_vec.numel()):
        raise ValueError(f"rank mismatch: matrix rank={rank}, S={int(s_vec.numel())}")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if dim % N_MICRO != 0:
        raise ValueError(f"dim={dim} must be divisible by microtile width {N_MICRO}")

    prefix = f"quantized.layer{layer_idx}.expert{expert_idx}.{mtype}"

    # Build deterministic bit buckets in logical rank order.
    groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    seen = torch.zeros(rank, dtype=torch.bool)

    for bits, (s, e) in segments:
        bits = int(bits)
        s = int(s)
        e = int(e)

        if bits not in _SUPPORTED_BITS and bits != 0:
            raise ValueError(f"Unsupported bits={bits}")
        if not (0 <= s <= e <= rank):
            raise ValueError(f"Invalid segment range (s={s}, e={e}) for rank={rank}")
        if e == s:
            continue
        if seen[s:e].any().item():
            raise ValueError(f"Overlapping segments in range [{s}, {e})")

        seen[s:e] = True
        groups[bits].append((s, e))

    bit_order = sorted(groups.keys(), reverse=True)
    for b in bit_order:
        groups[b].sort(key=lambda x: x[0])

    # Keep lightweight segment metadata (bit buckets in reordered rank space).
    seg_meta: List[List[int]] = []
    rank_cursor = 0

    # Runtime global buffers.
    payload_chunks: List[Tensor] = []
    rank_idx_chunks: List[Tensor] = []
    scale_chunks: List[Tensor] = []
    s_chunks: List[Tensor] = []

    slab_meta_rows: List[List[int]] = []
    tile_meta_rows: List[List[int]] = []

    payload_offset = 0
    rank_idx_offset = 0
    scale_offset = 0

    n_tiles = (dim + N_TILE - 1) // N_TILE

    # Accurate wall-clock timing for asynchronous CUDA work.
    torch.cuda.synchronize(device)
    t_pack_total = time.perf_counter()
    quantize_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

    for bits in bit_order:
        if bits == 0:
            continue

        bit_ranges = groups[bits]
        bit_idx_parts: List[Tensor] = [
            torch.arange(s, e, dtype=torch.int32) for s, e in bit_ranges if e > s
        ]
        if not bit_idx_parts:
            continue
        bit_indices_cpu = torch.cat(bit_idx_parts, dim=0)
        k_total = int(bit_indices_cpu.numel())
        if k_total == 0:
            continue

        seg_meta.append([int(bits), int(rank_cursor), int(rank_cursor + k_total)])
        rank_cursor += k_total
        bit_indices_gpu = bit_indices_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        bit_rank_all = rank_major.index_select(0, bit_indices_gpu).contiguous()
        bit_s_all = s_vec.index_select(0, bit_indices_gpu).contiguous()

        for slab_start in range(0, k_total, K_SLAB):
            slab_end = min(k_total, slab_start + K_SLAB)
            k_slab = slab_end - slab_start
            slab_rank = bit_rank_all[slab_start:slab_end]           # [k_slab, dim]
            slab_s = bit_s_all[slab_start:slab_end]                 # [k_slab]

            tile_meta_offset = len(tile_meta_rows)
            slab_meta_rows.append([int(bits), int(k_slab), int(rank_idx_offset), int(tile_meta_offset)])

            rank_idx_cpu = bit_indices_cpu[slab_start:slab_end].contiguous()
            rank_idx_chunks.append(rank_idx_cpu)
            rank_idx_offset += k_slab

            s_chunks.append(slab_s.to(torch.float16))

            if bits == 16:
                payload_tiles, scale_flat = _pack_fp16_slab_payload_and_scale(slab_rank)
                words_per_row = 16
            else:
                payload_tiles, scale_flat = _pack_lowbit_slab_payload_and_scale(
                    slab_rank, bits, quantize_events=quantize_events
                )
                words_per_row = bits

            words_per_tile = k_slab * MICROS_PER_TILE * words_per_row
            if int(payload_tiles.shape[1]) != words_per_tile:
                raise RuntimeError(
                    f"payload width mismatch: expected {words_per_tile}, got {int(payload_tiles.shape[1])}"
                )
            if int(payload_tiles.shape[0]) != n_tiles:
                raise RuntimeError(
                    f"tile count mismatch: expected {n_tiles}, got {int(payload_tiles.shape[0])}"
                )

            # Metadata is still tile-centric and unchanged.
            for tile_id in range(n_tiles):
                c0 = tile_id * N_TILE
                n_valid = min(N_TILE, dim - c0)
                tile_meta_rows.append(
                    [
                        int(payload_offset + tile_id * words_per_tile),
                        int(scale_offset + tile_id * k_slab),
                        int(n_valid),
                    ]
                )

            payload_gpu = payload_tiles.reshape(-1).contiguous().to(torch.uint32)
            scale_gpu = scale_flat.reshape(-1).contiguous().to(torch.float16)

            payload_chunks.append(payload_gpu)
            scale_chunks.append(scale_gpu)
            payload_offset += int(payload_gpu.numel())
            scale_offset += int(scale_gpu.numel())

    payload_buffer = (
        torch.cat(payload_chunks, dim=0).cpu()
        if payload_chunks
        else torch.empty((0,), dtype=torch.uint32)
    )
    rank_idx_buffer = torch.cat(rank_idx_chunks, dim=0) if rank_idx_chunks else torch.empty((0,), dtype=torch.int32)
    scale_buffer = (
        torch.cat(scale_chunks, dim=0).cpu()
        if scale_chunks
        else torch.empty((0,), dtype=torch.float16)
    )
    s_buffer = (
        torch.cat(s_chunks, dim=0).cpu()
        if s_chunks
        else torch.empty((0,), dtype=torch.float16)
    )

    slab_meta_buffer = _to_compact_int_tensor(slab_meta_rows)
    tile_meta_buffer = _to_compact_int_tensor(tile_meta_rows)

    # Runtime-facing buffers.
    state_dict[f"{prefix}.{packed_field}.payload_buffer"] = payload_buffer
    state_dict[f"{prefix}.{packed_field}.rank_idx_buffer"] = rank_idx_buffer
    state_dict[f"{prefix}.{packed_field}.tile_meta_buffer"] = tile_meta_buffer
    state_dict[f"{prefix}.{packed_field}.slab_meta_buffer"] = slab_meta_buffer
    state_dict[f"{prefix}.{packed_field}.scale_buffer"] = scale_buffer

    # S aligned with rank_idx_buffer order (same local-rank sequence as slabs).
    state_dict[f"{prefix}.s_buffer"] = s_buffer

    # Keep high-level mapping metadata.
    state_dict[f"{prefix}.segments"] = torch.tensor(seg_meta, dtype=torch.int32)
    state_dict[f"{prefix}.original_indices"] = rank_idx_buffer.clone()
    state_dict[f"{prefix}.original_rank"] = torch.tensor(rank, dtype=torch.int32)
    state_dict[f"{prefix}.groupsize"] = torch.tensor(groupsize, dtype=torch.int16)

    torch.cuda.synchronize(device)
    total_elapsed = time.perf_counter() - t_pack_total
    quantize_total = 0.0
    for q_start, q_end in quantize_events:
        quantize_total += float(q_start.elapsed_time(q_end)) / 1000.0
    pack_elapsed = max(0.0, total_elapsed - quantize_total)

    # Optional global validation hook.
    if validate_layout and payload_buffer.numel() > 0:
        validate_runtime_layout(
            rank_major_ref=rank_major,
            slab_meta_buffer=slab_meta_buffer,
            tile_meta_buffer=tile_meta_buffer,
            rank_idx_buffer=rank_idx_buffer,
            payload_buffer=payload_buffer,
            scale_buffer=scale_buffer,
            groupsize=groupsize,
        )

    return {
        "total_s": total_elapsed,
        "quantize_s": quantize_total,
        "pack_s": pack_elapsed,
    }


def save_sharded_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    save_dir: str,
    compressed_ratio: float
):
    """
    Save quantized state_dict without sharding.
    Always produces a single-file HF-style checkpoint: pytorch_model.bin
    """
    logger = setup_logger(__name__)
    os.makedirs(save_dir, exist_ok=True)

    filename = "pytorch_model.bin"
    output_path = os.path.join(save_dir, filename)

    torch.save(state_dict, output_path)

    logger.info(f"Saved ratio={compressed_ratio} quantized checkpoint to: {output_path}")

    return output_path
