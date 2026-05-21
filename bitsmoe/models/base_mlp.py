from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


_BITSMOE_LOAD_PROGRESS = {
    "enabled": False,
    "total": 0,
    "current": 0,
    "bar": None,
    "external_updates": False,
}


def configure_bitsmoe_load_progress(
    total_experts: int,
    enabled: bool = True,
    external_updates: bool = False,
) -> None:
    finalize_bitsmoe_load_progress()
    total = int(total_experts)
    if not enabled or total <= 0:
        return

    _BITSMOE_LOAD_PROGRESS["enabled"] = True
    _BITSMOE_LOAD_PROGRESS["total"] = total
    _BITSMOE_LOAD_PROGRESS["current"] = 0
    _BITSMOE_LOAD_PROGRESS["external_updates"] = bool(external_updates)
    if _tqdm is not None:
        _BITSMOE_LOAD_PROGRESS["bar"] = _tqdm(
            total=total,
            desc="BitsMoE packed expert load",
            dynamic_ncols=True,
        )


def advance_bitsmoe_load_progress(step: int = 1, source: str = "internal") -> None:
    if not _BITSMOE_LOAD_PROGRESS["enabled"]:
        return
    if source == "internal" and _BITSMOE_LOAD_PROGRESS["external_updates"]:
        return

    s = int(step)
    if s <= 0:
        return

    total = int(_BITSMOE_LOAD_PROGRESS["total"])
    current = min(total, int(_BITSMOE_LOAD_PROGRESS["current"]) + s)
    _BITSMOE_LOAD_PROGRESS["current"] = current

    bar = _BITSMOE_LOAD_PROGRESS["bar"]
    if bar is not None:
        bar.update(current - bar.n)
    else:
        # Fallback without tqdm: sparse periodic progress output.
        interval = max(1, total // 20)
        if current == total or current % interval == 0:
            print(f"BitsMoE packed expert load: {current}/{total}", flush=True)


def finalize_bitsmoe_load_progress() -> None:
    bar = _BITSMOE_LOAD_PROGRESS["bar"]
    if bar is not None:
        total = int(_BITSMOE_LOAD_PROGRESS["total"])
        current = int(_BITSMOE_LOAD_PROGRESS["current"])
        if current < total:
            bar.update(total - bar.n)
        bar.close()

    _BITSMOE_LOAD_PROGRESS["enabled"] = False
    _BITSMOE_LOAD_PROGRESS["total"] = 0
    _BITSMOE_LOAD_PROGRESS["current"] = 0
    _BITSMOE_LOAD_PROGRESS["bar"] = None
    _BITSMOE_LOAD_PROGRESS["external_updates"] = False


class BitsMoE_BaseMoeMLP(nn.Module):
    """
    Runtime container for packed MoE expert weights.

    Forward compute is intentionally not implemented yet.
    """

    _MTYPE_TO_TAG = {
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
    }
    _RUNTIME_BUFFER_DTYPES = {
        "payload_buffer": torch.uint32,
        "rank_idx_buffer": torch.int32,
        "tile_meta_buffer": torch.int32,
        "slab_meta_buffer": torch.int32,
        "scale_buffer": torch.float16,
        "s_buffer": torch.float16,
        "segments": torch.int32,
        "original_indices": torch.int32,
        "groupsize": torch.int16,
        "original_rank": torch.int32,
    }

    @classmethod
    def _runtime_buffer_leaf_keys(cls):
        return tuple(
            f"{tag}_{name}"
            for tag in cls._MTYPE_TO_TAG.values()
            for name in cls._RUNTIME_BUFFER_DTYPES.keys()
        )

    def __init__(
        self,
        config,
        intermediate_size: Optional[int] = None,
        packed_state: Optional[Dict[str, torch.Tensor]] = None,
        packed_prefix: Optional[str] = None,
        packed_device: torch.device = torch.device("cpu"),
        defer_runtime_buffer_init: bool = False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self.is_bitsmoe_packed = False
        self.skip_expert = True
        self.runtime_meta: Dict[str, Dict[str, Any]] = {}
        self._load_progress_marked = False
        self._defer_runtime_buffer_init = bool(defer_runtime_buffer_init)
        if not self._defer_runtime_buffer_init:
            self._init_runtime_buffers()

        if packed_state is not None:
            if packed_prefix is None:
                raise ValueError("packed_prefix must be provided when packed_state is provided")
            self.load_runtime_packed_from_state(
                state_dict=packed_state,
                prefix=packed_prefix,
                device=packed_device,
            )

    def _init_runtime_buffers(self) -> None:
        for tag in self._MTYPE_TO_TAG.values():
            for name, dtype in self._RUNTIME_BUFFER_DTYPES.items():
                init_tensor = (
                    torch.tensor(0, dtype=dtype)
                    if name in {"groupsize", "original_rank"}
                    else torch.empty(0, dtype=dtype)
                )
                self.register_buffer(
                    f"{tag}_{name}",
                    init_tensor,
                    persistent=True,
                )

    @staticmethod
    def _to_tensor_on(t: torch.Tensor, device: torch.device) -> torch.Tensor:
        return t.detach().to(device=device, non_blocking=True).contiguous()

    @staticmethod
    def _scalar_tensor_to_int(t: torch.Tensor, name: str) -> int:
        if t.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(t.shape)}")
        return int(t.detach().to(device="cpu", dtype=torch.int64).reshape(()))

    def _register_or_replace_buffer(self, name: str, tensor: torch.Tensor) -> None:
        # Fast-path: when the buffer already exists, update the internal buffer
        # slot directly to avoid costly del/register cycles for hundreds of
        # thousands of tensors during large-MoE checkpoint loading.
        if name in self._buffers:
            self._buffers[name] = tensor
            return
        self.register_buffer(name, tensor, persistent=True)

    def _load_one_mtype(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        mtype: str,
        device: torch.device,
    ) -> bool:
        tag = self._MTYPE_TO_TAG[mtype]
        packed_field = "u" if mtype in {"gate_proj", "up_proj"} else "vh"
        base = f"{prefix}.{mtype}.{packed_field}"

        required = [
            f"{base}.payload_buffer",
            f"{base}.rank_idx_buffer",
            f"{base}.tile_meta_buffer",
            f"{base}.slab_meta_buffer",
            f"{base}.scale_buffer",
            f"{prefix}.{mtype}.s_buffer",
            f"{prefix}.{mtype}.segments",
            f"{prefix}.{mtype}.original_indices",
            f"{prefix}.{mtype}.original_rank",
            f"{prefix}.{mtype}.groupsize",
        ]
        if all(k in state_dict for k in required):
            payload = self._to_tensor_on(state_dict[f"{base}.payload_buffer"], device).to(torch.uint32)
            rank_idx = self._to_tensor_on(state_dict[f"{base}.rank_idx_buffer"], device).to(torch.int32)
            tile_meta = self._to_tensor_on(state_dict[f"{base}.tile_meta_buffer"], device).to(torch.int32)
            slab_meta = self._to_tensor_on(state_dict[f"{base}.slab_meta_buffer"], device).to(torch.int32)
            scale = self._to_tensor_on(state_dict[f"{base}.scale_buffer"], device).to(torch.float16)
            s_buf = self._to_tensor_on(state_dict[f"{prefix}.{mtype}.s_buffer"], device).to(torch.float16)
            segments = self._to_tensor_on(state_dict[f"{prefix}.{mtype}.segments"], device)
            original_indices = self._to_tensor_on(
                state_dict[f"{prefix}.{mtype}.original_indices"], device
            )
            original_rank = self._scalar_tensor_to_int(
                state_dict[f"{prefix}.{mtype}.original_rank"],
                f"{prefix}.{mtype}.original_rank",
            )
            groupsize = self._scalar_tensor_to_int(
                state_dict[f"{prefix}.{mtype}.groupsize"],
                f"{prefix}.{mtype}.groupsize",
            )
        else:
            # Fallback path for loading already-materialized module keys from model.state_dict().
            mat_required = [
                f"{prefix}.{tag}_payload_buffer",
                f"{prefix}.{tag}_rank_idx_buffer",
                f"{prefix}.{tag}_tile_meta_buffer",
                f"{prefix}.{tag}_slab_meta_buffer",
                f"{prefix}.{tag}_scale_buffer",
                f"{prefix}.{tag}_s_buffer",
                f"{prefix}.{tag}_segments",
                f"{prefix}.{tag}_original_indices",
            ]
            if not all(k in state_dict for k in mat_required):
                return False

            payload = self._to_tensor_on(state_dict[f"{prefix}.{tag}_payload_buffer"], device).to(torch.uint32)
            rank_idx = self._to_tensor_on(state_dict[f"{prefix}.{tag}_rank_idx_buffer"], device).to(torch.int32)
            tile_meta = self._to_tensor_on(state_dict[f"{prefix}.{tag}_tile_meta_buffer"], device).to(torch.int32)
            slab_meta = self._to_tensor_on(state_dict[f"{prefix}.{tag}_slab_meta_buffer"], device).to(torch.int32)
            scale = self._to_tensor_on(state_dict[f"{prefix}.{tag}_scale_buffer"], device).to(torch.float16)
            s_buf = self._to_tensor_on(state_dict[f"{prefix}.{tag}_s_buffer"], device).to(torch.float16)
            segments = self._to_tensor_on(state_dict[f"{prefix}.{tag}_segments"], device)
            original_indices = self._to_tensor_on(
                state_dict[f"{prefix}.{tag}_original_indices"], device
            )

            rk_key = f"{prefix}.{tag}_original_rank"
            gs_key = f"{prefix}.{tag}_groupsize"
            original_rank = int(
                self._scalar_tensor_to_int(state_dict[rk_key], rk_key)
                if rk_key in state_dict
                else original_indices.numel()
            )
            groupsize = int(
                self._scalar_tensor_to_int(state_dict[gs_key], gs_key)
                if gs_key in state_dict
                else 128
            )

        self._register_or_replace_buffer(f"{tag}_payload_buffer", payload)
        self._register_or_replace_buffer(f"{tag}_rank_idx_buffer", rank_idx)
        self._register_or_replace_buffer(f"{tag}_tile_meta_buffer", tile_meta)
        self._register_or_replace_buffer(f"{tag}_slab_meta_buffer", slab_meta)
        self._register_or_replace_buffer(f"{tag}_scale_buffer", scale)
        self._register_or_replace_buffer(f"{tag}_s_buffer", s_buf)
        self._register_or_replace_buffer(f"{tag}_segments", segments)
        self._register_or_replace_buffer(f"{tag}_original_indices", original_indices)
        self._register_or_replace_buffer(
            f"{tag}_groupsize",
            torch.tensor(groupsize, dtype=torch.int16, device=device),
        )
        self._register_or_replace_buffer(
            f"{tag}_original_rank",
            torch.tensor(original_rank, dtype=torch.int32, device=device),
        )

        # Keep scalar metadata as Python ints to avoid .item() in runtime path.
        slab_rows: List[Tuple[int, int, int, int]] = [
            tuple(int(v) for v in row)
            for row in slab_meta.to(dtype=torch.int64, device="cpu").tolist()
        ]
        tile_rows: List[Tuple[int, int, int]] = [
            tuple(int(v) for v in row)
            for row in tile_meta.to(dtype=torch.int64, device="cpu").tolist()
        ]
        seg_rows: List[Tuple[int, int, int]] = [
            tuple(int(v) for v in row)
            for row in segments.to(dtype=torch.int64, device="cpu").tolist()
        ]
        self.runtime_meta[mtype] = {
            "groupsize": groupsize,
            "original_rank": original_rank,
            "slab_meta_rows": slab_rows,
            "tile_meta_rows": tile_rows,
            "segments": seg_rows,
            "packed_field": packed_field,
        }
        return True

    def load_runtime_packed_from_state(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        device: torch.device = torch.device("cpu"),
    ) -> bool:
        ok = True
        for mtype in ("gate_proj", "up_proj", "down_proj"):
            ok = self._load_one_mtype(state_dict, prefix, mtype, device=device) and ok

        self.is_bitsmoe_packed = ok
        self.skip_expert = not ok
        return ok

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # HF with assign=True loads one tensor at a time; pre-shape runtime buffers
        # to the incoming tensor to avoid size mismatch on initially empty buffers.
        # For very large MoE checkpoints, this function is on the hot path.
        assign_to_params_buffers = bool(local_metadata.get("assign_to_params_buffers", False))
        has_local_tensor = False
        consumed_keys = []

        for local_key in self._runtime_buffer_leaf_keys():
            full_key = f"{prefix}{local_key}"
            tensor = state_dict.get(full_key, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            has_local_tensor = True
            consumed_keys.append(full_key)

            current = getattr(self, local_key, None)
            if (
                isinstance(current, torch.Tensor)
                and current.shape == tensor.shape
                and current.dtype == tensor.dtype
            ):
                continue

            # assign=True path can bind checkpoint tensor directly and avoid
            # allocating huge empty placeholders for every packed buffer.
            replacement = tensor.detach() if assign_to_params_buffers else torch.empty_like(tensor)
            self._register_or_replace_buffer(local_key, replacement)

        if has_local_tensor and not self._load_progress_marked:
            self._load_progress_marked = True
            advance_bitsmoe_load_progress(1, source="internal")

        # We have already consumed and bound all local runtime keys explicitly.
        # Removing consumed entries avoids extra work in parent recursive loaders.
        for k in consumed_keys:
            state_dict.pop(k, None)

        # Keep compatibility for paths where this module may eventually include
        # additional parameters/buffers.
        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        self.is_bitsmoe_packed = all(
            hasattr(self, f"{tag}_payload_buffer")
            and isinstance(getattr(self, f"{tag}_payload_buffer"), torch.Tensor)
            and getattr(self, f"{tag}_payload_buffer").numel() > 0
            for tag in self._MTYPE_TO_TAG.values()
        )
        self.skip_expert = not self.is_bitsmoe_packed

    # Backward-compatible entrypoint used by old loader paths.
    def load_quantized_weight_from_state(
        self,
        state_dict: Dict[str, torch.Tensor],
        orig_prefix: str,
    ) -> bool:
        return self.load_runtime_packed_from_state(state_dict=state_dict, prefix=orig_prefix)

    def forward(
        self,
        x: torch.Tensor,
        h_gate_proj: Optional[torch.Tensor] = None,
        h_up_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "BitsMoE_BaseMoeMLP forward is intentionally not implemented in this stage. "
            "This module currently only stores packed payload/metadata."
        )
