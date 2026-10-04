from typing import Dict, Optional

import torch
import torch.nn as nn

from bitsmoe.models.shared_basis_marlin import marlin_shared_basis_linear


class BitsMoE_BaseSparseMoeBlock(nn.Module):
    _SHARED_BASIS_BUFFER_KEYS = tuple(
        f"{name}_{suffix}"
        for name in ("shared_vh_gate_proj", "shared_vh_up_proj", "shared_u_down")
        for suffix in ("qweight", "scales")
    )

    def _init_bitsmoe_common_state(self) -> None:
        for name in ("shared_vh_gate_proj", "shared_vh_up_proj", "shared_u_down"):
            self.register_buffer(f"{name}_qweight", torch.empty(0, dtype=torch.int32))
            self.register_buffer(f"{name}_scales", torch.empty(0, dtype=torch.float16))
        self._shared_marlin_workspaces: Dict[str, torch.Tensor] = {}
        self._shared_marlin_scales_cache: Dict[str, torch.Tensor] = {}
        self._invalidate_runtime_cache()

    def _shared_basis_rank(self, name: str) -> int:
        scales = getattr(self, f"{name}_scales")
        return int(scales.shape[0] * 128 if name == "shared_u_down" else scales.shape[1])

    def _shared_basis_linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        weight = getattr(self, f"{name}_qweight")
        scales = getattr(self, f"{name}_scales")
        if weight.numel() == 0 or scales.numel() == 0:
            raise RuntimeError(f"Layer {self.layer_idx} missing {name} Marlin weights.")
        x = x.to(torch.float16)
        if scales.dtype != torch.float16:
            cached = self._shared_marlin_scales_cache.get(name)
            if cached is None or cached.device != scales.device:
                cached = scales.to(torch.float16)
                self._shared_marlin_scales_cache[name] = cached
            scales = cached
        if x.shape[0] >= max(128, scales.numel() // 256):
            from bitsmoe.algorithms.triton_shared import prefill_shared_linear

            return prefill_shared_linear(x, weight, scales)
        workspace = self._ensure_shared_marlin_workspace(name, x.device)
        return marlin_shared_basis_linear(x, weight, scales, workspace)

    def _ensure_shared_marlin_workspace(self, name: str, device: torch.device) -> torch.Tensor:
        workspace = self._shared_marlin_workspaces.get(name)
        if workspace is None or workspace.device != device:
            from bitsmoe.algorithms.vllm_compat import marlin_make_workspace_new

            workspace = marlin_make_workspace_new(device)
            self._shared_marlin_workspaces[name] = workspace
        return workspace

    def set_expert(self, expert_idx: int, expert_module: Optional[nn.Module]) -> None:
        self.experts[int(expert_idx)] = expert_module
        self._invalidate_runtime_cache()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._shared_marlin_workspaces.clear()
        self._shared_marlin_scales_cache.clear()
        self._invalidate_runtime_cache()
        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        assign = bool(local_metadata.get("assign_to_params_buffers", False))
        for name in self._SHARED_BASIS_BUFFER_KEYS:
            tensor = state_dict.get(f"{prefix}{name}")
            if not isinstance(tensor, torch.Tensor):
                continue
            current = getattr(self, name)
            if current.shape != tensor.shape or current.dtype != tensor.dtype:
                setattr(self, name, tensor.detach() if assign else torch.empty_like(tensor))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )
        self._shared_marlin_workspaces.clear()
        self._shared_marlin_scales_cache.clear()
        self._invalidate_runtime_cache()

    def _invalidate_runtime_cache(self) -> None:
        self._runtime_cache_ready = False
        self._runtime_cache_device = None
        self._packed_kernel = None

    def _ensure_runtime_cache(self, device: torch.device) -> None:
        if self._runtime_cache_ready and self._runtime_cache_device == device:
            return
        if device.type != "cuda":
            raise RuntimeError("Packed expert inference requires CUDA.")
        if str(getattr(self.config, "hidden_act", "silu")).lower() != "silu":
            raise ValueError("Packed experts require SiLU activation.")
        experts = []
        ids = []
        for index, expert in enumerate(self.experts):
            if expert is None or getattr(expert, "skip_expert", False):
                continue
            if not getattr(expert, "is_bitsmoe_packed", False):
                raise RuntimeError(f"Layer {self.layer_idx} expert {index} is not packed.")
            ids.append(index)
            experts.append(expert)
        self._packed_kernel = None
        if experts:
            from bitsmoe.algorithms.triton_moe import TritonPackedMoE

            intermediate = int(experts[0].intermediate_size)
            if any(int(expert.intermediate_size) != intermediate for expert in experts):
                raise ValueError("Experts in one MoE layer must have the same intermediate size.")
            buffers = [
                [getattr(expert, f"{projection}_{suffix}_buffer") for expert in experts]
                for projection in ("gate", "up", "down")
                for suffix in ("payload", "rank_idx", "tile_meta", "slab_meta", "scale", "s")
            ]
            if any(tensor.device != device for group in buffers for tensor in group):
                raise RuntimeError("Packed experts must share the activation's CUDA device.")
            expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32, device=device)
            expert_map[torch.tensor(ids, device=device)] = torch.arange(
                len(ids), dtype=torch.int32, device=device,
            )
            ranks = [self._shared_basis_rank(name) for name in (
                "shared_vh_gate_proj", "shared_vh_up_proj", "shared_u_down",
            )]
            self._packed_kernel = TritonPackedMoE(buffers, expert_map, *ranks, intermediate)
        self._runtime_cache_device = device
        self._runtime_cache_ready = True

    def _routed_forward(self, hidden_states, selected_experts, routing_weights):
        with torch.cuda.device(hidden_states.device):
            self._ensure_runtime_cache(hidden_states.device)
            if self._packed_kernel is None:
                return torch.zeros_like(hidden_states)
            gate = self._shared_basis_linear(hidden_states, "shared_vh_gate_proj")
            up = self._shared_basis_linear(hidden_states, "shared_vh_up_proj")
            accum = self._packed_kernel.forward(gate, up, selected_experts, routing_weights.float())
            return self._shared_basis_linear(accum, "shared_u_down").to(hidden_states.dtype)
