from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from bitsmoe.algorithms import _HAS_MLP_FORWARD_CUDA, mlp_forward_cuda
from bitsmoe.models.shared_basis_marlin import marlin_shared_basis_linear


class BitsMoE_BaseSparseMoeBlock(nn.Module):
    _SHARED_BASIS_BUFFER_KEYS = (
        "shared_vh_gate_proj_qweight",
        "shared_vh_gate_proj_scales",
        "shared_vh_up_proj_qweight",
        "shared_vh_up_proj_scales",
        "shared_u_down_qweight",
        "shared_u_down_scales",
    )

    def _init_bitsmoe_common_state(self) -> None:
        for name in ("shared_vh_gate_proj", "shared_vh_up_proj", "shared_u_down"):
            self.register_buffer(f"{name}_qweight", torch.empty(0, dtype=torch.int32), persistent=True)
            self.register_buffer(f"{name}_scales", torch.empty(0, dtype=torch.float16), persistent=True)
        self._shared_marlin_workspaces: Dict[str, torch.Tensor] = {}
        self._shared_marlin_scales_cache: Dict[str, torch.Tensor] = {}

        # Runtime caches for packed path (built lazily on target device).
        self._runtime_cache_ready = False
        self._runtime_cache_device: Optional[torch.device] = None
        self._cached_token_count = -1
        self._cached_flat_token_ids: Optional[torch.Tensor] = None

        self._packed_expert_count = 0
        self._packed_intermediate_size = 0
        self._packed_global_to_local: Optional[torch.Tensor] = None

        self._gate_payload_static: Sequence[torch.Tensor] = ()
        self._gate_rank_idx_static: Sequence[torch.Tensor] = ()
        self._gate_tile_meta_static: Sequence[torch.Tensor] = ()
        self._gate_slab_meta_static: Sequence[torch.Tensor] = ()
        self._gate_scale_static: Sequence[torch.Tensor] = ()
        self._gate_s_static: Sequence[torch.Tensor] = ()

        self._up_payload_static: Sequence[torch.Tensor] = ()
        self._up_rank_idx_static: Sequence[torch.Tensor] = ()
        self._up_tile_meta_static: Sequence[torch.Tensor] = ()
        self._up_slab_meta_static: Sequence[torch.Tensor] = ()
        self._up_scale_static: Sequence[torch.Tensor] = ()
        self._up_s_static: Sequence[torch.Tensor] = ()

        self._down_payload_static: Sequence[torch.Tensor] = ()
        self._down_rank_idx_static: Sequence[torch.Tensor] = ()
        self._down_tile_meta_static: Sequence[torch.Tensor] = ()
        self._down_slab_meta_static: Sequence[torch.Tensor] = ()
        self._down_scale_static: Sequence[torch.Tensor] = ()
        self._down_s_static: Sequence[torch.Tensor] = ()

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
        workspace = self._ensure_shared_marlin_workspace(name, x.device)
        return marlin_shared_basis_linear(x, weight, scales, workspace)

    def _ensure_shared_marlin_workspace(self, name: str, device: torch.device) -> torch.Tensor:
        workspace = self._shared_marlin_workspaces.get(name)
        if workspace is None or workspace.device != device:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

            workspace = marlin_make_workspace_new(device)
            self._shared_marlin_workspaces[name] = workspace
        return workspace

    def set_expert(self, expert_idx: int, expert_module: Optional[nn.Module]) -> None:
        idx = int(expert_idx)
        self.experts[idx] = expert_module
        self._invalidate_runtime_cache()

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
        assign_to_params_buffers = bool(local_metadata.get("assign_to_params_buffers", False))
        for local_key in self._SHARED_BASIS_BUFFER_KEYS:
            tensor = state_dict.get(f"{prefix}{local_key}")
            if not isinstance(tensor, torch.Tensor):
                continue
            current = getattr(self, local_key)
            if current.shape == tensor.shape and current.dtype == tensor.dtype:
                continue
            replacement = tensor.detach() if assign_to_params_buffers else torch.empty_like(tensor)
            setattr(self, local_key, replacement)

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        self._shared_marlin_workspaces.clear()
        self._shared_marlin_scales_cache.clear()
        self._invalidate_runtime_cache()

    def _act_type(self) -> int:
        hidden_act = str(getattr(self.config, "hidden_act", "silu")).lower()
        if hidden_act != "silu":
            raise RuntimeError(f"Unsupported hidden_act for packed CUDA path: {hidden_act}")
        return 0

    def _invalidate_runtime_cache(self) -> None:
        self._runtime_cache_ready = False
        self._runtime_cache_device = None
        self._cached_token_count = -1
        self._cached_flat_token_ids = None

    def _ensure_runtime_cache(self, device: torch.device) -> None:
        if self._runtime_cache_ready and self._runtime_cache_device == device:
            return

        packed_items = []
        for expert_idx, expert_layer in enumerate(self.experts):
            if expert_layer is None:
                continue
            if getattr(expert_layer, "skip_expert", False):
                continue
            if not getattr(expert_layer, "is_bitsmoe_packed", False):
                raise RuntimeError(
                    f"Layer {self.layer_idx} expert {expert_idx} is active but not packed. "
                    "BitsMoE forward expects routed experts to use the packed kernel."
                )
            packed_items.append((expert_idx, expert_layer))

        self._packed_expert_count = len(packed_items)
        self._packed_intermediate_size = 0
        self._cached_token_count = -1
        self._cached_flat_token_ids = None

        if self._packed_expert_count > 0:
            packed_global_ids = [idx for idx, _ in packed_items]
            packed_experts = [layer for _, layer in packed_items]
            self._packed_intermediate_size = int(packed_experts[0].intermediate_size)
            map_tensor = torch.full(
                (self.num_experts,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            gid_t = torch.tensor(packed_global_ids, dtype=torch.long, device=device)
            lid_t = torch.arange(self._packed_expert_count, dtype=torch.int32, device=device)
            map_tensor[gid_t] = lid_t
            self._packed_global_to_local = map_tensor

            self._gate_payload_static = tuple(expert.gate_payload_buffer for expert in packed_experts)
            self._gate_rank_idx_static = tuple(expert.gate_rank_idx_buffer for expert in packed_experts)
            self._gate_tile_meta_static = tuple(expert.gate_tile_meta_buffer for expert in packed_experts)
            self._gate_slab_meta_static = tuple(expert.gate_slab_meta_buffer for expert in packed_experts)
            self._gate_scale_static = tuple(expert.gate_scale_buffer for expert in packed_experts)
            self._gate_s_static = tuple(expert.gate_s_buffer for expert in packed_experts)

            self._up_payload_static = tuple(expert.up_payload_buffer for expert in packed_experts)
            self._up_rank_idx_static = tuple(expert.up_rank_idx_buffer for expert in packed_experts)
            self._up_tile_meta_static = tuple(expert.up_tile_meta_buffer for expert in packed_experts)
            self._up_slab_meta_static = tuple(expert.up_slab_meta_buffer for expert in packed_experts)
            self._up_scale_static = tuple(expert.up_scale_buffer for expert in packed_experts)
            self._up_s_static = tuple(expert.up_s_buffer for expert in packed_experts)

            self._down_payload_static = tuple(expert.down_payload_buffer for expert in packed_experts)
            self._down_rank_idx_static = tuple(expert.down_rank_idx_buffer for expert in packed_experts)
            self._down_tile_meta_static = tuple(expert.down_tile_meta_buffer for expert in packed_experts)
            self._down_slab_meta_static = tuple(expert.down_slab_meta_buffer for expert in packed_experts)
            self._down_scale_static = tuple(expert.down_scale_buffer for expert in packed_experts)
            self._down_s_static = tuple(expert.down_s_buffer for expert in packed_experts)
        else:
            self._packed_global_to_local = torch.full(
                (self.num_experts,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._gate_payload_static = ()
            self._gate_rank_idx_static = ()
            self._gate_tile_meta_static = ()
            self._gate_slab_meta_static = ()
            self._gate_scale_static = ()
            self._gate_s_static = ()
            self._up_payload_static = ()
            self._up_rank_idx_static = ()
            self._up_tile_meta_static = ()
            self._up_slab_meta_static = ()
            self._up_scale_static = ()
            self._up_s_static = ()
            self._down_payload_static = ()
            self._down_rank_idx_static = ()
            self._down_tile_meta_static = ()
            self._down_slab_meta_static = ()
            self._down_scale_static = ()
            self._down_s_static = ()

        self._runtime_cache_ready = True
        self._runtime_cache_device = device

    def _get_flat_token_ids(self, token_count: int, device: torch.device) -> torch.Tensor:
        if (
            self._cached_flat_token_ids is not None
            and self._cached_token_count == token_count
            and self._cached_flat_token_ids.device == device
        ):
            return self._cached_flat_token_ids

        token_ids = torch.arange(token_count, dtype=torch.long, device=device).repeat_interleave(self.top_k)
        self._cached_flat_token_ids = token_ids
        self._cached_token_count = token_count
        return token_ids

    def _build_packed_routing(
        self,
        flat_selected_experts: torch.Tensor,
        flat_token_ids: torch.Tensor,
        flat_route_weights_fp32: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._packed_expert_count <= 0 or self._packed_global_to_local is None:
            return None

        local_ids = self._packed_global_to_local[flat_selected_experts]
        packed_mask = local_ids >= 0
        if not packed_mask.any():
            return None

        local_ids = local_ids[packed_mask].to(torch.int64)
        token_ids = flat_token_ids[packed_mask]
        route_weights = flat_route_weights_fp32[packed_mask]

        order = torch.argsort(local_ids)
        local_sorted = local_ids[order]
        token_indices = token_ids[order].to(torch.int32).contiguous()
        route_flat = route_weights[order].contiguous()

        counts = torch.bincount(local_sorted, minlength=self._packed_expert_count)
        expert_offsets_t = torch.empty(
            (self._packed_expert_count + 1,),
            dtype=torch.int32,
            device=token_indices.device,
        )
        expert_offsets_t[0] = 0
        expert_offsets_t[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
        return token_indices, expert_offsets_t, route_flat

    def _packed_forward_cuda(
        self,
        h_gate_proj: torch.Tensor,
        h_up_proj: torch.Tensor,
        token_indices: torch.Tensor,
        expert_offsets_t: torch.Tensor,
        route_flat: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        rank_out = self._shared_basis_rank("shared_u_down")
        intermediate_size = int(self._packed_intermediate_size)
        if intermediate_size <= 0:
            raise RuntimeError(f"Layer {self.layer_idx} has no packed experts for CUDA path.")
        rank_accum = mlp_forward_cuda.moe_packed_forward(
            h_gate_proj,
            h_up_proj,
            token_indices,
            expert_offsets_t,
            route_flat,
            self._gate_payload_static,
            self._gate_rank_idx_static,
            self._gate_tile_meta_static,
            self._gate_slab_meta_static,
            self._gate_scale_static,
            self._gate_s_static,
            self._up_payload_static,
            self._up_rank_idx_static,
            self._up_tile_meta_static,
            self._up_slab_meta_static,
            self._up_scale_static,
            self._up_s_static,
            self._down_payload_static,
            self._down_rank_idx_static,
            self._down_tile_meta_static,
            self._down_slab_meta_static,
            self._down_scale_static,
            self._down_s_static,
            rank_out,
            intermediate_size,
            self._act_type(),
        )
        if rank_accum.shape[0] != token_count or rank_accum.shape[1] != rank_out:
            raise RuntimeError(
                f"Invalid rank_accum shape from CUDA kernel: got {tuple(rank_accum.shape)}, "
                f"expected ({token_count}, {rank_out})"
            )
        return self._shared_basis_linear(rank_accum, "shared_u_down")

    def _packed_forward_grouped(
        self,
        h_gate_proj: torch.Tensor,
        h_up_proj: torch.Tensor,
        token_indices: torch.Tensor,
        expert_offsets_t: torch.Tensor,
        route_flat: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        if self.shared_u_down_qweight.numel() == 0 or self.shared_u_down_scales.numel() == 0:
            raise RuntimeError(f"Layer {self.layer_idx} missing shared_u_down Marlin weights.")

        can_use_cuda = (
            _HAS_MLP_FORWARD_CUDA
            and mlp_forward_cuda is not None
            and h_gate_proj.is_cuda
            and h_up_proj.is_cuda
        )
        if not can_use_cuda:
            raise RuntimeError(
                "Packed routed experts require CUDA mlp_forward extension; "
                "python fallback is intentionally disabled."
            )
        return self._packed_forward_cuda(
            h_gate_proj=h_gate_proj,
            h_up_proj=h_up_proj,
            token_indices=token_indices,
            expert_offsets_t=expert_offsets_t,
            route_flat=route_flat,
            token_count=token_count,
        )
