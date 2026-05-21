from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from bitsmoe.algorithms import _HAS_MLP_FORWARD_CUDA, mlp_forward_cuda


class BitsMoE_BaseSparseMoeBlock(nn.Module):
    _SHARED_BASIS_BUFFER_KEYS = (
        "shared_vh_gate_proj",
        "shared_vh_up_proj",
        "shared_u_down",
    )

    def _init_bitsmoe_common_state(
        self,
        super_experts: Optional[List[int]] = None,
        track_super_projected_weights: bool = False,
    ) -> None:
        self.super_experts = sorted(set(int(x) for x in (super_experts or [])))
        self._super_expert_set = set(self.super_experts)
        self.packed_expert_mask = [False for _ in range(len(self.experts))]
        if track_super_projected_weights:
            self.super_projected_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        self.register_buffer("shared_vh_gate_proj", torch.empty(0, dtype=torch.float16), persistent=True)
        self.register_buffer("shared_vh_up_proj", torch.empty(0, dtype=torch.float16), persistent=True)
        self.register_buffer("shared_u_down", torch.empty(0, dtype=torch.float16), persistent=True)

        # Runtime caches for packed path (built lazily on target device).
        self._runtime_cache_ready = False
        self._runtime_cache_device: Optional[torch.device] = None
        self._runtime_warmed = False
        self._cached_token_count = -1
        self._cached_flat_token_ids: Optional[torch.Tensor] = None

        self._super_expert_items: List[Tuple[int, nn.Module]] = []
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

    def _runtime_device(self) -> torch.device:
        gate = getattr(self, "gate", None)
        weight = getattr(gate, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight.device
        return next(gate.parameters()).device

    def _set_buffer(self, name: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().to(
            device=self._runtime_device(),
            dtype=torch.float16,
            non_blocking=True,
        ).contiguous()
        setattr(self, name, t)

    def set_shared_basis(self, gate_vh: torch.Tensor, up_vh: torch.Tensor, down_u: torch.Tensor) -> None:
        self._set_buffer("shared_vh_gate_proj", gate_vh)
        self._set_buffer("shared_vh_up_proj", up_vh)
        self._set_buffer("shared_u_down", down_u)
        self._invalidate_runtime_cache()

    def set_expert(self, expert_idx: int, expert_module: nn.Module, packed: bool) -> None:
        idx = int(expert_idx)
        self.experts[idx] = expert_module
        self.packed_expert_mask[idx] = bool(packed)
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
        for key, tensor in state_dict.items():
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            if "." in local_key:
                continue
            if not hasattr(self, local_key):
                continue
            current = getattr(self, local_key)
            if not isinstance(current, torch.Tensor):
                continue
            if current.shape == tensor.shape and current.dtype == tensor.dtype:
                continue
            setattr(self, local_key, torch.empty_like(tensor))

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        self._invalidate_runtime_cache()

    def set_super_expert_projected_weights(
        self,
        expert_idx: int,
        gate_us: torch.Tensor,
        up_us: torch.Tensor,
        down_svh: torch.Tensor,
    ) -> None:
        if not hasattr(self, "super_projected_weights"):
            self.super_projected_weights = {}
        self.super_projected_weights[int(expert_idx)] = {
            "gate_us": gate_us.detach().to(dtype=torch.float16).contiguous(),
            "up_us": up_us.detach().to(dtype=torch.float16).contiguous(),
            "down_svh": down_svh.detach().to(dtype=torch.float16).contiguous(),
        }

    def _act_type(self) -> int:
        hidden_act = str(getattr(self.config, "hidden_act", "silu")).lower()
        if hidden_act != "silu":
            raise RuntimeError(f"Unsupported hidden_act for packed CUDA path: {hidden_act}")
        return 0

    def _invalidate_runtime_cache(self) -> None:
        self._runtime_cache_ready = False
        self._runtime_cache_device = None
        self._runtime_warmed = False
        self._cached_token_count = -1
        self._cached_flat_token_ids = None

    def _ensure_runtime_cache(self, device: torch.device) -> None:
        if self._runtime_cache_ready and self._runtime_cache_device == device:
            return

        super_items: List[Tuple[int, nn.Module]] = []
        packed_items: List[Tuple[int, nn.Module]] = []
        for expert_idx, expert_layer in enumerate(self.experts):
            if expert_layer is None:
                continue
            if expert_idx in self._super_expert_set:
                super_items.append((expert_idx, expert_layer))
                continue
            if not getattr(expert_layer, "is_bitsmoe_packed", False):
                raise RuntimeError(
                    f"Layer {self.layer_idx} expert {expert_idx} is active but not packed. "
                    "BitsMoE forward expects non-super routed experts to use packed kernel."
                )
            packed_items.append((expert_idx, expert_layer))

        self._super_expert_items = super_items
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
        self._runtime_warmed = False

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

    def _run_super_experts(
        self,
        hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
        flat_selected_experts: torch.Tensor,
        flat_token_ids: torch.Tensor,
        flat_route_weights_hidden: torch.Tensor,
    ) -> None:
        for expert_idx, expert_layer in self._super_expert_items:
            mask = flat_selected_experts == expert_idx
            if not mask.any():
                continue
            token_ids = flat_token_ids[mask]
            current_state = hidden_states.index_select(0, token_ids)
            current_hidden_states = expert_layer(current_state)
            current_hidden_states = current_hidden_states * flat_route_weights_hidden[mask].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_ids, current_hidden_states.to(hidden_states.dtype))

    def prepare_runtime_fastpath(self, warmup: bool = True) -> None:
        device = self._runtime_device()
        self._ensure_runtime_cache(device)
        if warmup:
            self._warmup_runtime_once(device)

    def _warmup_runtime_once(self, device: torch.device) -> None:
        if self._runtime_warmed:
            return
        if device.type != "cuda":
            return
        if not (_HAS_MLP_FORWARD_CUDA and mlp_forward_cuda is not None):
            return
        if self._packed_expert_count <= 0:
            self._runtime_warmed = True
            return
        if self.shared_vh_gate_proj.numel() == 0 or self.shared_vh_up_proj.numel() == 0:
            return

        rank_dim = int(self.shared_vh_gate_proj.shape[0])
        if rank_dim <= 0:
            return

        active_assign = min(self.top_k, self._packed_expert_count)
        if active_assign <= 0:
            self._runtime_warmed = True
            return

        h_gate_proj = torch.zeros((1, rank_dim), dtype=torch.float16, device=device)
        h_up_proj = torch.zeros((1, rank_dim), dtype=torch.float16, device=device)
        token_indices = torch.zeros((active_assign,), dtype=torch.int32, device=device)
        route_flat = torch.full(
            (active_assign,),
            1.0 / float(active_assign),
            dtype=torch.float32,
            device=device,
        )
        counts = torch.zeros((self._packed_expert_count,), dtype=torch.int32, device=device)
        counts[:active_assign] = 1
        expert_offsets_t = torch.empty(
            (self._packed_expert_count + 1,),
            dtype=torch.int32,
            device=device,
        )
        expert_offsets_t[0] = 0
        expert_offsets_t[1:] = torch.cumsum(counts, dim=0)

        with torch.no_grad():
            _ = self._packed_forward_cuda(
                h_gate_proj=h_gate_proj,
                h_up_proj=h_up_proj,
                token_indices=token_indices,
                expert_offsets_t=expert_offsets_t,
                route_flat=route_flat,
                token_count=1,
            )
        torch.cuda.synchronize(device)
        self._runtime_warmed = True

    def _packed_forward_cuda(
        self,
        h_gate_proj: torch.Tensor,
        h_up_proj: torch.Tensor,
        token_indices: torch.Tensor,
        expert_offsets_t: torch.Tensor,
        route_flat: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        rank_out = int(self.shared_u_down.shape[1])
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
        return torch.matmul(rank_accum.to(self.shared_u_down.dtype), self.shared_u_down.T)

    def _packed_forward_grouped(
        self,
        h_gate_proj: torch.Tensor,
        h_up_proj: torch.Tensor,
        token_indices: torch.Tensor,
        expert_offsets_t: torch.Tensor,
        route_flat: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        if self.shared_u_down.numel() == 0:
            raise RuntimeError(f"Layer {self.layer_idx} missing shared_u_down for packed routed experts.")

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
