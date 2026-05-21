import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsmoe.models.base_block import BitsMoE_BaseSparseMoeBlock


class BitsMoE_Qwen3NextSparseMoeBlock(BitsMoE_BaseSparseMoeBlock):
    def __init__(
        self,
        config,
        layer_idx: int,
        source_block: Optional[nn.Module] = None,
        super_experts: Optional[List[int]] = None,
        copy_source_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_experts = int(getattr(config, "num_experts", 0))
        self.top_k = int(getattr(config, "num_experts_per_tok", 0))
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))

        if source_block is None:
            raise ValueError("source_block is required for BitsMoE_Qwen3NextSparseMoeBlock")

        self.gate = copy.deepcopy(source_block.gate)
        if copy_source_experts:
            self.experts = nn.ModuleList([copy.deepcopy(expert) for expert in source_block.experts])
        else:
            self.experts = nn.ModuleList([None] * len(source_block.experts))

        if hasattr(source_block, "shared_expert"):
            self.shared_expert = copy.deepcopy(source_block.shared_expert)
        if hasattr(source_block, "shared_expert_gate"):
            self.shared_expert_gate = copy.deepcopy(source_block.shared_expert_gate)

        self._init_bitsmoe_common_state(
            super_experts=super_experts,
            track_super_projected_weights=False,
        )

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
            tensor = state_dict.get(f"{prefix}{local_key}", None)
            if not isinstance(tensor, torch.Tensor):
                continue

            current = getattr(self, local_key, None)
            if (
                isinstance(current, torch.Tensor)
                and current.shape == tensor.shape
                and current.dtype == tensor.dtype
            ):
                continue

            replacement = tensor.detach() if assign_to_params_buffers else torch.empty_like(tensor)
            setattr(self, local_key, replacement)

        nn.Module._load_from_state_dict(
            self,
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        self._invalidate_runtime_cache()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        token_count = hidden_states.shape[0]
        self._ensure_runtime_cache(hidden_states.device)
        router_logits = self.gate(hidden_states)

        routing_weights_fp32 = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_fp32, selected_experts = torch.topk(routing_weights_fp32, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights_fp32 /= routing_weights_fp32.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights_fp32.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (token_count, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if self.shared_vh_gate_proj.numel() == 0 or self.shared_vh_up_proj.numel() == 0:
            raise RuntimeError(
                f"Layer {self.layer_idx} missing shared_vh basis for packed routed experts."
            )
        h_gate_proj = hidden_states @ self.shared_vh_gate_proj.T
        h_up_proj = hidden_states @ self.shared_vh_up_proj.T

        flat_selected_experts = selected_experts.reshape(-1)
        flat_route_weights_hidden = routing_weights.reshape(-1)
        flat_route_weights_fp32 = flat_route_weights_hidden.to(torch.float32)
        flat_token_ids = self._get_flat_token_ids(token_count, hidden_states.device)

        if self._super_expert_items:
            self._run_super_experts(
                hidden_states=hidden_states,
                final_hidden_states=final_hidden_states,
                flat_selected_experts=flat_selected_experts,
                flat_token_ids=flat_token_ids,
                flat_route_weights_hidden=flat_route_weights_hidden,
            )

        packed_route = self._build_packed_routing(
            flat_selected_experts=flat_selected_experts,
            flat_token_ids=flat_token_ids,
            flat_route_weights_fp32=flat_route_weights_fp32,
        )
        if packed_route is not None:
            token_indices, expert_offsets_t, route_flat = packed_route
            packed_hidden = self._packed_forward_grouped(
                h_gate_proj=h_gate_proj,
                h_up_proj=h_up_proj,
                token_indices=token_indices,
                expert_offsets_t=expert_offsets_t,
                route_flat=route_flat,
                token_count=token_count,
            )
            final_hidden_states = final_hidden_states + packed_hidden.to(hidden_states.dtype)

        if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
