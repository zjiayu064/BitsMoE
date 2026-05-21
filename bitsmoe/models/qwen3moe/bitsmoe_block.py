import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsmoe.models.base_block import BitsMoE_BaseSparseMoeBlock


class BitsMoE_Qwen3MoeSparseMoeBlock(BitsMoE_BaseSparseMoeBlock):
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
            raise ValueError("source_block is required for BitsMoE_Qwen3MoeSparseMoeBlock")

        self.gate = copy.deepcopy(source_block.gate)
        # Experts are populated by the caller after init.
        self.experts = nn.ModuleList([None] * len(source_block.experts))

        self._init_bitsmoe_common_state(
            super_experts=super_experts,
            track_super_projected_weights=True,
        )

    def forward(self, hidden_states):
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

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
