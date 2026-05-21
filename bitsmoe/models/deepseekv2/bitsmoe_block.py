import copy
from typing import List, Optional

import torch
import torch.nn as nn

from bitsmoe.models.base_block import BitsMoE_BaseSparseMoeBlock


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class BitsMoE_DeepSeekSparseMoeBlock(BitsMoE_BaseSparseMoeBlock):
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
        self.num_experts_per_tok = int(getattr(config, "num_experts_per_tok", 0))
        self.top_k = self.num_experts_per_tok

        if source_block is None:
            raise ValueError("source_block is required for BitsMoE_DeepSeekSparseMoeBlock")

        self.gate = copy.deepcopy(source_block.gate)
        if copy_source_experts:
            self.experts = nn.ModuleList([
                copy.deepcopy(expert) if expert is not None else None
                for expert in source_block.experts
            ])
        else:
            self.experts = nn.ModuleList([None] * len(source_block.experts))
        self.num_experts = len(self.experts)

        if hasattr(source_block, "shared_experts"):
            self.shared_experts = copy.deepcopy(source_block.shared_experts)

        if hasattr(source_block, "ep_size"):
            self.ep_size = int(source_block.ep_size)
        if hasattr(source_block, "experts_per_rank"):
            self.experts_per_rank = int(source_block.experts_per_rank)
        if hasattr(source_block, "ep_rank"):
            self.ep_rank = int(source_block.ep_rank)

        self._init_bitsmoe_common_state(
            super_experts=super_experts,
            track_super_projected_weights=True,
        )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        token_count = hidden_states.shape[0]
        self._ensure_runtime_cache(hidden_states.device)

        topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
        topk_weight_fp32 = topk_weight.reshape(-1, topk_weight.shape[-1]).to(torch.float32)
        topk_weight_hidden = topk_weight_fp32.to(hidden_states.dtype)

        if self.shared_vh_gate_proj.numel() == 0 or self.shared_vh_up_proj.numel() == 0:
            raise RuntimeError(
                f"Layer {self.layer_idx} missing shared_vh basis for packed routed experts."
            )
        h_gate_proj = hidden_states @ self.shared_vh_gate_proj.T
        h_up_proj = hidden_states @ self.shared_vh_up_proj.T

        final_hidden_states = torch.zeros_like(hidden_states)

        flat_selected_experts = topk_idx.reshape(-1)
        flat_route_weights_hidden = topk_weight_hidden.reshape(-1)
        flat_route_weights_fp32 = topk_weight_fp32.reshape(-1)
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

        y = final_hidden_states.reshape(*orig_shape)
        if getattr(self.config, "n_shared_experts", None) is not None and hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity)
        if self.training and aux_loss is not None:
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        return y
