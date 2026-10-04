import copy
from typing import Optional

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

        self._init_bitsmoe_common_state()

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights_fp32 = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_fp32, selected_experts = torch.topk(routing_weights_fp32, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights_fp32 /= routing_weights_fp32.sum(dim=-1, keepdim=True)
        final_hidden_states = self._routed_forward(
            hidden_states, selected_experts, routing_weights_fp32.to(hidden_states.dtype).float(),
        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
