"""DeepSeekV2 native MLA with BitsMoE packed experts."""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLAAttention,
    DeepseekV2MLP, DeepseekV2Model,
)
from vllm.model_executor.models.utils import extract_layer_index

from ..layers.moe import BitsMoESparseMoE
from .base import PackedModel, PackedForCausalLM


class BitsMoEDeepseekV2MoE(BitsMoESparseMoE):
    def __init__(self, vllm_config, prefix):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_text_config
        self.topk_method = config.topk_method
        self.scoring_func = config.scoring_func
        self.num_groups = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.topk_method not in ("greedy", "group_limited_greedy", "noaux_tc"):
            raise ValueError(f"Unsupported DeepSeek routing method: {self.topk_method}")
        if self.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(f"Unsupported DeepSeek scoring function: {self.scoring_func}")
        if self.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                config.hidden_size, config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act, reduce_results=False, prefix=f"{prefix}.shared_experts",
            )
        self._combine_shared_expert_reduce = self.shared_experts is not None

    def route(self, hidden_states, router_logits):
        logits = router_logits.float()
        if self.topk_method != "greedy":
            from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk

            return grouped_topk(
                hidden_states, logits, self.top_k, self.renormalize,
                num_expert_group=self.num_groups, topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                e_score_correction_bias=getattr(self.gate, "e_score_correction_bias", None),
            )
        scores = logits.softmax(dim=-1) if self.scoring_func == "softmax" else logits.sigmoid()
        weights, selected = scores.topk(self.top_k, dim=-1)
        if self.renormalize and self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, selected

    def forward(self, hidden_states):
        output = super().forward(hidden_states)
        # Match the residual scaling in vLLM's FP16 DeepSeek decoder.
        if hidden_states.dtype != torch.float16:
            output = output * self.routed_scaling_factor
        if self.shared_experts is not None:
            shared = self.shared_experts(hidden_states)
            if self.tp_size > 1 and not self.combines_shared_expert_reduce:
                shared = tensor_model_parallel_all_reduce(shared)
            if hidden_states.dtype == torch.float16:
                shared = shared / self.routed_scaling_factor
            output = output + shared
            if self.combines_shared_expert_reduce:
                output = tensor_model_parallel_all_reduce(output)
        return output


class BitsMoEDeepseekV2DecoderLayer(nn.Module):
    forward = DeepseekV2DecoderLayer.forward

    def __init__(self, vllm_config, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        self.routed_scaling_factor = config.routed_scaling_factor
        attention_class = DeepseekV2MLAAttention if vllm_config.model_config.use_mla else DeepseekV2Attention
        self.self_attn = attention_class(
            vllm_config=vllm_config, config=config, hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim, q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank, rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=vllm_config.cache_config, quant_config=None,
            prefix=f"{prefix}.self_attn", topk_indices_buffer=None,
        )
        if config.n_routed_experts is not None and (
            self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = BitsMoEDeepseekV2MoE(vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = DeepseekV2MLP(
                config.hidden_size, config.intermediate_size, config.hidden_act,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@support_torch_compile
class BitsMoEDeepseekV2Model(PackedModel):
    forward = DeepseekV2Model.forward
    decoder_class = BitsMoEDeepseekV2DecoderLayer


class BitsMoEDeepseekV2ForCausalLM(PackedForCausalLM):
    model_class = BitsMoEDeepseekV2Model
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
