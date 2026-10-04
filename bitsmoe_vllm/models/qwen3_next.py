"""Qwen3Next hybrid attention with BitsMoE packed experts."""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet, Qwen3NextMLP, Qwen3NextModel, Qwen3NextRMSNorm,
)
from vllm.model_executor.models.utils import extract_layer_index

from ..layers.moe import BitsMoESparseMoE
from .base import PackedModel, PackedForCausalLM


class BitsMoEQwen3NextMoE(BitsMoESparseMoE):
    def __init__(self, vllm_config, prefix):
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_text_config
        self.shared_expert = None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                config.hidden_size, config.shared_expert_intermediate_size,
                config.hidden_act, reduce_results=False, prefix=f"{prefix}.shared_expert",
            )
        self._combine_shared_expert_reduce = self.shared_expert is not None
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        output = super().forward(hidden_states)
        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states)
            if self.tp_size > 1 and not self.combines_shared_expert_reduce:
                shared = tensor_model_parallel_all_reduce(shared)
            output = output + self.shared_expert_gate(hidden_states).sigmoid() * shared
            if self.combines_shared_expert_reduce:
                output = tensor_model_parallel_all_reduce(output)
        return output


class BitsMoEQwen3NextDecoderLayer(nn.Module):
    forward = Qwen3NextDecoderLayer.forward

    def __init__(self, vllm_config, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.layer_idx = extract_layer_index(prefix)
        self.layer_type = config.layer_types[self.layer_idx]
        attention_args = dict(
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config, **attention_args, speculative_config=None,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config, **attention_args, prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Unsupported Qwen3Next layer type: {self.layer_type}")
        if self.layer_idx not in getattr(config, "mlp_only_layers", []) and (
            config.num_experts > 0 and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = BitsMoEQwen3NextMoE(vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = Qwen3NextMLP(
                config.hidden_size, config.intermediate_size, config.hidden_act,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))


@support_torch_compile
class BitsMoEQwen3NextModel(PackedModel):
    forward = Qwen3NextModel.forward
    decoder_class = BitsMoEQwen3NextDecoderLayer
    norm_class = Qwen3NextRMSNorm


class BitsMoEQwen3NextForCausalLM(PackedForCausalLM):
    model_class = BitsMoEQwen3NextModel
    packed_modules_mapping = Qwen3NextForCausalLM.packed_modules_mapping
    has_inner_state = True
    is_hybrid = True
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3NextForCausalLM.get_mamba_state_dtype_from_config.__func__
    )
    get_mamba_state_shape_from_config = classmethod(
        Qwen3NextForCausalLM.get_mamba_state_shape_from_config.__func__
    )

    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Qwen3Next does not support prefix caching in vLLM 0.11.0")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
