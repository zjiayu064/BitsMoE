"""Qwen3MoE native attention with BitsMoE packed experts."""

from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM, Qwen3MoeMLP, Qwen3MoeModel,
)
from vllm.model_executor.models.utils import extract_layer_index

from ..layers.moe import BitsMoESparseMoE
from .base import PackedModel, PackedForCausalLM


class BitsMoEQwen3MoeDecoderLayer(nn.Module):
    forward = Qwen3MoeDecoderLayer.forward

    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=getattr(config, "dual_chunk_attention_config", None),
        )
        layer_index = extract_layer_index(prefix)
        if layer_index not in getattr(config, "mlp_only_layers", []) and (
            config.num_experts > 0 and (layer_index + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = BitsMoESparseMoE(vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(
                config.hidden_size, config.intermediate_size, config.hidden_act,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@support_torch_compile
class BitsMoEQwen3MoeModel(PackedModel):
    forward = Qwen3MoeModel.forward
    decoder_class = BitsMoEQwen3MoeDecoderLayer


class BitsMoEQwen3MoeForCausalLM(PackedForCausalLM):
    model_class = BitsMoEQwen3MoeModel
    packed_modules_mapping = Qwen3MoeForCausalLM.packed_modules_mapping
