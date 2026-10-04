"""Shared model construction and packed checkpoint loading."""

import json
import re
from itertools import chain
from pathlib import Path

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix,
)

from ..layers.moe import BitsMoESparseMoE
from ..config import validate_options
from ..placement import make_plan, read_checkpoint_metadata, validate_plan
from ..layers.weights import BUFFER_DTYPES, PROJECTIONS, SHARED_BASES


logger = init_logger(__name__)
_LAYER_WEIGHT = re.compile(r"model\.layers\.(\d+)\.mlp\.(.+)")
_EXPERT_LEAVES = {
    f"{tag}_{leaf}"
    for tag in PROJECTIONS
    for leaf in (*BUFFER_DTYPES, "segments", "original_indices", "groupsize", "original_rank")
}


class PackedModel(nn.Module):
    decoder_class = None
    norm_class = RMSNorm

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config = vllm_config.model_config.hf_text_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_class(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = self.norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )


def validate_config(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_text_config
    options = getattr(config, "bitsmoe_vllm", {})
    validate_options(options)
    parallel = vllm_config.parallel_config
    if not getattr(config, "bitsmoe", False) or config.model_type not in ("qwen3_moe", "deepseek_v2", "qwen3_next"):
        raise ValueError("This backend requires a supported packed BitsMoE checkpoint")
    if model.dtype != torch.float16 or config.hidden_act != "silu":
        raise ValueError("BitsMoE vLLM currently requires --dtype half and SiLU")
    if vllm_config.quant_config is not None:
        raise ValueError("BitsMoE loads its own packed format; do not set --quantization")
    if parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("BitsMoE supports tensor/expert parallelism with DP=1 and PP=1")
    if parallel.tensor_parallel_size > 1 and not parallel.enable_expert_parallel:
        raise ValueError("BitsMoE multi-GPU inference requires --enable-expert-parallel")
    if parallel.enable_eplb or parallel.use_sequence_parallel_moe:
        raise ValueError(
            "BitsMoE uses static expert placement; EPLB and sequence parallelism are unsupported"
        )
    if vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
        raise ValueError("LoRA and speculative decoding are not supported by this backend")
    if vllm_config.cache_config.cpu_offload_gb:
        raise ValueError("Packed expert pointers require GPU-resident weights; CPU offload is unsupported")
    if model.enable_sleep_mode:
        raise ValueError("Sleep mode can invalidate packed expert pointers and is unsupported")
    if vllm_config.load_config.load_format not in ("auto", "safetensors"):
        raise ValueError("BitsMoE requires safetensors checkpoints")


class PackedForCausalLM(nn.Module):
    model_class = None
    fall_back_to_pt_during_load = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        validate_config(vllm_config)
        self.vllm_config = vllm_config
        self.config = config = vllm_config.model_config.hf_text_config
        self.model = self.model_class(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._weights_loaded = False

    def _checkpoint_directory(self) -> Path:
        model = self.vllm_config.model_config
        directory = Path(model.model)
        if directory.is_dir():
            return directory
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(
            model.model, revision=model.revision,
            cache_dir=self.vllm_config.load_config.download_dir, local_files_only=True,
        ))

    def load_weights(self, weights) -> set[str]:
        if self._weights_loaded:
            raise RuntimeError("Create a new engine to reload BitsMoE weights")
        iterator = iter(weights)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("Checkpoint contains no weights") from None
        metadata = read_checkpoint_metadata(self._checkpoint_directory())
        rank = get_tensor_model_parallel_rank()
        world_size = self.vllm_config.parallel_config.tensor_parallel_size
        layers = {
            index: layer.mlp for index, layer in enumerate(self.model.layers)
            if isinstance(layer.mlp, BitsMoESparseMoE)
        }
        num_experts = getattr(self.config, "num_experts", None) or self.config.n_routed_experts
        options = getattr(self.config, "bitsmoe_vllm", {})
        if options.get("expert_map"):
            plan = json.loads(Path(options["expert_map"]).read_text())
        else:
            plan = make_plan(metadata, num_experts, world_size)
        validate_plan(plan, set(layers), num_experts, world_size)
        for index, layer in layers.items():
            layer.packed.configure(
                plan["layers"][str(index)], rank, metadata,
                f"model.layers.{index}.mlp", layer.gate.weight.device,
                tp_size=layer.tp_size,
            )

        params = dict(self.named_parameters())
        loaded = set()
        stacked = (
            ("q_proj", "qkv_proj", "q"), ("k_proj", "qkv_proj", "k"),
            ("v_proj", "qkv_proj", "v"), ("gate_proj", "gate_up_proj", 0),
            ("up_proj", "gate_up_proj", 1),
            ("q_a_proj", "fused_qkv_a_proj", 0),
            ("kv_a_proj_with_mqa", "fused_qkv_a_proj", 1),
        )
        shards: dict[str, set] = {}
        for name, tensor in chain((first,), iterator):
            match = _LAYER_WEIGHT.fullmatch(name)
            if match and int(match[1]) in layers:
                layer, leaf = layers[int(match[1])], match[2]
                if leaf.startswith("experts."):
                    parts = leaf.split(".")
                    if len(parts) != 3 or parts[2] not in _EXPERT_LEAVES:
                        raise ValueError(f"Unsupported packed expert tensor: {name}")
                    layer.packed.load_tensor(leaf, tensor)
                    continue
                if leaf in SHARED_BASES and tensor.numel() == 0:
                    continue
                if any(
                    leaf == f"{base}_{suffix}"
                    for base in SHARED_BASES for suffix in ("qweight", "scales")
                ):
                    layer.packed.load_tensor(leaf, tensor)
                    continue
            if name.startswith("mtp."):
                continue
            if name.endswith("rotary_emb.inv_freq"):
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            mapped, shard = name, None
            for source, target, shard_id in stacked:
                if f".{source}." in name:
                    candidate = name.replace(f".{source}.", f".{target}.")
                    if candidate in params:
                        mapped, shard = candidate, shard_id
                        break
            if mapped not in params:
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[mapped]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shard)
                shards.setdefault(mapped, set()).add(shard)
            loaded.add(mapped)
        for name, seen in shards.items():
            expected = {"q", "k", "v"} if ".qkv_proj." in name else {0, 1}
            if seen != expected:
                raise ValueError(f"Missing weight shards for {name}: {expected - seen}")
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Missing model weights: {sorted(missing)[:8]}")
        for layer in layers.values():
            layer.prepare()
        self._weights_loaded = True
        logger.info("BitsMoE rank %d/%d: loaded %d local experts across %d layers",
                    rank, world_size, sum(len(m.packed.local_ids) for m in layers.values()), len(layers))
        basis_bytes = sum(buffer.numel() * buffer.element_size()
                          for layer in layers.values()
                          for name, buffer in layer.packed.named_buffers()
                          if name.startswith("shared"))
        logger.info("BitsMoE rank %d/%d: shared basis buffers %.3f GiB (%s)",
                    rank, world_size, basis_bytes / (1024 ** 3),
                    "sharded" if world_size > 1 else "single GPU")
        return loaded
