import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from bitsmoe.algorithms.vllm_compat import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
    scalar_types,
)

from ..compat import custom_op_registration
from .weights import PackedWeights, SHARED_BASES
from .shared_basis import split_gathered


def _moe_forward(
    hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.forward_impl(hidden_states, router_logits)


def _moe_forward_fake(
    hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


custom_op_registration()(
    op_name="bitsmoe_forward", op_func=_moe_forward, mutates_args=[],
    fake_impl=_moe_forward_fake, tags=(torch.Tag.needs_fixed_stride_order,),
)


class BitsMoESparseMoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.layer_name = prefix
        num_experts = getattr(config, "num_experts", None) or config.n_routed_experts
        self.gate = ReplicatedLinear(config.hidden_size, num_experts, bias=False)
        self.packed = PackedWeights(
            config.hidden_size, config.moe_intermediate_size, num_experts
        )
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self
        self._workspaces = {}

    def prepare(self) -> None:
        self.packed.prepare()
        device = self.gate.weight.device
        self.register_buffer("_empty", torch.empty(0, dtype=torch.int32, device=device), persistent=False)
        for base in SHARED_BASES:
            workspace = marlin_make_workspace_new(device)
            self.register_buffer(f"_{base}_workspace", workspace, persistent=False)
            self._workspaces[base] = workspace
        # Concatenate whole Marlin tiles, retaining the checkpoint format. A single GEMM serves both input projections, including on one GPU.
        for suffix in ("qweight", "scales"):
            gate = getattr(self.packed, f"shared_vh_gate_proj_{suffix}")
            up = getattr(self.packed, f"shared_vh_up_proj_{suffix}")
            self.packed.register_buffer(f"shared_gate_up_{suffix}", torch.cat((gate, up), dim=1))
            delattr(self.packed, f"shared_vh_gate_proj_{suffix}")
            delattr(self.packed, f"shared_vh_up_proj_{suffix}")
        self._workspaces["shared_gate_up"] = self._workspaces["shared_vh_gate_proj"]

    def shared_linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        scales = getattr(self.packed, f"{name}_scales")
        expand_rows = max(128, scales.numel() // (512 if name == "shared_gate_up" else 256))
        if x.shape[0] >= expand_rows:
            from bitsmoe.algorithms.triton_shared import prefill_shared_linear

            return prefill_shared_linear(
                x, getattr(self.packed, f"{name}_qweight"),
                getattr(self.packed, f"{name}_scales"),
            )
        # Larger row counts can produce incorrect results in vLLM 0.11's W8 Marlin kernel.
        if x.shape[0] > 32:
            return torch.cat([self.shared_linear(chunk, name) for chunk in x.split(32)])
        return apply_gptq_marlin_linear(
            input=x.to(torch.float16).contiguous(),
            weight=getattr(self.packed, f"{name}_qweight"),
            weight_scale=scales, weight_zp=self._empty,
            g_idx=self._empty, g_idx_sort_indices=self._empty,
            workspace=self._workspaces[name], wtype=scalar_types.uint8b128,
            output_size_per_partition=scales.shape[1],
            input_size_per_partition=scales.shape[0] * 128, is_k_full=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return torch.ops.vllm.bitsmoe_forward(hidden_states, router_logits, self.layer_name)

    def route(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        from bitsmoe.algorithms.triton_routing import route

        return route(router_logits, self.top_k, self.renormalize)

    @property
    def combines_shared_expert_reduce(self):
        return (getattr(self, "_combine_shared_expert_reduce", False)
                and self.tp_size > 1 and self.packed.down_shard_dim == 0)

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        weights, selected = self.route(hidden_states, router_logits)
        gate_rank, up_rank, _ = self.packed.ranks
        projected = self.shared_linear(hidden_states, "shared_gate_up")
        if self.tp_size > 1:
            tokens = hidden_states.shape[0]
            # vLLM's equal-size all_gatherv uses PyNccl directly, including during CUDA Graph capture; avoid the torch.distributed wrapper.
            gathered = get_tp_group().all_gatherv(projected, dim=0)
            gate, up = split_gathered(gathered, tokens, gate_rank, up_rank, self.tp_size)
            accum = self.packed(gate, up, selected, weights)
            if self.packed.down_shard_dim == 0:
                # Sum experts in FP32 and keep only this rank's spectral slice. As in vLLM RowParallelLinear, reduce the FP16 down-projection partial results after the local GEMM.
                accum = get_tp_group().reduce_scatterv(
                    accum.view(self.tp_size * tokens, self.packed.ranks[2] // self.tp_size), dim=0)
                output = self.shared_linear(accum, "shared_u_down")
                if self.combines_shared_expert_reduce:
                    return output.to(hidden_states.dtype)
                return tensor_model_parallel_all_reduce(output).to(hidden_states.dtype)
            accum = tensor_model_parallel_all_reduce(accum)
            output = self.shared_linear(accum, "shared_u_down")
            gathered = get_tp_group().all_gatherv(output, dim=0)
            return gathered.view(self.tp_size, tokens, self.packed.hidden_size // self.tp_size).permute(1, 0, 2).reshape(
                tokens, self.packed.hidden_size).to(hidden_states.dtype)
        gate, up = projected.split((gate_rank, up_rank), dim=-1)
        accum = self.packed(gate.contiguous(), up.contiguous(), selected, weights)
        return self.shared_linear(accum, "shared_u_down").to(hidden_states.dtype)
