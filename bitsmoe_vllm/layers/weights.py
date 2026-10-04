"""Load packed expert weights owned by a tensor-parallel rank."""

import torch
from torch import nn

from ..placement import TensorInfo


PROJECTIONS = ("gate", "up", "down")
BUFFER_DTYPES = {
    "payload_buffer": torch.uint32,
    "rank_idx_buffer": torch.int32,
    "tile_meta_buffer": torch.int32,
    "slab_meta_buffer": torch.int32,
    "scale_buffer": torch.float16,
    "s_buffer": torch.float16,
}
SHARED_BASES = ("shared_vh_gate_proj", "shared_vh_up_proj", "shared_u_down")


class PackedWeights(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts = nn.ModuleDict()
        self.local_ids: list[int] = []
        self._kernel = None
        self._configured = False

    def configure(
        self, owners: list[int], rank: int,
        metadata: dict[str, TensorInfo], prefix: str, device: torch.device,
        *, tp_size: int,
    ) -> None:
        if self._configured:
            raise RuntimeError("Packed weight placement is already configured")
        if tp_size < 1 or not 0 <= rank < tp_size:
            raise ValueError("Invalid shared basis tensor-parallel size or rank")
        self._configured = True
        self._device = device
        self.tp_size = tp_size
        self.tp_rank = rank
        self.ranks = tuple(
            metadata[f"{prefix}.{base}_scales"].shape[0] * 128
            if base == "shared_u_down" else metadata[f"{prefix}.{base}_scales"].shape[1]
            for base in SHARED_BASES
        )
        # A one-group partition uses a different Marlin scale permutation. Keep output sharding in that case, so loading is always pure slicing. For FP32 spectral sums / FP16 outputs, communication is proportional to 4*(R+H) for row sharding and 8*R+2*H for column sharding. Prefer columns for narrow spectral ranks when both layouts are available.
        column_valid = self.hidden_size % (64 * tp_size) == 0
        self.down_shard_dim = 0 if (tp_size > 1
            and self.ranks[2] % (128 * tp_size) == 0
            and self.ranks[2] // tp_size > 128
            and (self.hidden_size <= 2 * self.ranks[2] or not column_valid)) else 1
        self._expected: dict[str, TensorInfo] = {}
        self._loaded: set[str] = set()
        for base in SHARED_BASES:
            k_groups, n = metadata[f"{prefix}.{base}_scales"].shape
            k = k_groups * 128
            if min(k, n) <= 0 or metadata[f"{prefix}.{base}_qweight"].shape != (k // 16, n * 4):
                raise ValueError(f"Invalid W8 Marlin layout: {base}")
            if (base == "shared_u_down" and n != self.hidden_size) or (
                base != "shared_u_down" and k != self.hidden_size
            ):
                raise ValueError(f"Shared basis hidden size mismatch: {base}")
            output_tp = 1 if base == "shared_u_down" and self.down_shard_dim == 0 else tp_size
            if n % (64 * output_tp):
                raise ValueError(f"{base} output width must be divisible by 64 * TP")
            for suffix in ("qweight", "scales"):
                name = f"{base}_{suffix}"
                self._expected[name] = metadata[f"{prefix}.{name}"]
        for expert, owner in enumerate(owners):
            if owner != rank:
                continue
            expert_prefix = f"{prefix}.experts.{expert}"
            payloads = [metadata.get(f"{expert_prefix}.{tag}_payload_buffer") for tag in PROJECTIONS]
            if any(info is None for info in payloads):
                raise ValueError(f"Missing expert projection: {expert_prefix}")
            if any(info.nbytes == 0 for info in payloads):
                # An empty projection makes the routed expert identically zero.
                continue
            self.local_ids.append(expert)
            self.experts[str(expert)] = nn.Module()
            for tag in PROJECTIONS:
                for suffix in BUFFER_DTYPES:
                    name = f"experts.{expert}.{tag}_{suffix}"
                    self._expected[name] = metadata[f"{prefix}.{name}"]

    def load_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self._kernel is not None:
            raise RuntimeError("Reloading captured packed weights is unsupported")
        info = self._expected.get(name)
        if info is None:
            return
        if name in self._loaded:
            raise ValueError(f"Duplicate packed tensor: {name}")
        if tuple(tensor.shape) != info.shape:
            raise ValueError(f"Packed tensor shape mismatch: {name}")
        if name.startswith("experts."):
            _, expert, leaf = name.split(".")
            dtype = BUFFER_DTYPES[leaf.split("_", 1)[1]]
            module = self.experts[expert]
        else:
            leaf = name
            dtype = torch.int32 if name.endswith("_qweight") else torch.float16
            module = self
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if module is self and self.tp_size > 1:
            # Slice complete 16x64 Marlin tiles / 128-row quantization groups. Neither the packed bytes nor the scale permutation changes.
            dim = self.down_shard_dim if name.startswith("shared_u_down_") else 1
            width = tensor.shape[dim] // self.tp_size
            tensor = tensor.narrow(dim, self.tp_rank * width, width)
        # A contiguous row view may still own the full source storage if the loader already supplied a tensor on the destination device.
        copy_shard = module is self and self.tp_size > 1
        module.register_buffer(leaf, tensor.to(device=self._device, copy=copy_shard).contiguous())
        self._loaded.add(name)

    def prepare(self) -> None:
        missing = self._expected.keys() - self._loaded
        if missing:
            raise ValueError(f"Missing packed tensors: {sorted(missing)[:8]}")
        for base in SHARED_BASES:
            weight = getattr(self, f"{base}_qweight")
            scales = getattr(self, f"{base}_scales")
            if scales.ndim != 2 or weight.ndim != 2:
                raise ValueError(f"Invalid shared basis dimensions: {base}")
            k, n = scales.shape[0] * 128, scales.shape[1]
            if min(k, n) <= 0 or tuple(weight.shape) != (k // 16, n * 4):
                raise ValueError(f"Invalid W8 Marlin layout: {base}")
            output_tp = 1 if base == "shared_u_down" and self.down_shard_dim == 0 else self.tp_size
            if (base == "shared_u_down" and n * output_tp != self.hidden_size) or (
                base != "shared_u_down" and k != self.hidden_size
            ):
                raise ValueError(f"Shared basis hidden size mismatch: {base}")
        buffers = []
        for tag, rank in zip(PROJECTIONS, self.ranks):
            for expert in self.experts.values():
                indices = getattr(expert, f"{tag}_rank_idx_buffer").cpu()
                singular = getattr(expert, f"{tag}_s_buffer")
                if indices.ndim != 1 or indices.numel() != singular.numel():
                    raise ValueError(f"Invalid {tag} spectral indices")
                if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= rank):
                    raise ValueError(f"{tag} spectral index exceeds shared basis rank")
                slabs = getattr(expert, f"{tag}_slab_meta_buffer").cpu()
                if slabs.ndim != 2 or slabs.shape[1] != 4:
                    raise ValueError(f"Invalid {tag} slab metadata")
                if not all(int(bits) in (1, 2, 3, 4, 6, 8, 16) for bits in slabs[:, 0]):
                    raise ValueError(f"Unsupported {tag} slab bit width")
            buffers.extend([
                [getattr(self.experts[str(e)], f"{tag}_{suffix}") for e in self.local_ids]
                for suffix in BUFFER_DTYPES
            ])
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
        for local, global_id in enumerate(self.local_ids):
            expert_map[global_id] = local
        self.register_buffer("expert_map", expert_map.to(self._device), persistent=False)
        from .triton_grouped import VllmTritonPackedMoE

        self._kernel = VllmTritonPackedMoE(
            buffers, self.expert_map, *self.ranks, self.intermediate_size
        )

    def forward(
        self, h_gate: torch.Tensor, h_up: torch.Tensor,
        selected: torch.Tensor, weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._kernel is None:
            raise RuntimeError("Packed weights must be loaded before inference")
        output_shards = self.tp_size if self.down_shard_dim == 0 else 1
        if selected.shape[0] == 1 and selected.numel() <= self._kernel.expert_map.numel() // 2:
            if output_shards < 1 or self._kernel.ranks[2] % output_shards:
                raise ValueError("Output rank must be divisible by output_shards")
            from bitsmoe.algorithms.triton_decode import single_token_forward

            return single_token_forward(self._kernel, h_gate, h_up, selected, weights, output_shards)
        return self._kernel.forward(h_gate, h_up, selected, weights,
                                    output_shards=output_shards)
