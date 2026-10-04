"""Checkpoint metadata and deterministic expert placement."""

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


EXPERT_KEY = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)")


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


def read_checkpoint_metadata(directory: str | Path) -> dict[str, TensorInfo]:
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    weight_map = None
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = [directory / "model.safetensors"]
    result = {}
    for path in files:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 100_000_000 or length > file_size - 8:
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if weight_map is not None and weight_map.get(name) != path.name:
                continue
            if name in result:
                raise ValueError(f"Duplicate tensor: {name}")
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= file_size - 8 - length:
                raise ValueError(f"Invalid tensor offsets: {name}")
            result[name] = TensorInfo(entry["dtype"], tuple(entry["shape"]), end - start)
    if weight_map is not None and set(weight_map) != set(result):
        raise ValueError("Checkpoint index contains missing tensors")
    return result


def expert_sizes(metadata: dict[str, TensorInfo], num_experts: int) -> dict[int, list[int]]:
    layers = {}
    for name, info in metadata.items():
        match = EXPERT_KEY.fullmatch(name)
        if match is None:
            continue
        layer, expert = int(match[1]), int(match[2])
        if not 0 <= expert < num_experts:
            raise ValueError(f"Expert index out of range: {name}")
        layers.setdefault(layer, [0] * num_experts)[expert] += info.nbytes
    if not layers:
        raise ValueError("No BitsMoE expert tensors found in checkpoint")
    return layers


def assign_experts(
    sizes: list[int], world_size: int, loads: list[float] | None = None
) -> list[int]:
    if world_size < 1 or not sizes or any(size < 0 for size in sizes):
        raise ValueError("Expected positive world size and nonnegative expert sizes")
    if loads is None:
        loads = [1.0] * len(sizes)
    if len(loads) != len(sizes) or any(not math.isfinite(x) or x < 0 for x in loads):
        raise ValueError("Expert loads must be finite, nonnegative, and match expert count")
    costs = [size * load for size, load in zip(sizes, loads)]
    memory_target = max(sum(sizes) / world_size, 1)
    cost_target = max(sum(costs) / world_size, 1)
    memory = [0] * world_size
    compute = [0.0] * world_size
    counts = [0] * world_size
    owners = [0] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda e: (-costs[e], -sizes[e], e))
    for expert in order:
        rank = min(
            range(world_size),
            key=lambda r: (
                max(
                    (memory[r] + sizes[expert]) / memory_target,
                    (compute[r] + costs[expert]) / cost_target,
                ),
                memory[r], counts[r], r,
            ),
        )
        owners[expert] = rank
        memory[rank] += sizes[expert]
        compute[rank] += costs[expert]
        counts[rank] += 1
    return owners


def make_plan(
    metadata: dict[str, TensorInfo], num_experts: int, world_size: int,
    loads: dict[str, list[float]] | None = None,
) -> dict:
    sizes = expert_sizes(metadata, num_experts)
    if loads is not None and set(loads) != {str(layer) for layer in sizes}:
        raise ValueError("Load statistics must cover exactly the checkpoint's MoE layers")
    return {
        "version": 1,
        "world_size": world_size,
        "num_experts": num_experts,
        "layers": {
            str(layer): assign_experts(values, world_size, None if loads is None else loads[str(layer)])
            for layer, values in sorted(sizes.items())
        },
    }


def validate_plan(plan: dict, layers: set[int], num_experts: int, world_size: int) -> None:
    if (plan.get("version"), plan.get("world_size"), plan.get("num_experts")) != (
        1, world_size, num_experts
    ):
        raise ValueError("Expert plan version, world size, or expert count does not match")
    if set(plan.get("layers", {})) != {str(layer) for layer in layers}:
        raise ValueError("Expert plan must cover exactly the model's MoE layers")
    for owners in plan["layers"].values():
        if len(owners) != num_experts or any(
            type(rank) is not int or not 0 <= rank < world_size for rank in owners
        ):
            raise ValueError("Invalid expert owner in placement plan")
