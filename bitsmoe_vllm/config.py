from . import ARCHITECTURES


def validate_options(options) -> None:
    if not isinstance(options, dict) or options.keys() - {"expert_map"}:
        raise ValueError("bitsmoe_vllm accepts only expert_map")


def checkpoint_type(model, config):
    from transformers import PretrainedConfig

    values, _ = PretrainedConfig.get_config_dict(
        model, revision=config.get("revision"),
        cache_dir=config.get("download_dir"),
    )
    model_type = values.get("model_type")
    if model_type not in ARCHITECTURES or not values.get("bitsmoe"):
        raise ValueError(f"Unsupported BitsMoE checkpoint: {model_type!r}")
    overrides = config.get("hf_overrides", {})
    validate_options(overrides.get("bitsmoe_vllm", values.get("bitsmoe_vllm", {})))
    return model_type


def evaluation_args(config: dict, *, model: str | None = None) -> dict:
    """Apply BitsMoE compatibility settings without inference tuning defaults."""
    args = dict(config)
    if args.get("dtype", "auto") == "auto":
        args["dtype"] = "float16"
    overrides = dict(args.get("hf_overrides", {}))
    validate_options(overrides.get("bitsmoe_vllm", {}))
    model = model or args.get("pretrained") or args.get("model")
    model_type = checkpoint_type(model, args) if model else "qwen3_moe"
    overrides["architectures"] = [ARCHITECTURES[model_type]]
    if model_type == "qwen3_next":
        if args.get("enable_prefix_caching") or args.get("reuse_prefixes"):
            raise ValueError("Qwen3Next does not support prefix caching in vLLM 0.11.0")
        args["enable_prefix_caching"] = False
    args["hf_overrides"] = overrides
    args.setdefault("worker_cls", "bitsmoe_vllm.runtime.worker.BitsMoEWorker")
    if int(args.get("tensor_parallel_size", 1)) > 1:
        args.setdefault("enable_expert_parallel", True)
    # Packed prefill kernels are not intended for full graph replay.
    args.setdefault("compilation_config", {"cudagraph_mode": "FULL_DECODE_ONLY"})
    return args


def engine_args(config: dict, *, model: str | None = None) -> dict:
    args = evaluation_args(config, model=model)
    args.setdefault("gpu_memory_utilization", 0.85)
    args.setdefault("max_num_batched_tokens", 512)
    sequences = int(args.setdefault("max_num_seqs", 16))
    if sequences < 1:
        raise ValueError("max_num_seqs must be positive")
    captures = [2**i for i in range(sequences.bit_length()) if 2**i < sequences]
    captures.append(sequences)
    compilation = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    if args.get("cuda_graph_sizes") is None:
        compilation["cudagraph_capture_sizes"] = captures
    if "compilation_config" not in config:
        args["compilation_config"] = compilation
    return args
