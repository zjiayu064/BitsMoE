import argparse
import json
import sys
from pathlib import Path

from . import register
from .config import engine_args
from .placement import expert_sizes, make_plan, read_checkpoint_metadata


def serve_args(args: list[str]) -> list[str]:
    args = list(args)
    if any(arg in ("-h", "--help") or arg.startswith("--help=") for arg in args):
        return args
    if len(args) < 2 or args[1].startswith("-"):
        raise ValueError("Usage: bitsmoe-vllm serve MODEL [options]")
    positions = {arg.split("=", 1)[0].lstrip("-").replace("-", "_"): i
                 for i, arg in enumerate(args) if arg.startswith("--")}
    config = {}
    for name in ("dtype", "hf_overrides", "tensor_parallel_size", "max_num_seqs",
                 "revision", "download_dir"):
        if name not in positions:
            continue
        i = positions[name]
        if "=" not in args[i] and (i + 1 == len(args) or args[i + 1].startswith("--")):
            raise ValueError(f"{args[i]} requires a value")
        value = args[i].split("=", 1)[1] if "=" in args[i] else args[i + 1]
        config[name] = json.loads(value) if name == "hf_overrides" else value
    for name in ("enable_prefix_caching", "enable_expert_parallel"):
        flags = [(positions[key], enabled) for key, enabled in
                 ((name, True), (f"no_{name}", False)) if key in positions]
        if flags:
            config[name] = max(flags)[1]
    for name, value in engine_args(config, model=args[1]).items():
        if (name in positions or f"no_{name}" in positions) and name not in ("dtype", "hf_overrides"):
            continue
        if name == "compilation_config" and "cuda_graph_sizes" in positions:
            value = {key: item for key, item in value.items() if key != "cudagraph_capture_sizes"}
        encoded = json.dumps(value) if isinstance(value, dict) else str(value)
        flag = "--" + name.replace("_", "-")
        if name in positions:
            i = positions[name]
            if "=" in args[i]:
                args[i] = f"{flag}={encoded}"
            else:
                args[i + 1] = encoded
        elif isinstance(value, bool):
            args.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            args.extend([flag, encoded])
    return args


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "eval":
        from .evaluation.cli import main as eval_main

        eval_main(args[1:])
        return
    if args and args[0] == "serve":
        if any(arg in ("-h", "--help") for arg in args):
            parser = argparse.ArgumentParser(
                prog="bitsmoe-vllm serve",
                description="Serve a BitsMoE checkpoint with an OpenAI-compatible API.",
                epilog="Additional arguments are passed to vLLM. Use --help=all for its complete option list.",
            )
            parser.add_argument("model", help="Local checkpoint path or Hugging Face model ID")
            parser.add_argument("--tensor-parallel-size", type=int, help="Number of GPUs")
            parser.add_argument("--gpu-memory-utilization", type=float, help="GPU memory fraction")
            parser.add_argument("--max-model-len", type=int, help="Maximum context length")
            parser.add_argument("--host", help="Server bind address")
            parser.add_argument("--port", type=int, help="Server port")
            parser.print_help()
            return
        from vllm.entrypoints.cli.main import main as vllm_main

        register()
        try:
            args = serve_args(args)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        sys.argv = ["vllm", *args]
        vllm_main()
        return
    parser = argparse.ArgumentParser(prog="bitsmoe-vllm")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Serve a BitsMoE model using vLLM arguments")
    commands.add_parser("eval", help="Run lm-eval using a BitsMoE YAML config")
    plan_parser = commands.add_parser("plan", help="Plan expert placement from a local checkpoint")
    plan_parser.add_argument("model")
    plan_parser.add_argument("--tp-size", type=int, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--loads", type=Path, help="JSON with per-layer expert load arrays")
    options = parser.parse_args(args)
    config = json.loads((Path(options.model) / "config.json").read_text())
    metadata = read_checkpoint_metadata(options.model)
    loads = json.loads(options.loads.read_text())["layers"] if options.loads else None
    num_experts = config.get("num_experts", config.get("n_routed_experts"))
    plan = make_plan(metadata, num_experts, options.tp_size, loads)
    options.output.write_text(json.dumps(plan, indent=2) + "\n")
    sizes = expert_sizes(metadata, num_experts)
    totals = [0] * options.tp_size
    for layer, owners in plan["layers"].items():
        for expert, rank in enumerate(owners):
            totals[rank] += sizes[int(layer)][expert]
    print(json.dumps({"expert_weight_gib": [round(x / 2**30, 3) for x in totals],
                      "plan": str(options.output)}, indent=2))


if __name__ == "__main__":
    main()
