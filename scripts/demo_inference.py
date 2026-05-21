import argparse
from typing import List, Optional

from bitsmoe.evaluation.ppl_eval import run_ppl_cli


def _default_model_path() -> str:
    return "zjiayu064/Qwen3-30B-A3B-Base-BitsMoE-2bit"


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="BitsMoE inference-only demo (PPL).")
    parser.add_argument(
        "--model_path",
        type=str,
        default=_default_model_path(),
        help="Model ID or local checkpoint path (default: Hugging Face model ID).",
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fa2", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    run_ppl_cli(
        model_name=args.model_path,
        use_fa2=bool(args.use_fa2),
        trust_remote_code=bool(args.trust_remote_code),
        seq_len=int(args.seq_len),
        max_samples=int(args.max_samples),
        stride=int(args.stride),
        dtype=str(args.dtype),
        model_args={},
    )


if __name__ == "__main__":
    main()
