"""Stream a short BitsMoE generation example."""

import argparse
from typing import List, Optional


DEFAULT_MODEL = "zjiayu064/Qwen3-30B-A3B-Base-BitsMoE-2bit"
DEFAULT_QUESTION = (
    "Why can a Mixture-of-Experts model use a lot of GPU memory "
    "even though each token activates only a few experts?"
)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Stream a BitsMoE text-generation demo.")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL,
        help="Model ID or local checkpoint path (default: Qwen3-30B-A3B BitsMoE checkpoint).",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fa2", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    if not args.question.strip():
        parser.error("--question must not be empty")
    return args


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    from bitsmoe.utils.transformers_compat import patch_transformers_cache_compat

    patch_transformers_cache_compat()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model_kwargs = {
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        "use_cache": True,
    }
    if args.use_fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = True
    if model.generation_config is not None:
        model.generation_config.use_cache = True
    model.eval()

    prompt = f"Q: {args.question.strip()}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "streamer": streamer,
    }
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    print(f"Q: {args.question.strip()}\nA: ", end="", flush=True)
    with torch.no_grad():
        model.generate(**inputs, **generation_kwargs)


if __name__ == "__main__":
    main()
