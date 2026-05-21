import sys
import argparse
import yaml
import json

from bitsmoe.evaluation.ppl_eval import run_ppl_cli
from bitsmoe.utils.set_seed import set_seed
from bitsmoe.utils.logger import hijack_lm_eval_only, setup_logger
from bitsmoe.utils.transformers_compat import patch_transformers_cache_compat
from bitsmoe.models.auto.bitsmoe_auto_patch import patch_quant_config_from_lm_cfg

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model, overwrites model_name_or_path / model_args.pretrained in all eval backends",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="(lm_eval only) Override lm_eval.tasks, e.g. --tasks gsm8k humaneval",
    )
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(lm_eval only) Override apply_chat_template from yaml config",
    )
    return parser.parse_args()


def build_lm_eval_argv(
    cfg: dict,
    model_path: str,
    apply_chat_template_override=None,
) -> list:
    logger = setup_logger(__name__)
    model_args = dict(cfg.get("model_args", {}))

    if model_path is not None:
        model_args["pretrained"] = model_path
        logger.warning(f"Model Path is overwritten to {model_path}")

    apply_chat_template = cfg.get("apply_chat_template", False)
    if apply_chat_template_override is not None:
        logger.warning(
            f"apply_chat_template overridden from yaml ({apply_chat_template}) "
            f"to {apply_chat_template_override}"
        )
        apply_chat_template = apply_chat_template_override

    argv = [
        sys.argv[0],
        "--model", cfg.get("model", "hf"),
        "--model_args", json.dumps(
            model_args,
            ensure_ascii=False
        ),
        "--device", cfg.get("device", "cuda"),
        "--batch_size", str(cfg.get("batch_size", "auto:32")),
        "--tasks", ",".join(cfg["tasks"]),
    ]

    if apply_chat_template:
        argv.append("--apply_chat_template")

    extra_args = cfg.get("extra_args", {})
    for k, v in extra_args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            # if False, skip the flag entirely
        else:
            argv.extend([f"--{k}", str(v)])

    logger.info(
        "lm_eval command:\n  lm_eval %s",
        " \\\n    ".join(argv[1:])
    )
    return argv


def main():
    logger = setup_logger(__name__)
    
    args = parse_args()
    cfg = load_yaml(args.config)
    model_path = args.model_path

    # Ensure compatibility between older model code (trust_remote_code)
    # and newer transformers cache APIs.
    patch_transformers_cache_compat()

    # lm_eval-only overrides
    if args.tasks is not None:
        cfg.setdefault("lm_eval", {})
        cfg["lm_eval"]["tasks"] = args.tasks
    # Set runtime parameters
    runtime_cfg = cfg.get("runtime", {})
    if "seed" in runtime_cfg:
        set_seed(runtime_cfg["seed"])

    task_count = 0
    
    # lm-eval
    lm_cfg = cfg.get("lm_eval", {})
    if lm_cfg.get("enable", False):
        task_count += 1
        logger.info(
            "\n%s\n%s\n%s",
            "=" * 70,
            f"{task_count}. LM evaluation enabled. Running lm-eval benchmarks...",
            "=" * 70,
        )
        lm_eval_argv = build_lm_eval_argv(
            lm_cfg,
            model_path,
            apply_chat_template_override=args.apply_chat_template,
        )
        sys.argv = lm_eval_argv
        hijack_lm_eval_only()
        from bitsmoe.evaluation.lm_eval.lm_eval.__main__ import cli_evaluate

        with patch_quant_config_from_lm_cfg(lm_cfg):
            cli_evaluate()

    # ppl eval
    ppl_cfg = cfg.get("ppl", {})
    if ppl_cfg.get("enable", False):
        task_count += 1
        logger.info(
            "\n%s\n%s\n%s",
            "=" * 70,
            f"{task_count}. PPL evaluation enabled. Running ppl benchmarks...",
            "=" * 70,
        )
        
        model_args = ppl_cfg.get("model_args", {})
        if model_path is not None:
            model_name = model_path
            logger.warning(f"Model Path is overwritten to {model_name}")
        else:
            model_name = ppl_cfg['model_name_or_path']
            
        run_ppl_cli(
            model_name=model_name,
            use_fa2=ppl_cfg['use_fa2'],
            trust_remote_code=ppl_cfg['trust_remote_code'],
            seq_len=ppl_cfg["seq_len"],
            max_samples=ppl_cfg["max_samples"],
            stride=ppl_cfg["stride"],
            dtype=ppl_cfg["dtype"],
            model_args=model_args
        )

if __name__ == "__main__":
    main()
