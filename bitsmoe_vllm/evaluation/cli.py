import argparse
import os
import sys
from pathlib import Path

from .. import register
from ..config import evaluation_args as model_args


def prepare_harness() -> None:
    from bitsmoe.evaluation.harness import prepare_harness as prepare_lm_eval

    # Evaluation runs vLLM as a library, so its CLI's spawn default does not apply. Avoid inheriting CUDA/runtime state from the evaluation process.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prepare_lm_eval()
    register()
    from . import harness  # noqa: F401


def main(argv: list[str] | None = None) -> None:
    import yaml
    from bitsmoe.evaluation.cli import build_lm_eval_argv

    parser = argparse.ArgumentParser(description="Run the bundled lm-eval harness with BitsMoE vLLM")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model_path")
    options, extra = parser.parse_known_args(argv)
    config = yaml.safe_load(options.config.read_text())["lm_eval"]
    config["backend"] = "vllm"
    sys.argv = build_lm_eval_argv(config, options.model_path) + extra
    prepare_harness()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
