"""Resolve vLLM integration capabilities outside the inference path."""

from importlib import import_module
from importlib.metadata import version
from inspect import signature


def custom_op_registration():
    for module in ("vllm.utils.torch_utils", "vllm.utils"):
        try:
            register = getattr(import_module(module), "direct_register_custom_op")
        except (ImportError, AttributeError):
            continue
        return register
    raise RuntimeError(f"vLLM {version('vllm')} lacks native custom-op registration")


def validate_scoring_runner(runner):
    required = ("_get_prompt_logprobs_dict", "_sync_device", "input_batch",
                "query_start_loc", "requests", "sampler")
    missing = [name for name in required if not hasattr(runner, name)]
    if not missing:
        batch = runner.input_batch
        missing.extend(f"input_batch.{name}" for name in (
            "num_prompt_logprobs", "in_progress_prompt_logprobs_cpu", "req_id_to_index"
        ) if not hasattr(batch, name))
        missing.extend(f"sampler.{name}" for name in (
            "compute_logprobs", "gather_logprobs"
        ) if not callable(getattr(runner.sampler, name, None)))
        if not hasattr(runner.query_start_loc, "np"):
            missing.append("query_start_loc.np")
    if missing:
        raise RuntimeError(
            f"BitsMoE scoring is incompatible with vLLM {version('vllm')}: "
            f"missing {', '.join(missing)}"
        )
    try:
        signature(runner._get_prompt_logprobs_dict).bind(None, {})
        signature(runner._sync_device).bind()
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"BitsMoE scoring is incompatible with vLLM {version('vllm')}: "
            "unsupported prompt-logprobs runner methods"
        ) from error
