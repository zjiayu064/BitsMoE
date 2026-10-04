"""vLLM backend for BitsMoE checkpoints."""

ARCHITECTURES = {
    "qwen3_moe": "BitsMoEQwen3MoeForCausalLM",
    "deepseek_v2": "BitsMoEDeepseekV2ForCausalLM",
    "qwen3_next": "BitsMoEQwen3NextForCausalLM",
}
__all__ = ["LLM"]


def register() -> None:
    from vllm import ModelRegistry

    for model_type, architecture in ARCHITECTURES.items():
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(architecture, f"bitsmoe_vllm.models.{model_type}:{architecture}")


def __getattr__(name):
    if name == "LLM":
        from .api import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
