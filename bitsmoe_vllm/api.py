from vllm import LLM as VLLM

from . import register
from .config import engine_args


class LLM(VLLM):
    """vLLM inference with BitsMoE checkpoint loading and packed experts."""

    def __init__(self, model: str, **kwargs):
        register()
        super().__init__(model=model, **engine_args(kwargs, model=model))
