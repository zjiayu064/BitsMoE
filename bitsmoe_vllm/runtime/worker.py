"""Worker extensions for continuation scoring."""

from types import MethodType

from vllm.v1.worker.gpu_worker import Worker

from ..compat import validate_scoring_runner
from .scoring import prompt_logprobs


class BitsMoEWorker(Worker):
    def init_device(self):
        super().init_device()
        runner = self.model_runner
        validate_scoring_runner(runner)
        runner._bitsmoe_full_prompt_logprobs = runner._get_prompt_logprobs_dict
        runner._get_prompt_logprobs_dict = MethodType(prompt_logprobs, runner)
