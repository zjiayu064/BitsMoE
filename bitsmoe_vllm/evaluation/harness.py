"""BitsMoE continuation scoring for the bundled evaluation harness."""

from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.models.vllm_causallms import VLLM

from ..runtime.scoring import score_requests


@register_model("bitsmoe-vllm")
class BitsMoEVLLM(VLLM):
    def __init__(self, *args, reuse_prefixes=False, freeze_startup_gc=False, **kwargs):
        self.reuse_prefixes = reuse_prefixes
        if reuse_prefixes:
            kwargs.setdefault("enable_prefix_caching", True)
            kwargs.setdefault("scheduler_cls", "bitsmoe_vllm.runtime.scheduler.BitsMoEScheduler")
        kwargs.setdefault("worker_cls", "bitsmoe_vllm.runtime.worker.BitsMoEWorker")
        super().__init__(*args, **kwargs)
        if freeze_startup_gc:
            import gc

            gc.collect()
            gc.freeze()

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        ordered = Collator(requests, sort_fn=lambda r: (-len(r[1] + r[2]), tuple(r[1] + r[2])))
        batches = ordered.get_batched(n=0 if self.batch_size == "auto" else int(self.batch_size),
                                      batch_fn=None)
        answers = []
        maximum = self.max_length - 1
        with tqdm(total=len(requests), disable=disable_tqdm, desc="Running loglikelihood requests") as progress:
            for batch in batches:
                inputs, starts = [], []
                for _, context, continuation in batch:
                    tokens = (context + continuation)[-maximum:]
                    start = len(tokens) - len(continuation)
                    if start < 1:
                        raise ValueError("Continuation needs at least one context token within max_length")
                    inputs.append(tokens)
                    starts.append(start)
                outputs = score_requests(self.model, inputs, starts,
                                         reuse_prefixes=getattr(self, "reuse_prefixes", False))
                for output, start, tokens, (key, _, _) in zip(outputs, starts, inputs, batch, strict=True):
                    answer = self._parse_logprobs(tokens, output, start)
                    answers.append(answer)
                    if key is not None:
                        self.cache_hook.add_partial("loglikelihood", key, answer)
                    progress.update(1)
        return ordered.get_original(answers)
