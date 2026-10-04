"""vLLM scheduler with bounded prefix reuse for continuation scoring."""

from types import MethodType

from vllm.v1.core.sched.scheduler import Scheduler

from .prefix_cache import get_computed_blocks


class BitsMoEScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manager = self.kv_cache_manager
        manager._bitsmoe_original_get_computed_blocks = manager.get_computed_blocks
        manager.get_computed_blocks = MethodType(get_computed_blocks, manager)
