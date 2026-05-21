import logging
from typing import Optional
import os
import sys
from collections import Counter
from tqdm import tqdm

from bitsmoe.utils.get_path import get_abs_path

class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler compatible with tqdm progress bars.
    All logs are routed via tqdm.write to avoid corrupting the bar.
    """

    def __init__(self, level=logging.NOTSET, stream=None):
        super().__init__(level)
        self.stream = stream or sys.stderr
        self.is_tty = self.stream.isatty()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self.is_tty:
                tqdm.write(msg)
            else:
                self.stream.write(msg + "\n")
                self.stream.flush()
        except Exception:
            self.handleError(record)

def is_main_process():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank == 0 or local_rank == 0


def setup_logger(name: Optional[str] = None, file_name: Optional[str] = None):
    """
    Create or get a logger that is:
      - tqdm-safe (will not break progress bars)
      - supports both console and file logging
      - main-process only (for distributed setups)
      - backward compatible with the original interface
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers and is_main_process():
        # -------- tqdm-safe console handler --------
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] "
            "[%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # -------- optional file handler (unchanged semantics) --------
        if file_name is not None:
            file_name = get_abs_path(file_name)

            file_handler = logging.FileHandler(file_name, mode="w")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def hijack_lm_eval_only(level: int = logging.INFO) -> None:
    """
    Hijack ALL lm-eval loggers, including both:
      - official: lm_eval.*
      - vendored: bitsmoe.evaluation.lm_eval.lm_eval.*

    Does NOT affect other libraries.
    """

    bitsmoe_logger = setup_logger("bitsmoe.evaluation.lm_eval")

    if not bitsmoe_logger.handlers:
        raise RuntimeError("setup_logger returned logger without handlers.")

    bitsmoe_handlers = bitsmoe_logger.handlers

    def _patch(logger: logging.Logger):
        logger.handlers.clear()
        logger.setLevel(level)
        logger.propagate = False
        for h in bitsmoe_handlers:
            logger.addHandler(h)

    # 1. Always patch canonical root
    _patch(logging.getLogger("lm_eval"))

    # 2. Patch vendored root (your project-local lm-eval)
    _patch(logging.getLogger("bitsmoe.evaluation.lm_eval."))

    # 3. Auto-discover ALL sub-loggers under both trees
    for name, obj in logging.root.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue

        if (
            name.startswith("lm_eval.")
            or name.startswith("bitsmoe.evaluation.lm_eval.")
        ):
            _patch(obj)


def hijack_gptq_only(level: int = logging.INFO) -> None:
    """
    Hijack GPTQ-related and accelerator loggers and redirect them
    to the project logger.

    Covered logger trees:
      - auto_gptq.*
      - gptqmodel.*
      - transformers.quantization.gptq.*
      - optimum.gptq.*
      - accelerate.*
      - tokenicer.*

    This function also safely takes over the root logger stream handlers
    to guarantee full log unification.
    """

    bitsmoe_logger = setup_logger("bitsmoe.baselines.gptq")

    if not bitsmoe_logger.handlers:
        raise RuntimeError("setup_logger returned logger without handlers.")

    bitsmoe_handlers = list(bitsmoe_logger.handlers)

    # ---- Safe root takeover (only stream handlers) ----
    root = logging.getLogger()

    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            root.removeHandler(h)

    for h in bitsmoe_handlers:
        if h not in root.handlers:
            root.addHandler(h)

    root.setLevel(level)
    root.propagate = False

    def _patch(logger: logging.Logger) -> None:
        if logger is bitsmoe_logger:
            return

        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler):
                logger.removeHandler(h)

        logger.setLevel(level)
        logger.propagate = False

        for h in bitsmoe_handlers:
            if h not in logger.handlers:
                logger.addHandler(h)

    # Explicit roots to patch
    root_names = (
        "auto_gptq",
        "gptqmodel",
        "gptqmodel.env",
        "gptqmodel.utils",
        "gptqmodel.utils.perplexity",
        "transformers.quantization.gptq",
        "optimum.gptq",
        "accelerate",
        "accelerate.utils",
        "accelerate.utils.modeling",
        "tokenicer",
        "tokenicer.tokenicer",
    )

    for name in root_names:
        _patch(logging.getLogger(name))

    # Auto-discover all sub-loggers
    for name, obj in logging.root.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue

        if (
            name.startswith("auto_gptq.")
            or name.startswith("gptqmodel.")
            or name.startswith("transformers.quantization.gptq.")
            or name.startswith("optimum.gptq.")
            or name.startswith("accelerate.")
            or name.startswith("tokenicer.")
        ):
            _patch(obj)


def _format_skipped(skipped: list[tuple[int, int]]) -> str:
    """
    Public interface remains unchanged.
    Only returns a compact engineering summary for logging.
    """
    summary = _summarize_skipped(skipped)

    lines = [
        f"",
        f"  - Total skipped     : {summary['total_skipped']}",
        f"  - Affected layers   : {summary['affected_layers']}",
        f"  - Max skipped layer : L{summary['max_layer']} ({summary['max_layer_count']} experts)",
    ]
    return "\n".join(lines)


def _summarize_skipped(skipped: list[tuple[int, int]]) -> dict:
    """
    Compute high-level statistics for skipped experts.
    """

    if not skipped:
        return {
            "total_skipped": 0,
            "affected_layers": 0,
            "max_layer": -1,
            "max_layer_count": 0,
            "skip_ratio": 0.0,
        }

    layer_counter = Counter(layer_idx for layer_idx, _ in skipped)

    total_skipped = len(skipped)
    affected_layers = len(layer_counter)

    max_layer, max_layer_count = max(
        layer_counter.items(), key=lambda x: x[1]
    )

    return {
        "total_skipped": total_skipped,
        "affected_layers": affected_layers,
        "max_layer": max_layer,
        "max_layer_count": max_layer_count,
    }
