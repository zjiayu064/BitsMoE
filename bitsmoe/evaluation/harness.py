"""Locate the evaluation harness in a checkout or installed environment."""

from importlib.util import find_spec
from pathlib import Path
import sys


def prepare_harness() -> None:
    bundled = Path(__file__).resolve().parent / "lm_eval"
    if (bundled / "lm_eval" / "__init__.py").is_file():
        location = str(bundled)
        if location not in sys.path:
            sys.path.insert(0, location)
    elif find_spec("lm_eval") is None:
        raise RuntimeError("Install lm-eval or initialize the lm-eval submodule before running evaluation")
