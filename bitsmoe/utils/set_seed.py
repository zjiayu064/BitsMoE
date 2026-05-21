import random
import numpy as np
import torch

from bitsmoe.utils.logger import setup_logger

logger = setup_logger(__name__)

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    Works for Python, NumPy, PyTorch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")
