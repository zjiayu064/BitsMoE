import os
import bitsmoe


def get_project_root() -> str:
    """
    Return the absolute path to the project root directory.

    This function infers the root directory based on the installed
    or locally editable `bitsmoe` package location.
    It is robust to both editable (`pip install -e .`) and standard
    installation modes.

    Returns:
        str: Absolute path to the project root (e.g., /home/user/LLM/BitsMoE).
    """
    return os.path.dirname(os.path.dirname(bitsmoe.__file__))


PROJECT_ROOT = get_project_root()


def get_abs_path(relative_path: str) -> str:
    """
    Convert a relative path (with respect to the project root)
    into an absolute filesystem path.

    Args:
        relative_path (str): Relative path inside the project,
            e.g., "bitsmoe/configs/run_moe_statistics.yaml".

    Returns:
        str: The absolute path corresponding to the given relative path.
    """
    return os.path.join(PROJECT_ROOT, relative_path)
