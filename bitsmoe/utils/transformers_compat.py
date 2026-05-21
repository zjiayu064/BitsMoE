from __future__ import annotations

from bitsmoe.utils.logger import setup_logger


LOGGER = setup_logger(__name__)


def patch_transformers_cache_compat() -> None:
    """
    Bridge cache API differences across transformers versions.

    Older model implementations (including many trust_remote_code checkpoints)
    may access:
      - cache.seen_tokens
      - cache.get_max_length()
      - cache.get_usable_length(...)

    Newer transformers cache classes expose:
      - cache.get_seq_length()
      - cache.get_max_cache_shape()
    """
    try:
        from transformers.cache_utils import Cache
    except Exception as exc:
        LOGGER.warning("Skip cache compatibility patch: cannot import Cache (%s)", exc)
        return

    if not hasattr(Cache, "seen_tokens"):
        def _get_seen_tokens(self) -> int:
            if hasattr(self, "_seen_tokens_compat"):
                try:
                    return int(self._seen_tokens_compat)
                except Exception:
                    pass
            try:
                return int(self.get_seq_length())
            except Exception:
                return 0

        def _set_seen_tokens(self, value) -> None:
            # Keep backward-compatible mutability for code that sets this field.
            try:
                self._seen_tokens_compat = int(value)
            except Exception:
                self._seen_tokens_compat = value

        Cache.seen_tokens = property(_get_seen_tokens, _set_seen_tokens)
        LOGGER.info("Patched transformers Cache.seen_tokens compatibility property.")

    if not hasattr(Cache, "get_max_length"):
        def _get_max_length(self, *args, **kwargs):
            try:
                max_shape = self.get_max_cache_shape(*args, **kwargs)
            except TypeError:
                max_shape = self.get_max_cache_shape()
            except Exception:
                return None

            if max_shape is None:
                return None
            try:
                max_shape = int(max_shape)
            except Exception:
                return None
            return None if max_shape < 0 else max_shape

        Cache.get_max_length = _get_max_length
        LOGGER.info("Patched transformers Cache.get_max_length compatibility method.")

    if not hasattr(Cache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length: int = 0, *args, **kwargs) -> int:
            # Legacy signature: get_usable_length(new_seq_length, layer_idx=None)
            layer_idx = kwargs.get("layer_idx", None)
            if layer_idx is None and len(args) > 0:
                layer_idx = args[0]

            # Current cache length
            try:
                if layer_idx is None:
                    prev_len = int(self.get_seq_length())
                else:
                    try:
                        prev_len = int(self.get_seq_length(layer_idx))
                    except TypeError:
                        prev_len = int(self.get_seq_length())
            except Exception:
                prev_len = 0

            # Maximum cache length (None for unbounded dynamic cache)
            try:
                if layer_idx is None:
                    max_len = self.get_max_length()
                else:
                    try:
                        max_len = self.get_max_length(layer_idx)
                    except TypeError:
                        max_len = self.get_max_length()
            except Exception:
                max_len = None

            if max_len is None:
                return prev_len
            try:
                max_len = int(max_len)
            except Exception:
                return prev_len
            if max_len < 0:
                return prev_len

            try:
                new_seq_length = int(new_seq_length)
            except Exception:
                new_seq_length = 0

            if prev_len + new_seq_length > max_len:
                return max(max_len - new_seq_length, 0)
            return prev_len

        Cache.get_usable_length = _get_usable_length
        LOGGER.info("Patched transformers Cache.get_usable_length compatibility method.")
