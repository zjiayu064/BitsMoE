import copy
import importlib
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation import GenerationMixin

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from bitsmoe.models.qwen3moe.bitsmoe_mlp import BitsMoE_Qwen3MoeMLP
from bitsmoe.models.qwen3moe.bitsmoe_block import BitsMoE_Qwen3MoeSparseMoeBlock
from bitsmoe.models.qwen3next.bitsmoe_mlp import BitsMoE_Qwen3NextMLP
from bitsmoe.models.qwen3next.bitsmoe_block import BitsMoE_Qwen3NextSparseMoeBlock
from bitsmoe.models.deepseekv2.bitsmoe_mlp import BitsMoE_DeepSeekMLP
from bitsmoe.models.deepseekv2.bitsmoe_block import BitsMoE_DeepSeekSparseMoeBlock
from bitsmoe.models.base_mlp import (
    BitsMoE_BaseMoeMLP,
    advance_bitsmoe_load_progress,
    configure_bitsmoe_load_progress,
    finalize_bitsmoe_load_progress,
)
from bitsmoe.utils.expert_utils import extract_expert_weights
from bitsmoe.utils.logger import setup_logger

logger = setup_logger(__name__)


def _progress(iterable, desc: str, total: int = None):
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
    return iterable


_SUPER_EXPERTS_BY_MODEL: Dict[str, Dict[int, List[int]]] = {
    "qwen3moe": {1: [68], 2: [92], 3: [82]},
    "deepseekv2": {2: [54], 3: [38]},
    "qwen3next": {10: [288, 401], 21: [251], 25: [381], 26: [264], 35: [38, 69, 303]},
}

_BITSMOE_BLOCK_BY_MODEL = {
    "qwen3moe": BitsMoE_Qwen3MoeSparseMoeBlock,
    "deepseekv2": BitsMoE_DeepSeekSparseMoeBlock,
    "qwen3next": BitsMoE_Qwen3NextSparseMoeBlock,
}

_BITSMOE_MLP_BY_MODEL = {
    "qwen3moe": BitsMoE_Qwen3MoeMLP,
    "deepseekv2": BitsMoE_DeepSeekMLP,
    "qwen3next": BitsMoE_Qwen3NextMLP,
}

_OLD_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained
_PATCHED = False
_PATCHED_CLASS_CACHE: Dict[Tuple[type, str], type] = {}
_DIRECT_GEN_MIXIN_CACHE: Dict[Tuple[type, str, str], type] = {}
_BITSMOE_RUNTIME_BUFFER_SUFFIXES = tuple(
    f".{key}" for key in BitsMoE_BaseMoeMLP._runtime_buffer_leaf_keys()
)


def _ensure_direct_generation_mixin(cls: Type, name: Optional[str] = None, module: Optional[str] = None) -> Type:
    if any(base is GenerationMixin for base in cls.__bases__):
        return cls

    cls_name = name or cls.__name__
    cls_module = module or cls.__module__
    cache_key = (cls, cls_name, cls_module)
    cached = _DIRECT_GEN_MIXIN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    patched_cls = type(
        cls_name,
        (cls, GenerationMixin),
        {
            "__module__": cls_module,
            "__qualname__": cls_name,
        },
    )
    _DIRECT_GEN_MIXIN_CACHE[cache_key] = patched_cls
    return patched_cls


def _get_moe_attr_name(layer) -> Optional[str]:
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return "mlp"
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
        return "block_sparse_moe"
    return None


def _expert_intermediate_size(expert_module) -> int:
    w = extract_expert_weights(expert_module)
    return int(w["gate_proj"].shape[0])


def _resolve_source_experts(source_moe, config) -> List[Optional[Any]]:
    source_experts = list(getattr(source_moe, "experts", []))
    if source_experts:
        return source_experts

    num_experts = int(getattr(source_moe, "num_experts", getattr(config, "num_experts", 0)))
    return [None] * max(0, num_experts)


def _resolve_layer_expert_intermediate_size(config, source_experts: List[Optional[Any]]) -> int:
    for expert_module in source_experts:
        if expert_module is not None:
            return _expert_intermediate_size(expert_module)

    for attr_name in ("moe_intermediate_size", "intermediate_size"):
        val = getattr(config, attr_name, None)
        if val is not None:
            return int(val)

    raise RuntimeError(
        "Failed to infer MoE expert intermediate_size: no expert template found and "
        "config has neither `moe_intermediate_size` nor `intermediate_size`."
    )


def _build_dense_expert_fallback(source_moe, source_experts: List[Optional[Any]], config, intermediate_size: int):
    for expert_module in source_experts:
        if expert_module is not None:
            return copy.deepcopy(expert_module)

    shared_expert = getattr(source_moe, "shared_expert", None)
    if shared_expert is None:
        raise RuntimeError(
            "Cannot build dense super expert: no source expert template and no shared_expert fallback."
        )

    shared_cls = shared_expert.__class__
    try:
        return shared_cls(config, intermediate_size=intermediate_size)
    except TypeError:
        return shared_cls(config)


def _make_qwen3next_fast_sparse_moe_init(module, mlp_cls):
    def _fast_init(self, config):
        module.nn.Module.__init__(self)
        self.num_experts = int(getattr(config, "num_experts", 0))
        self.top_k = int(getattr(config, "num_experts_per_tok", 0))
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))

        self.gate = module.nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = module.nn.ModuleList([None] * self.num_experts)

        shared_intermediate = getattr(config, "shared_expert_intermediate_size", None)
        if shared_intermediate is None:
            shared_intermediate = getattr(config, "intermediate_size")
        shared_intermediate = int(shared_intermediate)
        self.shared_expert = mlp_cls(config, intermediate_size=shared_intermediate)
        self.shared_expert_gate = module.torch.nn.Linear(config.hidden_size, 1, bias=False)

    return _fast_init


@contextmanager
def _maybe_enable_qwen3next_fast_init(enable: bool):
    if not enable:
        yield
        return

    patched = []
    candidates = [
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextSparseMoeBlock", "Qwen3NextMLP"),
    ]

    for module_name, block_cls_name, mlp_cls_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        block_cls = getattr(module, block_cls_name, None)
        mlp_cls = getattr(module, mlp_cls_name, None)
        if block_cls is None or mlp_cls is None:
            continue

        old_init = block_cls.__init__
        block_cls.__init__ = _make_qwen3next_fast_sparse_moe_init(module, mlp_cls)
        patched.append((block_cls, old_init))

    if patched:
        logger.info(
            "Enabled qwen3next fast-init for %d SparseMoeBlock class(es).",
            len(patched),
        )

    try:
        yield
    finally:
        for block_cls, old_init in patched:
            block_cls.__init__ = old_init


@contextmanager
def _maybe_enable_meta_param_progress_hook(enable: bool):
    if not enable:
        yield
        return

    try:
        from transformers import modeling_utils as _mu
    except Exception:
        yield
        return

    old_loader = getattr(_mu, "_load_parameter_into_model", None)
    if old_loader is None:
        yield
        return

    def _wrapped_load_parameter_into_model(model, param_name: str, tensor):
        old_loader(model, param_name, tensor)
        if (
            (".mlp.experts." in param_name or ".block_sparse_moe.experts." in param_name)
            and param_name.endswith(".gate_payload_buffer")
        ):
            advance_bitsmoe_load_progress(1, source="external")

    _mu._load_parameter_into_model = _wrapped_load_parameter_into_model
    try:
        yield
    finally:
        _mu._load_parameter_into_model = old_loader


@contextmanager
def _maybe_enable_hf_key_precheck_fastpath(enable: bool):
    if not enable:
        yield
        return

    try:
        from transformers import modeling_utils as _mu
    except Exception:
        yield
        return

    old_find_missing = getattr(_mu, "_find_missing_and_unexpected_keys", None)
    old_find_mismatched = getattr(_mu, "_find_mismatched_keys", None)
    if not callable(old_find_missing) or not callable(old_find_mismatched):
        yield
        return

    def _wrapped_find_missing_and_unexpected_keys(*args, **kwargs):
        logger.info("HF preload stage: computing missing/unexpected keys ...")
        t0 = time.perf_counter()
        missing_keys, unexpected_keys = old_find_missing(*args, **kwargs)
        dt = time.perf_counter() - t0

        # For qwen3next with deferred expert runtime buffers, these keys are
        # intentionally absent from model.state_dict() pre-load, but they are
        # still valid load targets via module.load_state_dict(assign=True).
        unexpected_before = len(unexpected_keys)
        if unexpected_before > 0:
            unexpected_keys = [k for k in unexpected_keys if not _is_bitsmoe_runtime_expert_key(k)]
        skipped_runtime_unexpected = unexpected_before - len(unexpected_keys)

        missing_fast = _FastContainsList(missing_keys)
        unexpected_fast = _FastContainsList(unexpected_keys)
        logger.info(
            "HF preload stage done: missing=%d unexpected=%d skipped_runtime_unexpected=%d elapsed=%.2fs",
            len(missing_fast),
            len(unexpected_fast),
            skipped_runtime_unexpected,
            dt,
        )
        return missing_fast, unexpected_fast

    def _wrapped_find_mismatched_keys(*args, **kwargs):
        t0 = time.perf_counter()
        mismatched_keys, mismatched_shapes = old_find_mismatched(*args, **kwargs)
        dt = time.perf_counter() - t0

        mismatched_fast = _FastContainsList(mismatched_keys)
        logger.info(
            "HF preload mismatch check: mismatched=%d elapsed=%.2fs",
            len(mismatched_fast),
            dt,
        )
        return mismatched_fast, mismatched_shapes

    _mu._find_missing_and_unexpected_keys = _wrapped_find_missing_and_unexpected_keys
    _mu._find_mismatched_keys = _wrapped_find_mismatched_keys
    try:
        yield
    finally:
        _mu._find_missing_and_unexpected_keys = old_find_missing
        _mu._find_mismatched_keys = old_find_mismatched


def _infer_local_config_tags(pretrained_model_name_or_path) -> Dict[str, Any]:
    if not isinstance(pretrained_model_name_or_path, str):
        return {}
    if not os.path.isdir(pretrained_model_name_or_path):
        return {}
    cfg_path = os.path.join(pretrained_model_name_or_path, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _should_show_bitsmoe_load_progress() -> bool:
    v = str(os.environ.get("BITSMOE_PATCH_LOAD_PROGRESS", "1")).strip().lower()
    return v not in {"0", "false", "no", "off"}


def _log_checkpoint_index_stats(pretrained_model_name_or_path) -> None:
    if not isinstance(pretrained_model_name_or_path, str):
        return
    if not os.path.isdir(pretrained_model_name_or_path):
        return

    index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return

        keys = list(weight_map.keys())
        total = len(keys)
        expert_payload = 0
        first_expert_payload_idx = -1
        for i, key in enumerate(keys):
            if ".experts." in key and key.endswith(".gate_payload_buffer"):
                expert_payload += 1
                if first_expert_payload_idx < 0:
                    first_expert_payload_idx = i

        logger.info(
            "BitsMoE checkpoint index stats: total_keys=%d expert_gate_payload_keys=%d first_expert_gate_payload_idx=%d",
            total,
            expert_payload,
            first_expert_payload_idx,
        )
    except Exception as exc:
        logger.warning("Failed to inspect checkpoint index %s: %s", index_path, exc)


class _FastContainsList(list):
    __slots__ = ("_set",)

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._set = set(self)

    def __contains__(self, item):
        return item in self._set


def _is_bitsmoe_runtime_expert_key(key: str) -> bool:
    if ".experts." not in key:
        return False
    return any(key.endswith(suffix) for suffix in _BITSMOE_RUNTIME_BUFFER_SUFFIXES)


def _env_flag(name: str, default: bool) -> bool:
    default_raw = "1" if default else "0"
    v = str(os.environ.get(name, default_raw)).strip().lower()
    return v not in {"0", "false", "no", "off"}


def _maybe_prepare_bitsmoe_runtime(model, bitsmoe_model_type: str) -> None:
    if not _env_flag("BITSMOE_PREPARE_RUNTIME", True):
        return

    # Keep backward compatibility with older qwen3moe-specific switches.
    if bitsmoe_model_type == "qwen3moe" and not _env_flag("BITSMOE_QWEN3MOE_PREPARE_RUNTIME", True):
        return

    warmup = _env_flag("BITSMOE_PREPARE_WARMUP", True)
    if bitsmoe_model_type == "qwen3moe":
        warmup = _env_flag("BITSMOE_QWEN3MOE_PREPARE_WARMUP", warmup)
    prepared = 0
    failed = 0

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return

    for layer in layers:
        moe_attr = _get_moe_attr_name(layer)
        if moe_attr is None:
            continue
        block = getattr(layer, moe_attr, None)
        prepare_fn = getattr(block, "prepare_runtime_fastpath", None)
        if not callable(prepare_fn):
            continue
        try:
            prepare_fn(warmup=warmup)
            prepared += 1
        except Exception as exc:  # best-effort, don't block model load
            failed += 1
            logger.warning(
                "BitsMoE runtime prepare failed: model_type=%s layer_attr=%s err=%s",
                bitsmoe_model_type,
                moe_attr,
                exc,
            )

    logger.info(
        "BitsMoE runtime prepare done: model_type=%s prepared=%d failed=%d warmup=%s",
        bitsmoe_model_type,
        prepared,
        failed,
        warmup,
    )


def _patch_bitsmoe_inplace(model, bitsmoe_model_type: str) -> None:
    if bitsmoe_model_type not in _BITSMOE_BLOCK_BY_MODEL:
        raise ValueError(
            f"Unsupported bitsmoe_model_type={bitsmoe_model_type}. "
            f"Supported={list(_BITSMOE_BLOCK_BY_MODEL.keys())}"
        )

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("BitsMoE model patch requires model.model.layers")

    block_cls = _BITSMOE_BLOCK_BY_MODEL[bitsmoe_model_type]
    mlp_cls = _BITSMOE_MLP_BY_MODEL[bitsmoe_model_type]
    super_map = _SUPER_EXPERTS_BY_MODEL.get(bitsmoe_model_type, {})
    packed_expert_total = 0

    num_layers = len(model.model.layers)
    logger.info("BitsMoE class patch: rewriting %d layers ...", num_layers)
    for layer_idx, layer in _progress(
        enumerate(model.model.layers),
        desc="BitsMoE class patch",
        total=num_layers,
    ):
        moe_attr = _get_moe_attr_name(layer)
        if moe_attr is None:
            continue

        source_moe = getattr(layer, moe_attr)
        source_experts = _resolve_source_experts(source_moe, model.config)
        expert_intermediate_size = _resolve_layer_expert_intermediate_size(
            model.config,
            source_experts,
        )
        super_experts = sorted(set(int(x) for x in super_map.get(layer_idx, [])))
        super_set = set(super_experts)
        defer_runtime_buffer_init = bitsmoe_model_type == "qwen3next"

        bitsmoe_block = block_cls(
            config=model.config,
            layer_idx=layer_idx,
            source_block=source_moe,
            super_experts=super_experts,
            copy_source_experts=False,
        )

        for expert_idx, expert_module in enumerate(source_experts):
            if expert_idx in super_set:
                dense_expert = (
                    copy.deepcopy(expert_module)
                    if expert_module is not None
                    else _build_dense_expert_fallback(
                        source_moe=source_moe,
                        source_experts=source_experts,
                        config=model.config,
                        intermediate_size=expert_intermediate_size,
                    )
                )
                bitsmoe_block.set_expert(expert_idx, dense_expert, packed=False)
                continue

            if expert_module is None and not defer_runtime_buffer_init:
                bitsmoe_block.set_expert(expert_idx, None, packed=False)
                continue

            packed_kwargs = dict(
                config=model.config,
                intermediate_size=expert_intermediate_size,
            )
            if defer_runtime_buffer_init:
                packed_kwargs["defer_runtime_buffer_init"] = True
            packed_mlp = mlp_cls(**packed_kwargs)
            bitsmoe_block.set_expert(expert_idx, packed_mlp, packed=True)
            packed_expert_total += 1

        setattr(layer, moe_attr, bitsmoe_block)

    setattr(model, "_bitsmoe_packed_expert_total", int(packed_expert_total))
    logger.info("BitsMoE class patch: packed experts=%d", packed_expert_total)


def _resolve_base_causallm_class(
    config,
    pretrained_model_name_or_path,
    trust_remote_code: bool,
) -> Type:
    model_type = str(getattr(config, "model_type", "")).lower()

    # Prefer direct auto_map for custom models (e.g., deepseek_v2 local modeling file).
    if trust_remote_code and isinstance(pretrained_model_name_or_path, str):
        auto_map = getattr(config, "auto_map", None)
        if isinstance(auto_map, dict):
            class_ref = auto_map.get("AutoModelForCausalLM", None)
            if isinstance(class_ref, str):
                from transformers.dynamic_module_utils import get_class_from_dynamic_module

                resolved_cls = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path)
                if model_type in {"deepseek_v2", "deepseekv2"}:
                    resolved_cls = _ensure_direct_generation_mixin(
                        resolved_cls,
                        name=resolved_cls.__name__,
                        module=resolved_cls.__module__,
                    )
                return resolved_cls

    # Standard HF registry path.
    from transformers.models.auto.auto_factory import _get_model_class

    resolved_cls = _get_model_class(config, AutoModelForCausalLM._model_mapping)
    if model_type in {"deepseek_v2", "deepseekv2"}:
        resolved_cls = _ensure_direct_generation_mixin(
            resolved_cls,
            name=resolved_cls.__name__,
            module=resolved_cls.__module__,
        )
    return resolved_cls


def _build_patched_model_class(base_cls: Type, bitsmoe_model_type: str) -> Type:
    cache_key = (base_cls, bitsmoe_model_type)
    if cache_key in _PATCHED_CLASS_CACHE:
        return _PATCHED_CLASS_CACHE[cache_key]

    class BitsMoEPatchedModel(base_cls):
        _bitsmoe_model_type = bitsmoe_model_type
        _bitsmoe_external_progress_updates = False

        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            _patch_bitsmoe_inplace(self, self._bitsmoe_model_type)
            configure_bitsmoe_load_progress(
                total_experts=getattr(self, "_bitsmoe_packed_expert_total", 0),
                enabled=_should_show_bitsmoe_load_progress(),
                external_updates=bool(self._bitsmoe_external_progress_updates),
            )

        def get_parameter_or_buffer(self, target: str):
            try:
                return super().get_parameter_or_buffer(target)
            except AttributeError as exc:
                if self._bitsmoe_model_type != "qwen3next":
                    raise
                if not _is_bitsmoe_runtime_expert_key(target):
                    raise

                try:
                    module_path, leaf = target.rsplit(".", 1)
                    module = self.get_submodule(module_path)
                except Exception:
                    raise exc

                if not isinstance(module, BitsMoE_BaseMoeMLP):
                    raise exc

                # leaf: e.g. gate_payload_buffer -> payload_buffer
                base_leaf = leaf.split("_", 1)[1] if "_" in leaf else leaf
                dtype = module._RUNTIME_BUFFER_DTYPES.get(base_leaf)
                if dtype is None:
                    raise exc

                init_tensor = (
                    torch.tensor(0, dtype=dtype)
                    if base_leaf in {"groupsize", "original_rank"}
                    else torch.empty(0, dtype=dtype)
                )
                module.register_buffer(leaf, init_tensor, persistent=True)
                return module._buffers[leaf]

    BitsMoEPatchedModel.__name__ = f"BitsMoEPatched{base_cls.__name__}_{bitsmoe_model_type}"
    BitsMoEPatchedModel.__qualname__ = BitsMoEPatchedModel.__name__
    BitsMoEPatchedModel.__module__ = base_cls.__module__
    if bitsmoe_model_type == "deepseekv2":
        BitsMoEPatchedModel = _ensure_direct_generation_mixin(
            BitsMoEPatchedModel,
            name=BitsMoEPatchedModel.__name__,
            module=BitsMoEPatchedModel.__module__,
        )
    _PATCHED_CLASS_CACHE[cache_key] = BitsMoEPatchedModel
    return BitsMoEPatchedModel


def _bitsmoe_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    explicit_bitsmoe = kwargs.pop("bitsmoe", None)
    explicit_bitsmoe_model_type = kwargs.pop("bitsmoe_model_type", None)

    if "trust_remote_code" not in kwargs:
        cfg_tags = _infer_local_config_tags(pretrained_model_name_or_path)
        local_model_type = str(cfg_tags.get("model_type", ""))
        local_bitsmoe_model_type = str(cfg_tags.get("bitsmoe_model_type", ""))
        if local_model_type == "deepseek_v2" or local_bitsmoe_model_type == "deepseekv2":
            kwargs["trust_remote_code"] = True

    config = kwargs.get("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
        )
    else:
        # Align behavior with HF no-config path:
        # consume config-overridable kwargs before model __init__ to avoid
        # unexpected keyword errors (e.g. use_cache passed to __init__).
        for k in list(kwargs.keys()):
            if k == "config":
                continue
            if hasattr(config, k):
                setattr(config, k, kwargs.pop(k))

    if explicit_bitsmoe is None:
        is_bitsmoe = bool(getattr(config, "bitsmoe", False))
    else:
        is_bitsmoe = bool(explicit_bitsmoe)

    if not is_bitsmoe:
        logger.info("Loading base model from %s ...", pretrained_model_name_or_path)
        model = _OLD_FROM_PRETRAINED(pretrained_model_name_or_path, *args, **kwargs)
        logger.info("Base model loaded")
        return model

    bitsmoe_model_type = (
        explicit_bitsmoe_model_type
        or getattr(config, "bitsmoe_model_type", None)
        or getattr(config, "model_type", None)
    )
    bitsmoe_model_type = str(bitsmoe_model_type)
    if bitsmoe_model_type not in _BITSMOE_BLOCK_BY_MODEL:
        raise ValueError(
            f"Unsupported bitsmoe_model_type={bitsmoe_model_type}. "
            f"Supported={list(_BITSMOE_BLOCK_BY_MODEL.keys())}"
        )

    base_cls = _resolve_base_causallm_class(
        config=config,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
    )
    patched_cls = _build_patched_model_class(base_cls, bitsmoe_model_type)
    use_meta_param_hook = bool(kwargs.get("device_map", None) is not None or kwargs.get("low_cpu_mem_usage", False))
    if bitsmoe_model_type == "qwen3next" and not _env_flag("BITSMOE_QWEN3NEXT_META_PROGRESS_HOOK", True):
        use_meta_param_hook = False
    patched_cls._bitsmoe_external_progress_updates = use_meta_param_hook
    enable_hf_key_fastpath = bool(
        bitsmoe_model_type == "qwen3next" and _env_flag("BITSMOE_QWEN3NEXT_HF_KEY_PRECHECK_FASTPATH", True)
    )

    logger.info(
        "Loading BitsMoE model from %s with patched class %s ...",
        pretrained_model_name_or_path,
        patched_cls.__name__,
    )
    if bitsmoe_model_type == "qwen3next":
        _log_checkpoint_index_stats(pretrained_model_name_or_path)
    # Do not force-inject `config` when we only loaded it for routing/type
    # detection; let HF from_pretrained consume kwargs as usual.
    with (
        _maybe_enable_qwen3next_fast_init(bitsmoe_model_type == "qwen3next"),
        _maybe_enable_meta_param_progress_hook(use_meta_param_hook),
        _maybe_enable_hf_key_precheck_fastpath(enable_hf_key_fastpath),
    ):
        try:
            model = patched_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        finally:
            finalize_bitsmoe_load_progress()
    _maybe_prepare_bitsmoe_runtime(model, bitsmoe_model_type)
    logger.info("BitsMoE model loaded")
    return model


def ensure_auto_patch_installed() -> None:
    global _PATCHED
    if _PATCHED:
        return
    AutoModelForCausalLM.from_pretrained = classmethod(_bitsmoe_from_pretrained)
    _PATCHED = True
    logger.info("Installed BitsMoE AutoModelForCausalLM.from_pretrained patch")


# Install on import so regular AutoModelForCausalLM.from_pretrained auto-detects bitsmoe config.
ensure_auto_patch_installed()


@contextmanager
def patch_quant_config_from_lm_cfg(lm_cfg: Dict[str, Any]):
    """
    Temporarily inject or override `quantization_config` into HF config.

    This helper is kept for evaluation compatibility.
    """

    override_qcfg = lm_cfg.get("quantization_config", None)
    if not isinstance(override_qcfg, dict):
        yield
        return

    original_from_pretrained = AutoConfig.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        out = original_from_pretrained(*args, **kwargs)

        if isinstance(out, tuple):
            config, extra_kwargs = out
        else:
            config, extra_kwargs = out, None

        base_qcfg = {}
        if hasattr(config, "quantization_config") and isinstance(config.quantization_config, dict):
            base_qcfg = dict(config.quantization_config)

        base_qcfg.update(override_qcfg)
        config.quantization_config = base_qcfg

        if extra_kwargs is not None:
            return config, extra_kwargs
        return config

    AutoConfig.from_pretrained = patched_from_pretrained

    try:
        yield
    finally:
        AutoConfig.from_pretrained = original_from_pretrained
