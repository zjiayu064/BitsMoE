import os
import math
import random
import json
from typing import Dict, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.quantizers.auto import AutoQuantizationConfig
from datasets import load_dataset

from bitsmoe.utils.logger import setup_logger
from bitsmoe.utils.transformers_compat import patch_transformers_cache_compat


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger(__name__)


def parse_model_args(model_args_str: Optional[str]):
    if model_args_str is None:
        return {}

    model_args = {}
    for kv in model_args_str.split(","):
        key, val = kv.split("=", 1)

        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass

        model_args[key] = val

    return model_args


class IndexDataset(Dataset):
    """Tensor-backed dataset for language model evaluation."""
    def __init__(self, tensors: torch.Tensor):
        self.tensors = tensors
        # For HF-recommended strided PPL.
        self.encodings_input_ids: Optional[torch.Tensor] = None

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return self.tensors.size(0)


def get_wikitext2(max_samples=1024):
    data = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    # Keep original order for comparable PPL.
    n = min(max_samples, len(data))
    return [data[i] for i in range(n)]


def get_c4(max_samples=1024):
    data = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
    )
    # Do NOT shuffle for reproducibility.
    return list(data.take(max_samples))


def process_data(samples, tokenizer, seq_len: int, field_name: str):
    """
    Prepare token blocks (kept for backward compatibility) and store full token stream
    for HF-recommended strided sliding-window PPL.
    """
    # Hard reset fast tokenizer internal state.
    if hasattr(tokenizer, "_tokenizer"):
        try:
            tokenizer._tokenizer.no_truncation()
            tokenizer._tokenizer.no_padding()
        except Exception:
            pass

    texts = []
    for item in samples:
        text = item.get(field_name, "")
        if isinstance(text, str):
            text = text.strip()
            if text:
                texts.append(text)

    if len(texts) == 0:
        empty = torch.empty((0, seq_len), dtype=torch.long)
        return IndexDataset(empty)

    # HF guide uses "\n\n" between documents.
    all_text = "\n\n".join(texts)

    # No truncation/padding; follow HF guide behavior.
    encodings = tokenizer(
        all_text,
        return_tensors="pt",
        truncation=False,
        padding=False,
        verbose=False,
    )
    token_ids_1d = encodings.input_ids[0]  # [N]

    # Keep your original fixed blocks (may be used elsewhere).
    num_blocks = token_ids_1d.numel() // seq_len
    token_ids_2d = token_ids_1d[: num_blocks * seq_len].contiguous().view(num_blocks, seq_len)

    ds = IndexDataset(token_ids_2d)
    ds.encodings_input_ids = encodings.input_ids  # [1, N]
    return ds


def get_dataset(name: str, tokenizer, seq_len=2048, max_samples=10240):
    if name == "wikitext2":
        data = get_wikitext2(max_samples)
        return process_data(data, tokenizer, seq_len, "text")

    if name == "c4":
        while True:
            try:
                data = get_c4(max_samples)
                break
            except:
                continue
        return process_data(data, tokenizer, seq_len, "text")

    raise ValueError(f"Unknown dataset: {name}")


@torch.no_grad()
def compute_ppl_hf_strided(
    model,
    encodings_input_ids: torch.Tensor,
    test_name: str,
    max_length: int,
    stride: int = 512,
):
    """
    HF-recommended strided sliding-window PPL.
    Only tokens newly introduced by each stride contribute to the loss.
    """
    model.eval()

    # Clamp to model context limit if present.
    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max is None:
        model_max = getattr(model.config, "n_positions", None)
    if model_max is not None:
        max_length = min(max_length, int(model_max))

    stride = max(1, min(stride, max_length))
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    seq_len = encodings_input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    pbar = tqdm(
        range(0, seq_len, stride),
        desc=f"Evaluate {test_name}... PPL=NA",
        dynamic_ncols=True,
    )
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may differ on last step

        input_ids = encodings_input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context tokens

        # target_ids: [1, seq_len]
        if (target_ids != -100).sum() == 0:
            continue

        outputs = model(input_ids)
        logits = outputs.logits  # [B, T, V]

        # HF causal LM loss: shift by 1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        vocab = shift_logits.size(-1)

        # Mask out-of-vocab labels (robust for Mixtral/Mistral v0.1)
        oob = (shift_labels != -100) & (
            (shift_labels < 0) | (shift_labels >= vocab)
        )
        if oob.any():
            shift_labels[oob] = -100
            logger.warning("Warning: out-of-vocab labels ignored")

        # Cross-entropy over valid tokens
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        num_loss_tokens = (shift_labels != -100).sum().item()
        if num_loss_tokens > 0:
            nll_sum += loss_sum.item()
            n_tokens += num_loss_tokens
            running_ppl = math.exp(nll_sum / n_tokens)
            pbar.set_description(f"Evaluate {test_name}... PPL={running_ppl:.4f}")

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / max(1, n_tokens)
    final_ppl = math.exp(avg_nll)
    pbar.set_description(f"Evaluate {test_name}... PPL={final_ppl:.4f}")
    pbar.refresh()
    pbar.close()
    return final_ppl


def evaluate_ppl_all(
    model,
    tokenizer,
    seq_len=2048,
    max_samples=4096,
    stride=512
) -> Dict[str, float]:
    """Public API for evaluating PPL on WikiText2 and C4."""
    results = {}
    task_list = ["wikitext2", "c4"]
    dataset_dict = {}

    for name in task_list:
        dataset_dict[name] = get_dataset(
            name=name,
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_samples=max_samples,
        )

    # HF-recommended PPL for both tasks.
    for name in task_list:
        ds = dataset_dict[name]
        if ds.encodings_input_ids is None:
            raise RuntimeError(f"{name}: missing encodings_input_ids")
        results[name] = compute_ppl_hf_strided(
            model=model,
            encodings_input_ids=ds.encodings_input_ids,
            test_name=name,
            max_length=seq_len,
            stride=min(stride, seq_len),
        )
        logger.info(f"{name}: PPL={results[name]:.2f}")

    return results


def run_ppl_cli(
    model_name: str,
    use_fa2: bool = False,
    trust_remote_code: bool = False,
    seq_len: int = 2048,
    max_samples: int = 10240,
    stride: int = 512,
    dtype: str = "float16",
    model_args: Optional[dict] = None,
):
    """CLI-compatible entry for PPL evaluation."""
    if model_args is None:
        model_args = {}

    patch_transformers_cache_compat()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # fix_mistral_regex=True
    )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if (
        quantization_config := model_args.get("quantization_config", None)
    ) is not None and isinstance(quantization_config, dict):
        quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
        _ = model_args.pop("quantization_config", None)

        try:
            qc_dict = quantization_config.to_dict()
        except Exception:
            qc_dict = vars(quantization_config)

        logger.info(
            "Use quantization config:\n%s",
            json.dumps(qc_dict, indent=2, ensure_ascii=False),
        )

    model_kwargs = dict(
        dtype=dtype_map[dtype],
        device_map="auto",
        use_cache=False,
        trust_remote_code=trust_remote_code,
        **model_args,
    )

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if use_fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.eval()

    results = evaluate_ppl_all(
        model=model,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_samples,
        stride=stride,
    )

    logger.info("Final PPL Summary")
    logger.info(f"{'wikitext2':<10}: {results['wikitext2']:>8.2f}")
    logger.info(f"{'c4':<10}: {results['c4']:>8.2f}")

    return results
