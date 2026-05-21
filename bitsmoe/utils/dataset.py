"""Utilities for preparing merged calibration datasets for model training."""

import ast
import random
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from bitsmoe.utils.logger import setup_logger

__all__ = ["get_dataloader", "MultiDatasetLoader"]

logger = setup_logger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MultiDatasetLoader(Dataset):
    """Unified loader that merges multiple NLP datasets into a single Dataset."""

    STREAMING_DATASETS = {
        "allenai/c4",
        "narrativeqa",
        "Muennighoff/natural-instructions",
    }

    def __init__(
        self,
        tokenizer,
        dataset_dicts: Union[Dict, List[Dict]],
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.dataset_configs = (
            dataset_dicts if isinstance(dataset_dicts, list) else [dataset_dicts]
        )

        self.processed_data: List[Dict] = []
        for dataset_config in self.dataset_configs:
            while True:
                try:
                    raw_dataset = self._load_dataset(dataset_config, cache_dir)
                    break
                except:
                    logger.warning(f"Error loading dataset {dataset_config['name']}, retrying...")
                    continue
            processed_dataset = self._process_dataset(
                dataset_config, raw_dataset, sample_size
            )
            self.processed_data.extend(processed_dataset)

        logger.info(
            "Prepared %s samples from %s datasets.",
            len(self.processed_data),
            len(self.dataset_configs),
        )

    def _load_dataset(self, dataset_config: Dict, cache_dir: Optional[str]):
        """Load a single dataset based on its configuration.

        If ``data_files`` is specified in dataset_config, the dataset is loaded
        directly as JSON (bypassing any hub dataset script) with streaming=False,
        making sampling fully reproducible.  The path may be:
          - An absolute local path  (/data/c4/en/c4-train.00001-of-01024.json.gz)
          - A relative hub path     (en/c4-train.00001-of-01024.json.gz)
            → resolved to hf://datasets/<name>/<path>
          - A full URL / hf:// URI  (used as-is)

        Otherwise falls back to streaming for datasets in STREAMING_DATASETS.
        """
        data_files = dataset_config.get("data_files")
        name = dataset_config["name"]

        if data_files is not None:
            # Resolve relative hub paths to an hf:// URI so that datasets loads
            # only the specified shard, bypassing the custom dataset script.
            if not (data_files.startswith("/") or "://" in data_files):
                data_files = f"hf://datasets/{name}/{data_files}"
            try:
                dataset = load_dataset(
                    "json",
                    data_files=data_files,
                    split="train",
                    cache_dir=cache_dir,
                    streaming=False,
                )
                return dataset
            except Exception as exc:
                logger.error("Error loading dataset %s via data_files=%s: %s",
                             name, data_files, exc)
                raise

        streaming = name in self.STREAMING_DATASETS
        try:
            if dataset_config.get("config"):
                dataset = load_dataset(
                    name,
                    dataset_config["config"],
                    split=dataset_config["split"],
                    cache_dir=cache_dir,
                    streaming=streaming,
                )
            else:
                dataset = load_dataset(
                    name,
                    split=dataset_config["split"],
                    cache_dir=cache_dir,
                    streaming=streaming,
                )
            return dataset
        except Exception as exc:
            logger.error("Error loading dataset %s: %s", name, exc)
            raise

    def _process_dataset(
        self,
        dataset_config: Dict,
        raw_dataset,
        sample_size: Optional[int],
    ) -> List[Dict]:
        """Process datasets into model-ready samples."""
        processed: List[Dict] = []

        name = dataset_config["name"]
        task_type = dataset_config["task_type"]

        if sample_size is None:
            if isinstance(raw_dataset, IterableDataset):
                raise ValueError(
                    f"sample_size is required for streaming dataset {name}"
                )
            dataset_len = int(len(raw_dataset) * dataset_config["weight"])
        else:
            dataset_len = int(sample_size * dataset_config["weight"])

        dataset_len = max(dataset_len, 1)

        if isinstance(raw_dataset, IterableDataset):
            buffer_size = max(20 * dataset_len, 1)
            shuffled_ds = raw_dataset.shuffle(buffer_size=buffer_size, seed=self.seed)
            # Respect per-dataset weighted target size in streaming mode.
            take_count = dataset_len
            sampled_data = shuffled_ds.take(take_count)
        else:
            n = len(raw_dataset)
            sample_count = min(int(dataset_len * 2), n)
            rng = random.Random(self.seed)
            sampled_indices = rng.sample(range(n), sample_count)
            sampled_data = [raw_dataset[i] for i in sampled_indices]

        valid_len = 0
        for item in sampled_data:
            example = {"task_type": None, "input_ids": None}

            if name == "wikitext":
                text = item["text"].strip()
                if text:
                    tokenized_input = self._tokenize_text(text, add_template=False)
                    example.update(
                        {
                            "text": text,
                            "input_ids": tokenized_input["input_ids"],
                            "attention_mask": tokenized_input["attention_mask"],
                            "task_type": task_type,
                        }
                    )

            elif name == "allenai/c4":
                text = item["text"].strip()
                if text:
                    tokenized_input = self._tokenize_text(text, add_template=False)
                    example.update(
                        {
                            "text": text,
                            "input_ids": tokenized_input["input_ids"],
                            "attention_mask": tokenized_input["attention_mask"],
                            "task_type": task_type,
                        }
                    )

            elif name == "winogrande":
                sentence = item["sentence"].strip()
                full_text = f"{sentence}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "hellaswag":
                context = item["ctx"].strip()
                full_text = f"{context}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "piqa":
                context = item["goal"].strip()
                sol1 = item["sol1"].strip()
                sol2 = item["sol2"].strip()
                full_text = (
                    f"{context}.\n"
                    f"Options:\n"
                    f"1) {sol1}\n"
                    f"2) {sol2}\n"
                    f"Think step by step which makes sense and choose one."
                )
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "hotpotqa/hotpot_qa":
                question = item["question"]
                context = item["context"]

                context_text = ""
                for title, sentences in zip(context["title"], context["sentences"]):
                    context_text += f"Document: {title}\n"
                    for idx, sent in enumerate(sentences):
                        context_text += f"({idx}) {sent.strip()}\n"

                full_text = (
                    f"Question: {question}\n\n"
                    f"Use the information below to reason step by step and find the answer:\n"
                    f"{context_text}\n"
                    f"Explain your reasoning and provide the final answer at the end."
                )
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "gsm8k":
                question = item["question"].strip()
                full_text = f"{question}"
                tokenized_input = self._tokenize_text(full_text, add_template=False)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "math_qa":
                problem = item["Problem"].strip()
                full_text = f"{problem}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "EleutherAI/hendrycks_math":
                problem = item["problem"].strip()
                full_text = f"{problem}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "mbpp":
                problem = item["text"].strip()
                code = item["code"].strip()
                full_text = f"{problem}\n{code}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "codeparrot/apps":
                prompt = item["question"].strip()
                try:
                    code = ast.literal_eval(item["solutions"].strip())[0]
                    full_text = f"{prompt}\n{code}"
                    tokenized_input = self._tokenize_text(full_text)
                    example.update(
                        {
                            "text": full_text,
                            "input_ids": tokenized_input["input_ids"],
                            "attention_mask": tokenized_input["attention_mask"],
                            "task_type": task_type,
                        }
                    )
                except Exception:
                    pass

            elif name == "openai_humaneval":
                prompt = item["prompt"].strip()
                code = item["canonical_solution"].strip()
                full_text = f"{prompt}\n{code}"
                tokenized_input = self._tokenize_text(full_text, add_template=False)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "cais/mmlu":
                prompt = item["question"].strip()
                full_text = f"{prompt}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "allenai/ai2_arc":
                prompt = item["question"].strip()
                full_text = f"{prompt}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "Idavidrein/gpqa":
                question = item["Question"].strip()
                full_text = f"{question}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "squad_v2":
                title = item["title"].strip()
                context = item["context"].strip()
                question = item["question"].strip()
                full_text = (
                    f"Article title: {title}\n"
                    f"Context: {context}\n\n"
                    f"Question: {question}\n"
                    f"Think step by step and answer concisely."
                )
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "narrativeqa":
                title = item["document"]["summary"]["title"].strip()
                context = item["document"]["summary"]["text"].strip()
                question = item["question"]["text"].strip()
                full_text = (
                    f"Article title: {title}\n"
                    f"Context: {context}\n\n"
                    f"Question: {question}\n"
                    f"Think step by step and answer concisely."
                )
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "google/IFEval":
                prompt = item["prompt"].strip()
                full_text = f"Instruction: {prompt}"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            elif name == "Muennighoff/natural-instructions":
                definition = item["definition"].strip()
                inputs = item["inputs"].strip()
                targets = item["targets"].strip()
                full_text = f"{definition.strip()}\n\nInput: {inputs.strip()}\n{targets}\n"
                tokenized_input = self._tokenize_text(full_text)
                example.update(
                    {
                        "text": full_text,
                        "input_ids": tokenized_input["input_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                        "task_type": task_type,
                    }
                )

            else:
                logger.warning("Dataset %s is undefined in processing pipeline", name)
                continue

            if (
                example["input_ids"] is not None
                and example["input_ids"].shape[-1] > 10
            ):
                example["length"] = example["input_ids"].shape[0]
                processed.append(example)
                valid_len += 1
                if valid_len >= dataset_len:
                    break

        logger.info("Processed %s samples for dataset %s", valid_len, name)
        return processed

    def _tokenize_text(self, text: str, add_template: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize text and optionally apply chat templates."""
        if add_template:
            messages = [{"role": "user", "content": text}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_text = text

        encoded = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        }

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def get_dataloader(
    dataset_dict: Union[Dict, List[Dict]],
    tokenizer_name: str = "gpt2",
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 96,
    cache_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    trust_remote_code: bool = False,
    padding_side: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader that merges multiple datasets when provided."""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if padding_side is not None:
        if padding_side not in ("left", "right"):
            raise ValueError(f"Unsupported padding_side={padding_side!r}, expected 'left' or 'right'")
        tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_configs = dataset_dict if isinstance(dataset_dict, list) else [dataset_dict]

    dataset = MultiDatasetLoader(
        tokenizer=tokenizer,
        dataset_dicts=dataset_configs,
        max_length=max_length,
        cache_dir=cache_dir,
        sample_size=sample_size,
        seed=seed,
    )

    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    logger.info("DataLoader batch size set to %s", batch_size)

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g if shuffle else None,
    )
