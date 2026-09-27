# BitsMoE: Cost-Aware Bit Allocation in Spectral Space for MoE LLM Quantization

<p align="center">
  <img src="assets/bitsmoe-logo.png" alt="BitsMoE Logo" width="250">
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2606.00079"><img src="https://img.shields.io/badge/arXiv-2606.00079-b31b1b.svg" alt="Paper"></a>
    <a href="https://huggingface.co/zjiayu064"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Models"></a>
    <a href="https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html"><img src="https://img.shields.io/badge/Paper%20License-arXiv%20non--exclusive%201.0-lightgrey.svg" alt="Paper License"></a>
    <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/Code%20License-MIT-blue.svg" alt="Code License"></a>
</p>

<p align="center">
  <b><a href="#overview">🔍 Overview</a></b> •
  <b><a href="#preparation">🧩 Preparation</a></b> •
  <b><a href="#environment">🛠️ Environment</a></b> •
  <b><a href="#models">📦 Models</a></b> •
  <b><a href="#evaluation-config">⚙️ Evaluation Config</a></b> •
  <b><a href="#citation">📌 Citation</a></b>
</p>

---

<a id="overview"></a>

## 🔍 Overview

**BitsMoE** is a cost-aware mixed-precision quantization framework for Mixture-of-Experts (MoE) LLMs. Assigning one bit-width to an entire expert or linear block overlooks differences among the directions within it. Shared-basis Spectral Decomposition (SSD) separates a basis shared across experts from expert-specific spectral components, which serve as fine-grained quantization units. The shared basis is quantized to 8-bit precision, while the expert-specific spectral directions receive component-wise bit-widths.

Factorized Quantization Cost Modeling (FQCM) estimates each component's cost from its intrinsic spectral importance, activation-dependent importance, and bit-width-dependent distortion. BitsMoE uses these costs in a global integer linear program (ILP) to assign bit-widths under a fixed memory budget, then groups and packs the quantized directions for GPU inference.

<p align="center">
  <img src="assets/BitsMoE.png" alt="BitsMoE Framework" width="100%">
</p>
<p align="center"><i>Figure 1. Overview of the BitsMoE pipeline: shared-basis SVD decomposition, ILP-based mixed-precision quantization, and efficient inference.</i></p>

### ✨ Key Contributions

- **🧠 Shared-basis Spectral Decomposition (SSD).** SSD captures structure shared across experts and exposes expert-specific spectral components as fine-grained quantization units.
- **🎯 Factorized Quantization Cost Modeling (FQCM).** FQCM approximates component-wise quantization costs from expected routing-weighted output reconstruction loss by combining intrinsic spectral importance, activation-dependent importance, and bit-width-dependent distortion.
- **📊 Global bit allocation.** An ILP assigns a bit-width to each spectral component to minimize total modeled quantization cost under a fixed memory budget.
- **⚡ Measured accuracy and efficiency.** In the paper's 2-bit Qwen3-30B-A3B evaluation, BitsMoE reaches 64.29% average accuracy across seven tasks, 2.80 percentage points above GEMQ. It achieves 16.47× faster end-to-end offline quantization than GEMQ and up to 6.46× the decode throughput of GPTQ under the reported benchmark settings.

<a id="preparation"></a>

## Preparation

```bash
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh
```

<a id="environment-compatibility"></a>

## Environment Compatibility

- **CUDA Toolkit 12.8** (`nvcc -V` should show `release 12.8`)
- **GCC / G++ ≥ 12.0**
- Tested and recommended: **GCC/G++ 14.2.0**
- **Python 3.12** is recommended
- Other compiler/CUDA/Python combinations are currently not tested

<a id="environment"></a>

## Environment

Choose **one** method below.

### One-command install (**Recommended**):

```bash
bash scripts/install_env.sh
```

### Manual install:

```bash
conda create -n bitsmoe python=3.12 -y
conda activate bitsmoe
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install packaging ninja psutil
uv pip install vllm==0.11.0
uv pip install flash-attn --no-build-isolation
uv pip install flash-linear-attention --no-build-isolation
uv pip install --no-binary=causal-conv1d "git+https://github.com/Dao-AILab/causal-conv1d.git" --no-build-isolation
cd bitsmoe/evaluation/lm_eval
uv pip install -e .
cd ../../..
uv pip install -e . --no-build-isolation
```

<a id="quick-start"></a>

## Quick Start

By default, models are downloaded from Hugging Face (Transformers cache).

Run one inference demo (PPL):

```bash
bitsmoe demo
```

Run evaluation from YAML config:

```bash
bitsmoe eval --config configs/deepseekv2/eval.yaml
bitsmoe eval --config configs/qwen3moe/eval.yaml
bitsmoe eval --config configs/qwen3next/eval.yaml
```

<a id="models"></a>

## 📦 Models


| Base Model | Precision | Hugging Face Repo | Access Link |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-V2-Lite | 2-bit | `zjiayu064/DeepSeek-V2-Lite-BitsMoE-2bit` | [🤗 DeepSeek-V2-Lite-BitsMoE-2bit](https://huggingface.co/zjiayu064/DeepSeek-V2-Lite-BitsMoE-2bit) |
| Qwen/Qwen3-30B-A3B-Base | 2-bit | `zjiayu064/Qwen3-30B-A3B-Base-BitsMoE-2bit` | [🤗 Qwen3-30B-A3B-Base-BitsMoE-2bit](https://huggingface.co/zjiayu064/Qwen3-30B-A3B-Base-BitsMoE-2bit) |
| Qwen/Qwen3-Next-80B-A3B-Instruct | 2-bit | `zjiayu064/Qwen3-Next-80B-A3B-Instruct-BitsMoE-2bit` | [🤗 Qwen3-Next-80B-A3B-Instruct-BitsMoE-2bit](https://huggingface.co/zjiayu064/Qwen3-Next-80B-A3B-Instruct-BitsMoE-2bit) |

<a id="evaluation-config"></a>

## Evaluation Config (`eval.yaml`)

The main runtime configuration is in:

- `configs/deepseekv2/eval.yaml`
- `configs/qwen3moe/eval.yaml`
- `configs/qwen3next/eval.yaml`

These fields follow the usage of [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

- `lm_eval.model`: backend type (for example `hf`)
- `lm_eval.model_args`: model constructor arguments (for example `pretrained`, `dtype`)
- `lm_eval.tasks`: benchmark task list
- `lm_eval.apply_chat_template`: whether to apply chat template
- `lm_eval.batch_size`: batch size or `auto`
- `lm_eval.extra_args`: extra `lm-eval` flags
- `ppl`: perplexity evaluation settings
- `runtime.seed`: random seed

Current tasks:

- `mmlu`
- `hellaswag`
- `winogrande`
- `openbookqa`
- `mathqa`

<a id="quantization"></a>

## Quantization

This repository is currently **inference-only**.

- Released now: runtime kernels, model patching logic, and 2-bit checkpoints for evaluation/inference.
- Not released yet: end-to-end quantization/preprocessing pipeline.

Planned open-source items:

- Quantization configs and reproducible command examples
- Calibration/data preparation workflow
- Validation report (quality and efficiency deltas)

Status: **Coming soon**.

## License

- Code repository: MIT
- Paper (arXiv): arXiv.org perpetual, non-exclusive license 1.0

<a id="citation"></a>

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{zhao2026bitsmoe,
      title={{BitsMoE}: Efficient Spectral Energy-Guided Bit Allocation for {MoE} {LLM} Quantization}, 
      author={Jiayu Zhao and Zihan Teng and Minhao Fan and Tianrui Ma and Wentao Ren and Song Chen and Weichen Liu},
      year={2026},
      eprint={2606.00079},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.00079}
}
```
