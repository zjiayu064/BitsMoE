#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-bitsmoe}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not found." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

# 1) PyTorch CUDA 12.8
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 2) Build helpers
uv pip install packaging ninja psutil

# 3) CUDA runtime-related deps (keep no-build-isolation for compatibility)
uv pip install flash-attn --no-build-isolation
uv pip install flash-linear-attention --no-build-isolation
uv pip install --no-binary=causal-conv1d \
  "git+https://github.com/Dao-AILab/causal-conv1d.git" \
  --no-build-isolation

# 4) lm_eval submodule
pushd bitsmoe/evaluation/lm_eval >/dev/null
uv pip install -e .
popd >/dev/null

# 5) BitsMoE-arxiv itself
uv pip install -e . --no-build-isolation

echo "Environment ready. Activate with: conda activate ${ENV_NAME}"
