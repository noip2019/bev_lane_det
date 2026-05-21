#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/lijishuo/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bevlanedet}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/tools/once_sequence_split_config.py}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found under ${CONDA_BASE}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" tools/train_once.py --config "${CONFIG_PATH}" "$@"
