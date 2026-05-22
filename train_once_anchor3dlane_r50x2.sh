#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/lijishuo/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bevlanedet}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/tools/once_anchor3dlane_r50x2_config.py}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
CHECKPOINT_PATH=""

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage:
  bash train_once_anchor3dlane_r50x2.sh [checkpoint_path] [extra train args...]

Examples:
  bash train_once_anchor3dlane_r50x2.sh
  bash train_once_anchor3dlane_r50x2.sh /path/to/latest.pth
  GPU_IDS=0,1,2,3 NPROC_PER_NODE=4 bash train_once_anchor3dlane_r50x2.sh /path/to/latest.pth --epochs 80
EOF
    exit 0
fi

if [[ $# -gt 0 && "${1}" != -* ]]; then
    CHECKPOINT_PATH="$1"
    shift
fi

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found under ${CONDA_BASE}" >&2
    exit 1
fi

if [[ -n "${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

TRAIN_ARGS=(
    --config "${CONFIG_PATH}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
    TRAIN_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
fi

torchrun --standalone --monitor_interval "${MONITOR_INTERVAL}" --nproc_per_node="${NPROC_PER_NODE}" tools/train_once.py "${TRAIN_ARGS[@]}" "$@"
