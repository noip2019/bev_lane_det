#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/lijishuo/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bevlanedet}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/tools/once_dinov2_small_origsplit_config.py}"
CHECKPOINT_PATH="${1:-${REPO_ROOT}/work_dirs/once_3dlanes_dinov2_small_origsplit/latest.pth}"
GPU_IDS="${GPU_IDS:-0}"
RATIO_TH="${RATIO_TH:-0.6}"
DIST_TH="${DIST_TH:-1.5}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage:
  bash test_once_dinov2_small_origsplit.sh [checkpoint_path]
EOF
    exit 0
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found under ${CONDA_BASE}" >&2
    exit 1
fi

CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH}")"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.pth}"
PRED_ROOT="${PRED_ROOT:-${REPO_ROOT}/work_dirs/once_3dlanes_dinov2_small_origsplit/predictions/${CHECKPOINT_STEM}}"

VAL_ARGS=(
    --config "${CONFIG_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --pred-root "${PRED_ROOT}"
    --ratio-th "${RATIO_TH}"
    --dist-th "${DIST_TH}"
)

if [[ -n "${GT_ROOT:-}" ]]; then
    VAL_ARGS+=(--gt-root "${GT_ROOT}")
fi

if [[ -n "${INDEX_FILE:-}" ]]; then
    VAL_ARGS+=(--index-file "${INDEX_FILE}")
fi

if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
    VAL_ARGS+=(--batch-size "${VAL_BATCH_SIZE}")
fi

if [[ -n "${VAL_NUM_WORKERS:-}" ]]; then
    VAL_ARGS+=(--num-workers "${VAL_NUM_WORKERS}")
fi

if [[ -n "${IMAGE_ROOTS:-}" ]]; then
    OLD_IFS="${IFS}"
    IFS=":"
    read -r -a IMAGE_ROOT_ARRAY <<< "${IMAGE_ROOTS}"
    IFS="${OLD_IFS}"
    for image_root in "${IMAGE_ROOT_ARRAY[@]}"; do
        if [[ -n "${image_root}" ]]; then
            VAL_ARGS+=(--image-root "${image_root}")
        fi
    done
fi

cd "${REPO_ROOT}"
mkdir -p "${MPLCONFIGDIR}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MPLCONFIGDIR

python tools/val_once_ratio.py "${VAL_ARGS[@]}"

echo "Finished. Predictions are under: ${PRED_ROOT}"
