#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/lijishuo/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bevlanedet}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/tools/once_sequence_split_config.py}"
CHECKPOINT_PATH="${1:-${REPO_ROOT}/work_dirs/once_3dlanes_sequence_split/latest.pth}"
GPU_IDS="${GPU_IDS:-0}"
RATIO_TH="${RATIO_TH:-0.6}"
DIST_TH="${DIST_TH:-1.5}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

usage() {
    cat <<EOF
Usage:
  bash test_once_sequence_split.sh [checkpoint_path]

Defaults:
  checkpoint_path = ${REPO_ROOT}/work_dirs/once_3dlanes_sequence_split/latest.pth
  CONDA_ENV       = bevlanedet
  GPU_IDS         = 0
  RATIO_TH        = 0.6
  DIST_TH         = 1.5

Optional environment variables:
  CONDA_BASE      Conda install root. Default: /home/lijishuo/miniconda3
  CONDA_ENV       Conda env name. Default: bevlanedet
  CONFIG_PATH     Validation config. Default: tools/once_sequence_split_config.py
  GPU_IDS         CUDA_VISIBLE_DEVICES value for inference
  RATIO_TH        Ratio threshold for val_offical
  DIST_TH         Distance threshold for val_offical
  PRED_ROOT       Prediction output directory
  GT_ROOT         Override GT root
  INDEX_FILE      Override split index file
  IMAGE_ROOTS     Colon-separated image roots. Example: /data/a:/data/b
  VAL_BATCH_SIZE  Override validation batch size
  VAL_NUM_WORKERS Override validation num_workers
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
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
PRED_ROOT="${PRED_ROOT:-${REPO_ROOT}/work_dirs/once_3dlanes_sequence_split/predictions/${CHECKPOINT_STEM}}"

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
