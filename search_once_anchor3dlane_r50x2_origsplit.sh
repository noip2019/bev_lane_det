#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/lijishuo/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bevlanedet}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/tools/once_anchor3dlane_r50x2_origsplit_config.py}"
CHECKPOINT_PATH="${1:-${REPO_ROOT}/work_dirs/once_3dlanes_anchor_r50x2_origsplit/latest.pth}"
GPU_IDS="${GPU_IDS:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
POST_CONF_VALUES="${POST_CONF_VALUES:--0.8,-0.7,-0.6}"
POST_EMB_MARGIN_VALUES="${POST_EMB_MARGIN_VALUES:-5.0,6.0,7.0}"
POST_MIN_CLUSTER_SIZE_VALUES="${POST_MIN_CLUSTER_SIZE_VALUES:-10,15,20}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage:
  bash search_once_anchor3dlane_r50x2_origsplit.sh [checkpoint_path]

Optional environment variables:
  GPU_IDS                       CUDA_VISIBLE_DEVICES value for inference. Default: 0
  POST_CONF_VALUES              Comma-separated list. Default: -0.8,-0.7,-0.6
  POST_EMB_MARGIN_VALUES        Comma-separated list. Default: 5.0,6.0,7.0
  POST_MIN_CLUSTER_SIZE_VALUES  Comma-separated list. Default: 10,15,20
  PRED_ROOT                     Root directory for all trial predictions
  RESULTS_JSON                  Output json path for search summary
  GT_ROOT                       Override GT root
  INDEX_FILE                    Override split index file
  IMAGE_ROOTS                   Colon-separated image roots
  VAL_BATCH_SIZE                Override validation batch size
  VAL_NUM_WORKERS               Override validation num_workers
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

SEARCH_ARGS=(
    --config "${CONFIG_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    "--post-conf-values=${POST_CONF_VALUES}"
    "--post-emb-margin-values=${POST_EMB_MARGIN_VALUES}"
    "--post-min-cluster-size-values=${POST_MIN_CLUSTER_SIZE_VALUES}"
)

if [[ -n "${PRED_ROOT:-}" ]]; then
    SEARCH_ARGS+=(--pred-root "${PRED_ROOT}")
fi

if [[ -n "${RESULTS_JSON:-}" ]]; then
    SEARCH_ARGS+=(--results-json "${RESULTS_JSON}")
fi

if [[ -n "${GT_ROOT:-}" ]]; then
    SEARCH_ARGS+=(--gt-root "${GT_ROOT}")
fi

if [[ -n "${INDEX_FILE:-}" ]]; then
    SEARCH_ARGS+=(--index-file "${INDEX_FILE}")
fi

if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
    SEARCH_ARGS+=(--batch-size "${VAL_BATCH_SIZE}")
fi

if [[ -n "${VAL_NUM_WORKERS:-}" ]]; then
    SEARCH_ARGS+=(--num-workers "${VAL_NUM_WORKERS}")
fi

if [[ -n "${IMAGE_ROOTS:-}" ]]; then
    OLD_IFS="${IFS}"
    IFS=":"
    read -r -a IMAGE_ROOT_ARRAY <<< "${IMAGE_ROOTS}"
    IFS="${OLD_IFS}"
    for image_root in "${IMAGE_ROOT_ARRAY[@]}"; do
        if [[ -n "${image_root}" ]]; then
            SEARCH_ARGS+=(--image-root "${image_root}")
        fi
    done
fi

cd "${REPO_ROOT}"
mkdir -p "${MPLCONFIGDIR}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MPLCONFIGDIR

python tools/search_once_post_params.py "${SEARCH_ARGS[@]}"
