#!/bin/bash

PRED_ROOT="work_dirs/once_3dlanes/predictions"
RATIO_TH=0.6

for i in {0..29}; do
    epoch_str=$(printf "epoch_%03d" $i)
    pred_path="${PRED_ROOT}/${epoch_str}"

    echo "========================================"
    echo "Evaluating ${epoch_str}"
    echo "========================================"

    python tools/eval_once_with_ratio.py \
        --pred-root "${pred_path}" \
        --ratio-th ${RATIO_TH}
done
