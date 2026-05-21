import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.util_val.val_offical import LaneEval

BENCHMARK_ROOT = os.path.join(REPO_ROOT, "thirdparty", "once_3dlanes_benchmark")
if BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, BENCHMARK_ROOT)

from evaluation.eval_utils import LaneEval as OnceBenchmarkLaneEval


def _convert_once_lane(points):
    lane = np.asarray(points, dtype=np.float32)
    if lane.ndim != 2 or lane.shape[0] < 2 or lane.shape[1] != 3:
        return None

    # ONCE stores points as [x_lateral, y_height, z_forward].
    # val_offical expects [x_right, y_forward, z_up].
    lane = lane[:, [0, 2, 1]]
    lane = lane[np.isfinite(lane).all(axis=1)]
    if lane.shape[0] < 2:
        return None

    order = np.argsort(lane[:, 1], kind="mergesort")
    lane = lane[order]

    dedup = [lane[0]]
    for point in lane[1:]:
        if np.isclose(point[1], dedup[-1][1]):
            dedup[-1] = point
            continue
        dedup.append(point)
    if len(dedup) < 2:
        return None

    return np.asarray(dedup, dtype=np.float32).tolist()


def _load_gt_lanes(gt_path):
    with open(gt_path, "r") as handle:
        data = json.load(handle)
    lanes = []
    for lane in data.get("lanes", []):
        lane = _convert_once_lane(lane)
        if lane is None:
            continue
        lanes.append(lane)
    return lanes


def _load_pred_lanes(pred_path):
    with open(pred_path, "r") as handle:
        data = json.load(handle)
    lanes = []
    for lane in data.get("lanes", []):
        points = _convert_once_lane(lane.get("points", []))
        if points is None:
            continue
        lanes.append(points)
    return lanes


def _collect_gt_files(gt_root, index_file=None):
    gt_root = Path(gt_root)
    if index_file is None:
        gt_files = sorted(gt_root.glob("*/*/*.json"))
        if not gt_files:
            raise RuntimeError(f"No GT json files found under {gt_root}")
        return gt_files

    index_file = Path(index_file)
    if not index_file.is_file():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    gt_files = []
    missing_gt = []
    with index_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            rel_path = line.strip().lstrip("/")
            if not rel_path or not rel_path.endswith(".jpg"):
                continue
            sequence, camera, image_name = rel_path.split("/")
            gt_path = gt_root / sequence / camera / image_name.replace(".jpg", ".json")
            if gt_path.is_file():
                gt_files.append(gt_path)
            else:
                missing_gt.append(str(gt_path.relative_to(gt_root)))

    if missing_gt:
        raise RuntimeError(
            f"Missing {len(missing_gt)} GT json files under {gt_root}. Examples: {missing_gt[:5]}"
        )
    if not gt_files:
        raise RuntimeError(f"No GT json files were resolved from index file {index_file}")
    return gt_files


def evaluate_once(gt_root, pred_root, ratio_th=None, dist_th=None, index_file=None):
    gt_root = Path(gt_root)
    pred_root = Path(pred_root)
    evaluator = LaneEval()
    if ratio_th is not None:
        evaluator.ratio_th = float(ratio_th)
    if dist_th is not None:
        evaluator.dist_th = float(dist_th)

    gt_files = _collect_gt_files(gt_root, index_file=index_file)

    missing_predictions = []
    for gt_path in tqdm(gt_files, desc=f"ratio@{evaluator.ratio_th}"):
        rel_path = gt_path.relative_to(gt_root)
        pred_path = pred_root / rel_path
        if not pred_path.is_file():
            missing_predictions.append(str(rel_path))
            continue
        gt_lanes = _load_gt_lanes(gt_path)
        pred_lanes = _load_pred_lanes(pred_path)
        evaluator.bench_all(pred_lanes, gt_lanes)

    if missing_predictions:
        raise RuntimeError(
            f"Missing {len(missing_predictions)} prediction files under {pred_root}. "
            f"Examples: {missing_predictions[:5]}"
        )

    return evaluator.show()


def evaluate_once_benchmark(gt_root, pred_root, benchmark_cfg):
    benchmark_eval = OnceBenchmarkLaneEval()
    benchmark_eval.lane_evaluation(str(gt_root), str(pred_root), benchmark_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ONCE predictions with both val_offical ratio metric and ONCE official benchmark"
    )
    parser.add_argument("--gt-root", default=os.path.join("data", "ONCE-3DLanes", "val"))
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--index-file", default=None, help="Optional split index file to restrict evaluation scope.")
    parser.add_argument(
        "--benchmark-cfg",
        default=os.path.join("thirdparty", "once_3dlanes_benchmark", "cfg", "eval.json"),
        help="Config path for ONCE official benchmark evaluation.",
    )
    parser.add_argument(
        "--ratio-th",
        type=float,
        default=None,
        help="Override LaneEval.ratio_th at runtime. Defaults to the value in utils/util_val/val_offical.py.",
    )
    parser.add_argument(
        "--dist-th",
        type=float,
        default=None,
        help="Override LaneEval.dist_th at runtime. Defaults to the value in utils/util_val/val_offical.py.",
    )
    parser.add_argument(
        "--skip-ratio-metric",
        action="store_true",
        help="Skip val_offical ratio/dist based evaluation.",
    )
    parser.add_argument(
        "--skip-once-benchmark",
        action="store_true",
        help="Skip ONCE official benchmark evaluation.",
    )
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)

    if not args.skip_ratio_metric:
        print("=== val_offical ratio metric ===")
        evaluate_once(
            gt_root,
            pred_root,
            ratio_th=args.ratio_th,
            dist_th=args.dist_th,
            index_file=args.index_file,
        )

    # if not args.skip_once_benchmark:
    #     print("=== ONCE official benchmark ===")
    #     evaluate_once_benchmark(gt_root, pred_root, args.benchmark_cfg)


if __name__ == "__main__":
    main()
