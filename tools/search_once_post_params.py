import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.util.load_model import load_model
from tools.eval_once_with_ratio import evaluate_once
from tools.val_once import run_validation
from utils.config_util import load_config_module


def parse_args():
    parser = argparse.ArgumentParser(description="Search ONCE inference post-processing hyperparameters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpus", default=None, help="Comma-separated visible GPU ids")
    parser.add_argument("--gt-root", default=None, help="Ground-truth json root")
    parser.add_argument("--image-root", action="append", default=None, help="Image root, can be passed multiple times")
    parser.add_argument("--index-file", default=None, help="Optional split index file")
    parser.add_argument("--pred-root", default=None, help="Prediction output root for all trials")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--post-conf-values", default="-0.8,-0.7,-0.6")
    parser.add_argument("--post-emb-margin-values", default="5.0,6.0,7.0")
    parser.add_argument("--post-min-cluster-size-values", default="10,15,20")
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional output json path. Defaults to <pred-root>/search_results.json",
    )
    return parser.parse_args()


def configure_visible_devices(gpus):
    if gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def combo_tag(post_conf, post_emb_margin, post_min_cluster_size):
    conf_tag = f"{post_conf:.3f}".replace("-", "m").replace(".", "p")
    emb_tag = f"{post_emb_margin:.3f}".replace("-", "m").replace(".", "p")
    return f"conf_{conf_tag}__emb_{emb_tag}__min_{post_min_cluster_size:d}"


def build_model(configs, checkpoint_path, device):
    model = configs.model(train=False)
    model = load_model(model, checkpoint_path).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


def main():
    args = parse_args()
    configure_visible_devices(args.gpus)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    configs = load_config_module(args.config)
    gt_root = args.gt_root or configs.val_gt_root
    image_roots = args.image_root or configs.val_image_roots
    if args.index_file is not None:
        index_file = args.index_file
    elif args.gt_root is not None:
        index_file = None
    else:
        index_file = getattr(configs, "val_index_file", None)

    checkpoint_stem = Path(args.checkpoint).stem
    pred_root = args.pred_root or os.path.join(
        getattr(configs, "val_prediction_root", os.path.join(configs.model_save_path, "predictions")),
        f"{checkpoint_stem}_search",
    )
    os.makedirs(pred_root, exist_ok=True)

    results_json = args.results_json or os.path.join(pred_root, "search_results.json")

    post_conf_values = parse_float_list(args.post_conf_values)
    post_emb_margin_values = parse_float_list(args.post_emb_margin_values)
    post_min_cluster_size_values = parse_int_list(args.post_min_cluster_size_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(configs, args.checkpoint, device)

    ratio_th = getattr(configs, "val_ratio_th", 0.6)
    dist_th = getattr(configs, "val_dist_th", 1.5)

    trials = []
    best_result = None

    combos = list(itertools.product(post_conf_values, post_emb_margin_values, post_min_cluster_size_values))
    print(f"Running {len(combos)} trials with ratio_th={ratio_th} dist_th={dist_th}")

    for trial_idx, (post_conf, post_emb_margin, post_min_cluster_size) in enumerate(combos, start=1):
        tag = combo_tag(post_conf, post_emb_margin, post_min_cluster_size)
        trial_pred_root = os.path.join(pred_root, tag)
        print(
            f"[{trial_idx}/{len(combos)}] "
            f"post_conf={post_conf} post_emb_margin={post_emb_margin} post_min_cluster_size={post_min_cluster_size}"
        )

        run_validation(
            configs=configs,
            model=model,
            gt_root=gt_root,
            image_roots=image_roots,
            index_file=index_file,
            pred_root=trial_pred_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_eval=True,
            post_conf=post_conf,
            post_emb_margin=post_emb_margin,
            post_min_cluster_size=post_min_cluster_size,
            device=device,
            desc=f"search {trial_idx}/{len(combos)}",
        )

        metrics = evaluate_once(
            gt_root=gt_root,
            pred_root=trial_pred_root,
            ratio_th=ratio_th,
            dist_th=dist_th,
            index_file=index_file,
        )

        result = {
            "trial_idx": trial_idx,
            "pred_root": trial_pred_root,
            "post_conf": post_conf,
            "post_emb_margin": post_emb_margin,
            "post_min_cluster_size": post_min_cluster_size,
            "metrics": metrics,
        }
        trials.append(result)

        f1_score = metrics.get("f1_score", float("-inf"))
        if best_result is None or f1_score > best_result["metrics"].get("f1_score", float("-inf")):
            best_result = result
            print(f"New best f1_score={f1_score:.6f} at {tag}")

    payload = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "gt_root": os.path.abspath(gt_root),
        "index_file": None if index_file is None else os.path.abspath(index_file),
        "ratio_th": ratio_th,
        "dist_th": dist_th,
        "search_space": {
            "post_conf_values": post_conf_values,
            "post_emb_margin_values": post_emb_margin_values,
            "post_min_cluster_size_values": post_min_cluster_size_values,
        },
        "best_result": best_result,
        "trials": trials,
    }
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("=== Best Result ===")
    print(json.dumps(best_result, indent=2))
    print(f"Saved full search results to {results_json}")


if __name__ == "__main__":
    main()
