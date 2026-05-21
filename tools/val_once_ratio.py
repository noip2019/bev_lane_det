import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.eval_once_with_ratio import evaluate_once
from tools.val_once import run_validation
from utils.config_util import load_config_module


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONCE inference and evaluate with val_offical ratio metric")
    parser.add_argument("--config", default=os.path.join("tools", "once_sequence_split_config.py"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpus", default=None, help="Comma-separated visible GPU ids")
    parser.add_argument("--gt-root", default=None, help="Ground-truth json root")
    parser.add_argument("--image-root", action="append", default=None, help="Image root, can be passed multiple times")
    parser.add_argument("--index-file", default=None, help="Optional split index file")
    parser.add_argument("--pred-root", default=None, help="Prediction output root")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug option to limit inference samples")
    parser.add_argument("--skip-ratio-eval", action="store_true")
    parser.add_argument("--ratio-th", type=float, default=None)
    parser.add_argument("--dist-th", type=float, default=None)
    parser.add_argument("--post-conf", type=float, default=None)
    parser.add_argument("--post-emb-margin", type=float, default=None)
    parser.add_argument("--post-min-cluster-size", type=int, default=None)
    return parser.parse_args()


def configure_visible_devices(gpus):
    if gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def main():
    args = parse_args()
    configure_visible_devices(args.gpus)

    configs = load_config_module(args.config)
    gt_root = args.gt_root or configs.val_gt_root
    image_roots = args.image_root or configs.val_image_roots
    if args.index_file is not None:
        index_file = args.index_file
    elif args.gt_root is not None:
        index_file = None
    else:
        index_file = getattr(configs, "val_index_file", None)

    pred_root = run_validation(
        configs=configs,
        checkpoint=args.checkpoint,
        gt_root=gt_root,
        image_roots=image_roots,
        index_file=index_file,
        pred_root=args.pred_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        skip_eval=True,
        post_conf=args.post_conf,
        post_emb_margin=args.post_emb_margin,
        post_min_cluster_size=args.post_min_cluster_size,
        desc="val_offical inference",
    )

    if args.skip_ratio_eval:
        return
    if args.max_samples is not None:
        raise RuntimeError("val_offical evaluation requires full coverage. Remove --max-samples or pass --skip-ratio-eval.")

    ratio_th = args.ratio_th if args.ratio_th is not None else getattr(configs, "val_ratio_th", None)
    dist_th = args.dist_th if args.dist_th is not None else getattr(configs, "val_dist_th", None)
    print("=== val_offical ratio metric ===")
    evaluate_once(
        gt_root=gt_root,
        pred_root=pred_root,
        ratio_th=ratio_th,
        dist_th=dist_th,
        index_file=index_file,
    )


if __name__ == "__main__":
    main()
