import argparse
import json
import os
import sys

_mpl_config_dir = os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.makedirs(_mpl_config_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _mpl_config_dir

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.util.cluster import embedding_post
from models.util.load_model import load_model
from models.util.post_process import bev_instance2points_with_offset_z
from utils.config_util import load_config_module


BENCHMARK_ROOT = os.path.join(REPO_ROOT, "thirdparty", "once_3dlanes_benchmark")
if BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, BENCHMARK_ROOT)

from evaluation.eval_utils import LaneEval  # noqa: E402


_UNSET = object()


def parse_args():
    parser = argparse.ArgumentParser(description="Run BEV-LaneDet inference on ONCE-3DLanes")
    parser.add_argument("--config", default=os.path.join("tools", "once_config.py"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpus", default=None, help="Comma-separated visible GPU ids")
    parser.add_argument("--gt-root", default=None, help="Ground-truth json root, such as data/ONCE-3DLanes/val")
    parser.add_argument("--image-root", action="append", default=None, help="Image root, can be passed multiple times")
    parser.add_argument("--index-file", default=None, help="Optional split index file")
    parser.add_argument("--pred-root", default=None, help="Prediction output root")
    parser.add_argument("--benchmark-cfg", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug option to limit inference samples")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--post-conf", type=float, default=-0.7)
    parser.add_argument("--post-emb-margin", type=float, default=6.0)
    parser.add_argument("--post-min-cluster-size", type=int, default=15)
    return parser.parse_args()


def configure_visible_devices(gpus):
    if gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def count_expected_samples(label_root, index_file):
    if index_file and os.path.isfile(index_file):
        expected = 0
        with open(index_file, "r") as handle:
            for line in handle:
                rel_path = line.strip().lstrip("/")
                if not rel_path.endswith(".jpg"):
                    continue
                sequence, camera, image_name = rel_path.split("/")
                label_path = os.path.join(label_root, sequence, camera, image_name.replace(".jpg", ".json"))
                if os.path.isfile(label_path):
                    expected += 1
        return expected

    expected = 0
    for sequence in os.listdir(label_root):
        camera_dir = os.path.join(label_root, sequence, "cam01")
        if not os.path.isdir(camera_dir):
            continue
        expected += sum(1 for file_name in os.listdir(camera_dir) if file_name.endswith(".json"))
    return expected


def ensure_eval_coverage(dataset, label_root, index_file, max_samples):
    if max_samples is not None:
        raise RuntimeError("Benchmark evaluation requires full coverage. Remove --max-samples or pass --skip-eval.")
    expected = count_expected_samples(label_root, index_file)
    if expected != len(dataset):
        raise RuntimeError(
            f"Evaluation requires predictions for all GT frames, but only {len(dataset)} / {expected} samples have images."
        )


def save_prediction(pred_root, sample, lanes):
    save_dir = os.path.join(pred_root, sample["sequence"], sample["camera"])
    os.makedirs(save_dir, exist_ok=True)

    pred_lanes = []
    for lane in lanes:
        forward, lateral, height, _ = lane
        points = np.stack([lateral, height, forward], axis=1)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if points.shape[0] < 2:
            continue
        pred_lanes.append(
            {
                "points": np.round(points, 3).tolist(),
                "score": 1.0,
            }
        )

    save_path = os.path.join(save_dir, sample["frame"] + ".json")
    with open(save_path, "w") as handle:
        json.dump({"lanes": pred_lanes}, handle)


def infer_device_from_model(model):
    return next(model.parameters()).device


def extract_bev_outputs(model_outputs):
    if isinstance(model_outputs, tuple):
        if len(model_outputs) == 4:
            return model_outputs
        if len(model_outputs) == 2 and isinstance(model_outputs[0], tuple):
            return model_outputs[0]
    raise RuntimeError(f"Unsupported model output format for validation: {type(model_outputs)}")


def run_validation(
    configs,
    model=None,
    checkpoint=None,
    gt_root=None,
    image_roots=None,
    index_file=_UNSET,
    pred_root=None,
    benchmark_cfg=None,
    batch_size=None,
    num_workers=None,
    max_samples=None,
    skip_eval=False,
    post_conf=None,
    post_emb_margin=None,
    post_min_cluster_size=None,
    device=None,
    desc=None,
):
    gt_root = gt_root or configs.val_gt_root
    image_roots = image_roots or configs.val_image_roots
    if index_file is _UNSET:
        index_file = configs.val_index_file
    pred_root = pred_root or os.path.join(
        getattr(configs, "val_prediction_root", os.path.join(configs.model_save_path, "predictions")),
        os.path.basename(gt_root.rstrip("/")),
    )
    benchmark_cfg = benchmark_cfg or configs.benchmark_cfg_file
    post_conf = getattr(configs, "post_conf", -0.7) if post_conf is None else post_conf
    post_emb_margin = getattr(configs, "post_emb_margin", 6.0) if post_emb_margin is None else post_emb_margin
    post_min_cluster_size = (
        getattr(configs, "post_min_cluster_size", 15)
        if post_min_cluster_size is None
        else post_min_cluster_size
    )

    loader_args = dict(configs.val_loader_args)
    if batch_size is not None:
        loader_args["batch_size"] = batch_size
    if num_workers is not None:
        loader_args["num_workers"] = num_workers

    dataset = configs.build_eval_dataset(
        label_root=gt_root,
        image_roots=image_roots,
        index_file=index_file,
        max_samples=max_samples,
        skip_missing_images=True,
    )
    if not skip_eval:
        ensure_eval_coverage(dataset, gt_root, index_file, max_samples)

    dataloader = DataLoader(dataset, **loader_args, pin_memory=torch.cuda.is_available())

    created_model = False
    restore_training = False
    if model is None:
        checkpoint = checkpoint or getattr(configs, "default_eval_model", None)
        if checkpoint is None or not os.path.isfile(checkpoint):
            raise FileNotFoundError("Checkpoint not found. Pass --checkpoint or set default_eval_model in the config.")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = configs.model(train=False)
        model = load_model(model, checkpoint).to(device)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        created_model = True
    else:
        device = device or infer_device_from_model(model)
        restore_training = model.training

    model.eval()
    os.makedirs(pred_root, exist_ok=True)

    iterator = tqdm(dataloader, desc=desc) if desc else tqdm(dataloader)
    for images, samples in iterator:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            seg, emb, offset_y, height = extract_bev_outputs(model(images))
            seg = seg.detach().cpu().numpy()
            emb = emb.detach().cpu().numpy()
            offset_y = torch.sigmoid(offset_y).detach().cpu().numpy()
            height = height.detach().cpu().numpy()

        current_batch_size = seg.shape[0]
        for idx in range(current_batch_size):
            prediction = (seg[idx : idx + 1], emb[idx : idx + 1])
            canvas, _ = embedding_post(
                prediction,
                conf=post_conf,
                emb_margin=post_emb_margin,
                min_cluster_size=post_min_cluster_size,
                canvas_color=False,
            )
            lanes = bev_instance2points_with_offset_z(
                canvas,
                max_x=configs.x_range[1],
                meter_per_pixal=(configs.meter_per_pixel, configs.meter_per_pixel),
                offset_y=offset_y[idx][0],
                Z=height[idx][0],
            )
            sample = {key: value[idx] for key, value in samples.items()}
            save_prediction(pred_root, sample, lanes)

    print(f"Predictions saved to {pred_root}")
    if not skip_eval:
        LaneEval().lane_evaluation(gt_root, pred_root, benchmark_cfg)

    if restore_training:
        model.train()
    elif created_model:
        del model

    return pred_root


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
        index_file = configs.val_index_file
    run_validation(
        configs=configs,
        checkpoint=args.checkpoint,
        gt_root=gt_root,
        image_roots=image_roots,
        index_file=index_file,
        pred_root=args.pred_root,
        benchmark_cfg=args.benchmark_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        skip_eval=args.skip_eval,
        post_conf=args.post_conf,
        post_emb_margin=args.post_emb_margin,
        post_min_cluster_size=args.post_min_cluster_size,
    )


if __name__ == "__main__":
    main()
