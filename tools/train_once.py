import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.loss import IoULoss, NDPushPullLoss
from tools.eval_once_with_ratio import evaluate_once
from models.util.load_model import load_model
from tools.val_once import run_validation
from utils.config_util import load_config_module


class CombineModelAndLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.push_pull_loss = NDPushPullLoss(1.0, 1.0, 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        inputs,
        gt_seg=None,
        gt_instance=None,
        gt_offset_y=None,
        gt_height=None,
        image_gt_segment=None,
        image_gt_instance=None,
        train=True,
    ):
        bev_res, image_res = self.model(inputs)
        pred, emb, offset_y, height = bev_res
        pred_2d, emb_2d = image_res

        if not train:
            return pred

        loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
        loss_emb = self.push_pull_loss(emb, gt_instance)
        loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
        loss_height = self.mse_loss(gt_seg * height, gt_height)

        loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
        loss_emb_2d = self.push_pull_loss(emb_2d, image_gt_instance)

        loss_total_bev = 3 * loss_seg + 0.5 * loss_emb
        loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
        return pred, loss_total_bev, loss_total_2d, 60 * loss_offset, 30 * loss_height


def parse_args():
    parser = argparse.ArgumentParser(description="Train BEV-LaneDet on ONCE-3DLanes")
    parser.add_argument("--config", default=os.path.join("tools", "once_config.py"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpus", default=None, help="Comma-separated visible GPU ids")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug option to limit training samples")
    parser.add_argument("--val-every", type=int, default=None, help="Run validation every N epochs. Set 0 to disable.")
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None, help="Debug option to limit validation samples")
    parser.add_argument("--skip-val-eval", action="store_true", help="Skip validation scoring and only dump predictions")
    parser.add_argument("--post-conf", type=float, default=None)
    parser.add_argument("--post-emb-margin", type=float, default=None)
    parser.add_argument("--post-min-cluster-size", type=int, default=None)
    parser.add_argument("--dist-backend", default=None, help="DDP backend, defaults to nccl on CUDA and gloo on CPU")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    return parser.parse_args()


def configure_visible_devices(gpus):
    if gpus:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def main_print(message):
    if is_main_process():
        print(message)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def init_distributed_mode(args):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "-1"))

    if distributed:
        backend = args.dist_backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend=backend)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend=backend)
        main_print(f"Initialized DDP: rank={get_rank()} world_size={get_world_size()} backend={backend}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, local_rank, device


def unwrap_model(model_wrapper):
    return model_wrapper.module if hasattr(model_wrapper, "module") else model_wrapper


def get_validation_model(model_wrapper):
    return unwrap_model(model_wrapper).model


def load_training_weights(wrapper, checkpoint_path, optimizer=None):
    if checkpoint_path is None:
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint or "state_dict" in checkpoint:
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict"))
        try:
            wrapper.load_state_dict(state_dict)
        except RuntimeError:
            wrapper.model = load_model(wrapper.model, checkpoint_path)
        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        return int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else 0

    if "models" in checkpoint:
        wrapper.load_state_dict(checkpoint["models"])
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else 0

    wrapper.model = load_model(wrapper.model, checkpoint_path)
    return 0


def save_training_checkpoint(model_wrapper, optimizer, save_dir, epoch, file_name):
    os.makedirs(save_dir, exist_ok=True)
    model_state = unwrap_model(model_wrapper).state_dict()
    torch.save(
        {
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
        },
        os.path.join(save_dir, file_name),
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    total_epochs,
    epoch_start_time,
    train_start_time,
    start_epoch,
):
    model.train()
    total_iters = len(dataloader)
    iter_start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        (
            input_data,
            gt_seg_data,
            gt_emb_data,
            offset_y_data,
            height_data,
            image_gt_segment,
            image_gt_instance,
        ) = batch

        input_data = input_data.to(device, non_blocking=True)
        gt_seg_data = gt_seg_data.to(device, non_blocking=True)
        gt_emb_data = gt_emb_data.to(device, non_blocking=True)
        offset_y_data = offset_y_data.to(device, non_blocking=True)
        height_data = height_data.to(device, non_blocking=True)
        image_gt_segment = image_gt_segment.to(device, non_blocking=True)
        image_gt_instance = image_gt_instance.to(device, non_blocking=True)

        prediction, loss_bev, loss_2d, loss_offset, loss_height = model(
            input_data,
            gt_seg_data,
            gt_emb_data,
            offset_y_data,
            height_data,
            image_gt_segment,
            image_gt_instance,
        )

        loss_bev = loss_bev.mean()
        loss_2d = loss_2d.mean()
        loss_offset = loss_offset.mean()
        loss_height = loss_height.mean()
        loss_total = loss_bev + 0.5 * loss_2d + loss_offset + loss_height

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        completed_iters = batch_idx + 1
        avg_iter_time = (iter_end_time - epoch_start_time) / completed_iters
        epoch_eta_seconds = avg_iter_time * max(total_iters - completed_iters, 0)
        completed_epoch_progress = (epoch - start_epoch) + completed_iters / total_iters
        avg_epoch_time = (iter_end_time - train_start_time) / max(completed_epoch_progress, 1e-6)
        remaining_epoch_progress = max(total_epochs - (epoch + completed_iters / total_iters), 0)
        total_eta_seconds = avg_epoch_time * remaining_epoch_progress

        if is_main_process() and batch_idx % 50 == 0:
            print(
                f"epoch={epoch} iter={batch_idx} bev={loss_bev.item():.4f} "
                f"offset={loss_offset.item():.4f} height={loss_height.item():.4f} "
                f"iter_time={iter_time:.2f}s avg_iter_time={avg_iter_time:.2f}s "
                f"epoch_eta={format_duration(epoch_eta_seconds)} total_eta={format_duration(total_eta_seconds)}"
            )
        if is_main_process() and batch_idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            print(
                {
                    "epoch": epoch,
                    "iter": batch_idx,
                    "BEV Loss": loss_bev.item(),
                    "offset loss": loss_offset.item(),
                    "height loss": loss_height.item(),
                    "F1_BEV_seg": f1_bev_seg,
                    "avg_iter_time_sec": round(avg_iter_time, 3),
                    "epoch_eta": format_duration(epoch_eta_seconds),
                    "total_eta": format_duration(total_eta_seconds),
                }
            )
        iter_start_time = time.time()


def should_run_validation(epoch, total_epochs, val_every):
    if val_every is None or val_every <= 0:
        return False
    return (epoch + 1) % val_every == 0 or epoch == total_epochs - 1


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def main():
    args = parse_args()
    configure_visible_devices(args.gpus)
    distributed, local_rank, device = init_distributed_mode(args)

    configs = load_config_module(args.config)
    train_loader_args = dict(configs.loader_args)
    if args.batch_size is not None:
        train_loader_args["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_loader_args["num_workers"] = args.num_workers
    if args.epochs is not None:
        configs.epochs = args.epochs

    val_every = args.val_every if args.val_every is not None else getattr(configs, "val_every_epochs", 0)
    val_max_samples = args.val_max_samples if args.val_max_samples is not None else getattr(configs, "val_max_samples", None)
    val_eval_mode = getattr(configs, "val_eval_mode", "once_benchmark")
    val_ratio_th = getattr(configs, "val_ratio_th", None)
    val_dist_th = getattr(configs, "val_dist_th", None)
    skip_val_eval = (
        args.skip_val_eval
        or getattr(configs, "val_skip_benchmark", False)
        or val_eval_mode == "val_offical"
    )

    base_model = configs.model(train=True)
    model = CombineModelAndLoss(base_model).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    start_epoch = load_training_weights(unwrap_model(model), args.checkpoint, optimizer)

    train_dataset = configs.train_dataset(max_samples=args.max_samples)
    sampler = None
    if distributed:
        shuffle = train_loader_args.pop("shuffle", True)
        sampler = DistributedSampler(train_dataset, shuffle=shuffle)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            **train_loader_args,
        )
    else:
        train_loader = DataLoader(train_dataset, pin_memory=torch.cuda.is_available(), **train_loader_args)

    per_rank_batch_size = train_loader_args["batch_size"]
    global_batch_size = per_rank_batch_size * get_world_size()
    iters_per_epoch = len(train_loader)

    main_print(
        f"Training setup: device={device} world_size={get_world_size()} "
        f"train_samples={len(train_dataset)} per_rank_batch_size={per_rank_batch_size} "
        f"global_batch_size={global_batch_size} iters_per_epoch={iters_per_epoch} "
        f"start_epoch={start_epoch} val_every={val_every}"
    )

    train_start_time = time.time()
    try:
        for epoch in range(start_epoch, configs.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            main_print(f"{'*' * 40} epoch {epoch}/{configs.epochs - 1} iters={iters_per_epoch}")
            train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                configs.epochs,
                epoch_start_time,
                train_start_time,
                start_epoch,
            )
            scheduler.step()
            main_print(
                f"Finished epoch {epoch} in {format_duration(time.time() - epoch_start_time)} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}"
            )

            if is_main_process():
                save_training_checkpoint(model, optimizer, configs.model_save_path, epoch, f"ep{epoch:03d}.pth")
                save_training_checkpoint(model, optimizer, configs.model_save_path, epoch, "latest.pth")

            if distributed:
                dist.barrier()

            if should_run_validation(epoch, configs.epochs, val_every):
                if is_main_process():
                    pred_root = os.path.join(
                        getattr(configs, "val_prediction_root", os.path.join(configs.model_save_path, "predictions")),
                        f"epoch_{epoch:03d}",
                    )
                    main_print(f"Running validation for epoch {epoch} -> {pred_root}")
                    run_validation(
                        configs=configs,
                        model=get_validation_model(model),
                        pred_root=pred_root,
                        batch_size=args.val_batch_size,
                        num_workers=args.val_num_workers,
                        max_samples=val_max_samples,
                        skip_eval=skip_val_eval,
                        post_conf=args.post_conf,
                        post_emb_margin=args.post_emb_margin,
                        post_min_cluster_size=args.post_min_cluster_size,
                        device=device,
                        desc=f"val epoch {epoch}",
                    )
                    if val_eval_mode == "val_offical" and not args.skip_val_eval:
                        if val_max_samples is not None:
                            raise RuntimeError(
                                "val_offical evaluation requires full validation coverage. "
                                "Remove --val-max-samples or pass --skip-val-eval."
                            )
                        main_print(f"Running val_offical evaluation for epoch {epoch} ratio_th={val_ratio_th}")
                        metrics = evaluate_once(
                            gt_root=configs.val_gt_root,
                            pred_root=pred_root,
                            ratio_th=val_ratio_th,
                            dist_th=val_dist_th,
                            index_file=getattr(configs, "val_index_file", None),
                        )
                        main_print(f"val_offical metrics: {metrics}")
                if distributed:
                    dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()
