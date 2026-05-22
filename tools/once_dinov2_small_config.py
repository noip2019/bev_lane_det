import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from loader.bev_road.once_data import ONCE3DLanesDatasetWithOffset, ONCE3DLanesDatasetWithOffsetVal
from loader.bev_road.once_split_data import build_sequence_split_index
from models.model.single_camera_bev import BEV_LaneDet


def _env_paths(env_name, default_paths):
    raw_value = os.getenv(env_name)
    if not raw_value:
        return default_paths
    return [path for path in raw_value.split(os.pathsep) if path]


_UNSET = object()


repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_root = os.getenv("ONCE_3DLANES_ROOT", os.path.join(repo_root, "data", "ONCE-3DLanes"))

train_gt_root = os.path.join(dataset_root, "train")
val_gt_root = train_gt_root
base_index_file = os.path.join(dataset_root, "list", "train.txt")

default_train_split_json = "/home/lijishuo/BEVLaneDetCopy_toPKU/data/split_train_with_height_pitch_by_sequence_9.json"
default_test_split_json = "/home/lijishuo/BEVLaneDetCopy_toPKU/data/split_test_with_height_pitch_by_sequence_1.json"
train_split_json = os.getenv("ONCE_3DLANES_SEQ_TRAIN_SPLIT_JSON", default_train_split_json)
val_split_json = os.getenv("ONCE_3DLANES_SEQ_VAL_SPLIT_JSON", default_test_split_json)

train_image_roots = _env_paths("ONCE_3DLANES_TRAIN_IMAGE_ROOT", [os.path.join(dataset_root, "raw_data")])
val_image_roots = _env_paths("ONCE_3DLANES_VAL_IMAGE_ROOT", train_image_roots)

model_save_path = os.getenv("ONCE_3DLANES_WORKDIR", os.path.join(repo_root, "work_dirs", "once_3dlanes_dinov2_small"))
default_eval_model = os.path.join(model_save_path, "latest.pth")
benchmark_cfg_file = os.path.join(repo_root, "thirdparty", "once_3dlanes_benchmark", "cfg", "eval.json")
default_backbone_ckpt = os.getenv(
    "ONCE_3DLANES_BACKBONE_CKPT",
    "/home/lijishuo/dinov2/dinov2_vits14_reg4_pretrain.pth",
)

# DINOv2 ViT-S/14 requires input sizes divisible by 14.
input_shape = (728, 952)
output_2d_shape = (182, 238)

x_range = (0, 50)
y_range = (-10, 10)
meter_per_pixel = 0.5
bev_shape = (
    int((x_range[1] - x_range[0]) / meter_per_pixel),
    int((y_range[1] - y_range[0]) / meter_per_pixel),
)

loader_args = dict(
    batch_size=4,
    num_workers=8,
    shuffle=True,
)

val_loader_args = dict(
    batch_size=4,
    num_workers=4,
    shuffle=False,
)
val_every_epochs = int(os.getenv("ONCE_3DLANES_VAL_EVERY", "1"))
val_max_samples = None
val_skip_benchmark = True
val_eval_mode = "val_offical"
val_ratio_th = 0.6
val_dist_th = 1.5
val_prediction_root = os.path.join(model_save_path, "predictions")
post_conf = -0.7
post_emb_margin = 6.0
post_min_cluster_size = 15

epochs = 60
freeze_backbone_epochs = 3
optimizer = AdamW
optimizer_params = dict(
    lr=5e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
    amsgrad=False,
)
scheduler = CosineAnnealingLR
backbone_pretrained = True
backbone_pretrained_path = default_backbone_ckpt
backbone_name = "dinov2_vits14_reg"
split_index_dir = os.path.join(model_save_path, "split_indices")
train_index_file = build_sequence_split_index(base_index_file, train_split_json, split_index_dir)
val_index_file = build_sequence_split_index(base_index_file, val_split_json, split_index_dir)


def model(train=True):
    return BEV_LaneDet(
        bev_shape=bev_shape,
        output_2d_shape=output_2d_shape,
        train=train,
        pretrained_backbone=backbone_pretrained,
        pretrained_backbone_path=backbone_pretrained_path,
        backbone_name=backbone_name,
        input_shape=input_shape,
    )


def _image_transform(is_train):
    transforms = [A.Resize(height=input_shape[0], width=input_shape[1])]
    if is_train:
        transforms.extend(
            [
                A.MotionBlur(p=0.2),
                A.RandomBrightnessContrast(),
                A.ColorJitter(p=0.1),
                A.RandomGamma(p=0.2),
                A.HueSaturationValue(p=0.1),
            ]
        )
    transforms.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(transforms)


def train_dataset(max_samples=None):
    return ONCE3DLanesDatasetWithOffset(
        label_root=train_gt_root,
        image_roots=train_image_roots,
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        data_trans=_image_transform(is_train=True),
        output_2d_shape=output_2d_shape,
        index_file=train_index_file,
        max_samples=max_samples,
        skip_missing_images=True,
    )


def build_eval_dataset(
    label_root=_UNSET,
    image_roots=_UNSET,
    index_file=_UNSET,
    max_samples=None,
    skip_missing_images=True,
):
    return ONCE3DLanesDatasetWithOffsetVal(
        label_root=val_gt_root if label_root is _UNSET else label_root,
        image_roots=val_image_roots if image_roots is _UNSET else image_roots,
        data_trans=_image_transform(is_train=False),
        index_file=val_index_file if index_file is _UNSET else index_file,
        max_samples=max_samples,
        skip_missing_images=skip_missing_images,
    )


def val_dataset(max_samples=None):
    return build_eval_dataset(max_samples=max_samples)
