import os
import warnings

import torch
import torchvision as tv
from torch import nn

from models.model.dinov2_vits14_backbone import DINOv2ViTS14RegBackbone


def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])  # 这个 B,V,C,H*W
        bev_view = self.fc_transform(x)  # 拿出一个视角
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model_state", "state_dict", "models", "model", "net"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _strip_state_dict_prefixes(state_dict, prefixes=("module.", "backbone.")):
    cleaned = {}
    for key, value in state_dict.items():
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    changed = True
        cleaned[stripped] = value
    return cleaned


class AnchorResNet50V1c(nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        block = tv.models.resnet.Bottleneck
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 3, stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, 4, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, 6, stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, 3, stride=2, norm_layer=norm_layer)
        self.out_channels = 2048

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _load_pretrained_backbone(model, pretrained_path, prefix="backbone."):
    if not pretrained_path:
        return model
    if not os.path.isfile(pretrained_path):
        warnings.warn(f"Configured backbone checkpoint does not exist: {pretrained_path}")
        return model

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = _strip_state_dict_prefixes(_unwrap_checkpoint_state_dict(checkpoint), prefixes=("module.",))
    if prefix and any(key.startswith(prefix) for key in state_dict):
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(
        f"Loaded backbone from {pretrained_path}: "
        f"matched={len(compatible_state)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model


def build_resnet50_v1c_backbone(pretrained=True, pretrained_path=None):
    try:
        model = AnchorResNet50V1c()
        if pretrained:
            _load_pretrained_backbone(model, pretrained_path, prefix="backbone.")
        return model
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Failed to load Anchor3DLane ResNet50 backbone ({exc}). Falling back to random initialization."
        )
        return AnchorResNet50V1c()


def build_backbone(backbone_name, pretrained=True, pretrained_path=None):
    if backbone_name == "resnet34":
        backbone = build_resnet34_backbone(pretrained=pretrained, pretrained_path=pretrained_path)
        feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        feature_extractor.out_channels = 512
        return feature_extractor
    if backbone_name in {"resnet50_v1c_anchor", "anchor_resnet50_v1c", "r50x2_anchor"}:
        return build_resnet50_v1c_backbone(pretrained=pretrained, pretrained_path=pretrained_path)
    if backbone_name in {"dinov2_vits14_reg", "dinov2_small", "dinov2_vits14_small"}:
        return DINOv2ViTS14RegBackbone(pretrained_path=pretrained_path if pretrained else None)
    raise ValueError(f"Unsupported backbone_name: {backbone_name}")


def _infer_backbone_shapes(backbone, down_block, input_shape):
    was_backbone_training = backbone.training
    was_down_training = down_block.training
    backbone.eval()
    down_block.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 3, input_shape[0], input_shape[1])
        img_s32 = backbone(sample)
        img_s64 = down_block(img_s32)
    backbone.train(was_backbone_training)
    down_block.train(was_down_training)
    return tuple(img_s32.shape[1:]), tuple(img_s64.shape[1:])


def build_resnet34_backbone(pretrained=True, pretrained_path=None):
    try:
        if pretrained and pretrained_path:
            model = tv.models.resnet34(weights=None) if hasattr(tv.models, "ResNet34_Weights") else tv.models.resnet34(pretrained=False)
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded local ResNet34 pretrained backbone from {pretrained_path}")
            return model
        if hasattr(tv.models, "ResNet34_Weights"):
            weights = tv.models.ResNet34_Weights.DEFAULT if pretrained else None
            return tv.models.resnet34(weights=weights)
        return tv.models.resnet34(pretrained=pretrained)
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Failed to load pretrained ResNet34 weights ({exc}). Falling back to random initialization."
        )
        if hasattr(tv.models, "ResNet34_Weights"):
            return tv.models.resnet34(weights=None)
        return tv.models.resnet34(pretrained=False)

# model
# ResNet34 骨干网络 (self.bb)，在 ImageNet 上进行预训练。
# 一个下采样层 (self.down)，用于减小特征图的空间维度。
# 两个全连接变换层 (self.s32transformer 和 self.s64transformer)，将 ResNet 骨干网络的特征图转换为 BEV 表示。
# 车道线检测头 (self.lane_head)，以 BEV 表示作为输入，输出表示检测到的车道线的张量。
# 可选的 2D 图像车道线检测头 (self.lane_head_2d)，以 ResNet 骨干网络的输出作为输入，输出表示原始图像中检测到的车道线的张量。
class BEV_LaneDet(nn.Module):  # BEV-LaneDet
    def __init__(
        self,
        bev_shape,
        output_2d_shape,
        train=True,
        pretrained_backbone=True,
        pretrained_backbone_path=None,
        backbone_name="resnet34",
        input_shape=(576, 1024),
    ):
        super(BEV_LaneDet, self).__init__()
        if pretrained_backbone_path and not os.path.isfile(pretrained_backbone_path):
            warnings.warn(
                f"Configured pretrained backbone path does not exist: {pretrained_backbone_path}. "
                "Falling back to torchvision defaults."
            )
            pretrained_backbone_path = None
        self.backbone_name = backbone_name
        self.bb = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            pretrained_path=pretrained_backbone_path,
        )
        self.bb_out_channels = getattr(self.bb, "out_channels", 512)

        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(self.bb_out_channels, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)

                ),
                downsample=nn.Conv2d(self.bb_out_channels, 1024, kernel_size=3, stride=2, padding=1),
            )
        )

        image_featmap_size, down_featmap_size = _infer_backbone_shapes(self.bb, self.down, input_shape)
        self.image_featmap_size = image_featmap_size
        self.down_featmap_size = down_featmap_size
        self.s32transformer = FCTransform_(image_featmap_size, (256, 25, 5))
        self.s64transformer = FCTransform_(down_featmap_size, (256, 25, 5))
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=self.bb_out_channels)

    def forward(self, img):
        img_s32 = self.bb(img)
        img_s64 = self.down(img_s32)
        bev_32 = self.s32transformer(img_s32)
        bev_64 = self.s64transformer(img_s64)
        bev = torch.cat([bev_64, bev_32], dim=1)
        if self.is_train:
            return self.lane_head(bev), self.lane_head_2d(img_s32)
        else:
            return self.lane_head(bev)
