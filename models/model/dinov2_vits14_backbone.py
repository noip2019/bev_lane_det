import math
import os

import torch
from torch import nn


class DINOv2PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"DINOv2 input must be divisible by patch size {self.patch_size}, got {(height, width)}"
            )
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, height, width


class DINOv2LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))

    def forward(self, x):
        return x * self.gamma


class DINOv2Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DINOv2Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x):
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        x = attention @ value
        x = x.transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        return x


class DINOv2Block(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0, init_values=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DINOv2Attention(dim=dim, num_heads=num_heads, qkv_bias=True, proj_bias=True)
        self.ls1 = DINOv2LayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DINOv2Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), bias=True)
        self.ls2 = DINOv2LayerScale(dim, init_values=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2ViTS14RegBackbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.patch_size = 14
        self.embed_dim = 384
        self.num_register_tokens = 4
        self.out_channels = self.embed_dim

        self.patch_embed = DINOv2PatchEmbed(patch_size=self.patch_size, in_chans=3, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.blocks = nn.ModuleList(
            [DINOv2Block(dim=self.embed_dim, num_heads=6, mlp_ratio=4.0, init_values=1.0) for _ in range(12)]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def interpolate_pos_encoding(self, x, width, height):
        npatch = x.shape[1] - 1
        total_patches = self.pos_embed.shape[1] - 1
        if npatch == total_patches and width == height:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        cls_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        grid_width = width // self.patch_size
        grid_height = height // self.patch_size
        base_grid = int(math.sqrt(total_patches))
        if base_grid * base_grid != total_patches:
            raise RuntimeError(f"Unexpected DINOv2 pos_embed shape: {self.pos_embed.shape}")

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, base_grid, base_grid, dim).permute(0, 3, 1, 2),
            size=(grid_width, grid_height),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos_embed, patch_pos_embed), dim=1).to(dtype=x.dtype)

    def prepare_tokens(self, x):
        batch_size, _, width, height = x.shape
        x, patch_width, patch_height = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, width, height)
        x = torch.cat(
            (
                x[:, :1],
                self.register_tokens.expand(batch_size, -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )
        return x, patch_width, patch_height

    def forward(self, x):
        x, patch_width, patch_height = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 1 + self.num_register_tokens :]
        x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, patch_width, patch_height)
        return x

    def load_pretrained(self, pretrained_path):
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded DINOv2 ViT-S/14 backbone from {pretrained_path}")
