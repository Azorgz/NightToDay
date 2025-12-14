# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# DepthProEncoder combining patch and image encoders.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import (
    make_vit_16_backbone,
    resize_patch_embed,
    resize_vit,
)
from ..modules import ResnetBlock
from ..utilities import get_norm_layer

REPO_DIR = "/home/godeta/PycharmProjects/dinov3"

ViTModel_t = Literal["dinov3_vits16", "dinov3_convnext_tiny"]
ViTModel = ["dinov3_vits16", "dinov3_convnext_tiny"]


@dataclass
class ViTConfig:
    """Configuration for ViT."""

    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            img_size: int,
            patch_size: int,
            decoder_features: int,
            encoder_feature_layer_ids: List[int] = None,
            encoder_feature_dims: List[int] = None):
        self.in_chans: int = in_chans
        self.embed_dim: int = embed_dim

        self.img_size: int = img_size
        self.patch_size: int = patch_size
        self.decoder_features: int = decoder_features

        # The following 2 parameters are only used by DPT.  See dpt_factory.py.
        self.encoder_feature_layer_ids: List[int] = encoder_feature_layer_ids
        """The layers in the Beit/ViT used to constructs encoder features for DPT."""
        self.encoder_feature_dims: List[int] = encoder_feature_dims
        """The dimension of features of encoder layers from Beit/ViT features for DPT."""


def get_patch_encoder(preset: ViTModel_t, in_channels: int, decoder_features: int = 256) -> ViTConfig:
    return {
        "dinov3_vits16": ViTConfig(
            in_chans=in_channels,
            embed_dim=384,
            encoder_feature_layer_ids=[7, 11],
            encoder_feature_dims=[64, 64, 128, 256],
            decoder_features=decoder_features,
            img_size=224,
            patch_size=16),
        "dinov3_convnext_tiny": ViTConfig(
            in_chans=in_channels,
            embed_dim=768,
            encoder_feature_layer_ids=[2, 4, 12],
            encoder_feature_dims=[96, 192, 384, 768],
            decoder_features=decoder_features,
            img_size=256,
            patch_size=16)}[preset]


def create_vit(
        preset: Literal["dinov3_vits16", "dinov3_convnext_tiny"],
        config: ViTConfig,
) -> nn.Module:
    """Create and load a VIT backbone module.

    Args:
    ----
        preset: The VIT preset to load the pre-defined config.
        use_pretrained: Load pretrained weights if True, default is False.
        checkpoint_uri: Checkpoint to load the wights from.
        use_grad_checkpointing: Use gradient checkpointing.

    Returns:
    -------
        A Torch ViT backbone module.

    """
    img_size = (config.img_size, config.img_size)
    patch_size = (config.patch_size, config.patch_size)

    if preset == "dinov3_convnext_tiny":
        weights = "/home/godeta/PycharmProjects/MyTransform/checkpoints/dinov3_convnext_tiny.pth"
        model = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=weights)
    else:
        weights = "/home/godeta/PycharmProjects/MyTransform/checkpoints/dinov3_vits16.pth"
        model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weights)

    model.image_size = img_size
    model.patch_size = patch_size
    model.forward = model.forward_features
    # model.blocks = model.blocks[: max(config.encoder_feature_layer_ids) + 2]
    return model.train(True)


class MultiResGANEncoder(nn.Module):
    """MultiResGAN Encoder.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.
    """

    def __init__(self, preset: ViTModel_t, input_channels: int = 3, hidden_dim: int = 256, n_enc_layers: int = 4):
        """Initialize MultiResGANEncoder.

        The framework
            1. creates an image pyramid,
            2. generates overlapping patches with a sliding window at each pyramid level,
            3. creates batched encodings via vision transformer backbones,
            4. produces multi-resolution encodings.

        Args:
        ----
            img_size: Backbone image resolution.
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for patches.
            hook_block_ids: Hooks to obtain intermediate features for the patch encoder model.
            decoder_features: Number of feature output in the decoder.
        """
        super().__init__()
        if preset in ViTModel:
            config = get_patch_encoder(preset=preset,
                                       in_channels=input_channels,
                                       decoder_features=hidden_dim)
            dims_encoder: Iterable[int] = config.encoder_feature_dims
            hook_block_ids: Iterable[int] = config.encoder_feature_layer_ids
            patch_encoder = create_vit(preset=preset, config=config)

        else:
            raise KeyError(f"Preset {preset} not found.")

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.hook_block_ids = list(hook_block_ids)

        patch_encoder_embed_dim = patch_encoder.embed_dim
        self.preset = preset

        if not preset == "dinov3_convnext_tiny":
            self.out_size = int(patch_encoder.patch_embed.patches_resolution[0])

        def _create_project_upsample_block(
                dim_in: int, dim_out: int, upsample_layers: int, dim_int: Optional[int] = None) -> nn.Module:
            if dim_int is None:
                dim_int = dim_out
            # Projection.
            blocks = [nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_int,
                kernel_size=1, stride=1, padding=0, bias=False)]  # nn.PReLU(dim_int)

            # Upsampling.
            for i in range(upsample_layers):
                blocks += [nn.ConvTranspose2d(
                    in_channels=dim_int if i == 0 else dim_out,
                    out_channels=dim_out,
                    kernel_size=2, stride=2, padding=0, bias=False)]  #, nn.PReLU(dim_out)

            return nn.Sequential(*blocks)

        if preset == "dinov3_convnext_tiny":
            self.upsample_block_0 = _create_project_upsample_block(
                dim_in=self.dims_encoder[1] * 2, dim_out=self.dims_encoder[1], upsample_layers=1)
            self.upsample_block_1 = _create_project_upsample_block(
                dim_in=self.dims_encoder[2] * 2, dim_out=self.dims_encoder[1], upsample_layers=1)
            self.upsample_block_2 = _create_project_upsample_block(
                dim_in=self.dims_encoder[3], dim_out=self.dims_encoder[2], upsample_layers=1)
            self.fusion_block = _create_project_upsample_block(
                dim_in=self.dims_encoder[0] + self.dims_encoder[1], dim_out=hidden_dim, upsample_layers=0)
            self.res_block = nn.Sequential(
                *[ResnetBlock(hidden_dim, norm_layer=get_norm_layer('instance'), use_bias=True)] * n_enc_layers)
            self.patch_encoder.stages[0].register_forward_hook(self._hook0)
            self.patch_encoder.stages[1].register_forward_hook(self._hook1)
            self.patch_encoder.stages[2].register_forward_hook(self._hook2)
            self.patch_encoder.stages[-1].register_forward_hook(self._hook3)

        else:
            self.upsample_latent0 = _create_project_upsample_block(
                dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[0],
                upsample_layers=2, dim_int=self.dims_encoder[0] * 2)
            self.upsample_latent1 = _create_project_upsample_block(
                dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[1],
                upsample_layers=2, dim_int=self.dims_encoder[1] * 2)
            self.upsample0 = _create_project_upsample_block(
                dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[2],
                upsample_layers=2, dim_int=self.dims_encoder[2] * 2)
            self.upsample1 = _create_project_upsample_block(
                dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[3],
                upsample_layers=2, dim_int=self.dims_encoder[3] * 2)
            self.fusion_upsample = _create_project_upsample_block(
                dim_in=sum(self.dims_encoder), dim_out=hidden_dim, upsample_layers=0, dim_int=hidden_dim)
            # Obtain intermediate outputs of the blocks.
            self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(self._hook0)
            self.patch_encoder.blocks[self.hook_block_ids[1]].register_forward_hook(self._hook1)
            self.patch_encoder.blocks[-1].register_forward_hook(self._hook3)

        # Seg head
        seg = []
        self.train(True)

    def _hook0(self, model, input, output):
        if self.preset == "dinov3_convnext_tiny":
            self.backbone_highres_hook0 = output
        else:
            self.backbone_highres_hook0 = output[0]

    def _hook1(self, model, input, output):
        if self.preset == "dinov3_convnext_tiny":
            self.backbone_highres_hook1 = output
        else:
            self.backbone_highres_hook1 = output[0]

    def _hook2(self, model, input, output):
        if self.preset == "dinov3_convnext_tiny":
            self.backbone_highres_hook2 = output
        else:
            self.backbone_highres_hook2 = output[0]

    def _hook3(self, model, input, output):
        if self.preset == "dinov3_convnext_tiny":
            self.encoding = output
        else:
            self.encoding = output[0]

    @property
    def img_size(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_encoder.patch_embed.img_size[0]

    def _create_image(self, x: torch.Tensor) -> torch.Tensor:
        # Original resolution: BASE_SIZE * 2.
        x1 = F.interpolate(x, size=(x.shape[-2] // 16 * 16, x.shape[-1] // 16 * 16))
        return x1

    def split(self, x: torch.Tensor, overlap_ratio: float = 0.5) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        image_size = x.shape[-1]
        patch_size = 128
        patch_stride = int(patch_size * (1 - overlap_ratio))

        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    @staticmethod
    def merge(x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))
        idx = 0
        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx: batch_size * (idx + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    @staticmethod
    def reshape_feature(embeddings: torch.Tensor, width, height, cls_token_offset=1):
        """Discard class token and reshape 1D feature map to a 2D grid."""
        b, hw, c = embeddings.shape

        # Remove class token.
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]

        # Shape: (batch, height, width, dim) -> (batch, dim, height, width)
        embeddings = embeddings.reshape(b, height, width, c).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, *args) -> list[torch.Tensor]:
        """Encode input at multiple resolutions.
        Args:
        ----
            x (torch.Tensor): Input image.
        Returns:
        -------
            Multi resolution encoded features.
        """
        batch_size, _, h, w = x.shape

        # Step 0: create a sized image.
        x0 = self._create_image(x)

        if self.preset == "dinov3_convnext_tiny":
            x0_encoded = self.patch_encoder(x0)['x_norm_patchtokens']
            x0_features = self.reshape_feature(x0_encoded, h // 32, w // 32, cls_token_offset=0)
            x2_features = self.upsample_block_2(x0_features)
            x1_features = self.upsample_block_1(torch.cat([x2_features, self.backbone_highres_hook2], dim=1))
            x0_features = self.upsample_block_0(torch.cat([x1_features, self.backbone_highres_hook1], dim=1))
            features = self.fusion_block(torch.cat([x0_features, self.backbone_highres_hook0], dim=1))
            # return features
            return self.res_block(features)

        # Step 1: split to create batched overlapped mini-images at the backbone (BeiT/ViT/Dino) resolution.
        # 3x3 @ BASE_SIZE² at the highest resolution (BASE_SIZEx2 x BASE_SIZEx2).
        x0_patches = self.split(x0, overlap_ratio=0.5)
        x1_patches = x0 if len(args) == 0 else args[0]  # No splitting for the lowres image.

        # Step 2: Run the backbone (BeiT) model and get the result of large batch size.
        x0_pyramid_encodings = self.patch_encoder(x0_patches)['x_norm_patchtokens']
        x1_pyramid_encodings = self.patch_encoder(x1_patches)['x_norm_patchtokens']

        # Step 3: merging.
        # Reshape highres latent encoding.
        # 14x14 feature maps by merging 3x3 @ 8x8 patches with overlaps.
        out_size = x0_patches.shape[-1] // 16
        x0_pyramid_encodings = self.reshape_feature(x0_pyramid_encodings, out_size, out_size, cls_token_offset=0)
        x0_features = self.merge(x0_pyramid_encodings, batch_size=batch_size, padding=out_size // 4)

        x1_features = self.reshape_feature(x1_pyramid_encodings, h // 16, w // 16, cls_token_offset=0)
        x_latent0_features = self.reshape_feature(self.backbone_highres_hook0, h // 16, w // 16, cls_token_offset=5)
        x_latent1_features = self.reshape_feature(self.backbone_highres_hook1, h // 16, w // 16, cls_token_offset=5)

        # Upsample feature maps.
        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)
        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)

        return self.fusion_upsample(
            torch.cat([x_latent0_features, x_latent1_features, x0_features, x1_features], dim=1))

    def train(self, mode: bool = True) -> nn.Module:
        """Override the default train() to freeze the backbone."""
        super().train(mode)
        # self.patch_encoder.eval()
        # for param in self.patch_encoder.parameters():
        #     param.requires_grad = False
        return self
