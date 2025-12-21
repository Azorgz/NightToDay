# Defines the SegmentorHeadv2. Zero Padding and CSG for positional encoding.
from .modules import ResnetBlock, ASPP, CatersianGrid
from .utilities import get_norm_layer
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )



"""
lightweight_seg_head.py

Lightweight segmentation head for a DINOv3-style encoder (384 feature channels).

Features:
- Accepts encoder output as (B, C, H, W) or (B, N, C) + spatial_size argument.
- Small conv decoder + optional auxiliary classifier.
- Bilinear upsampling to target output size (e.g. original image resolution).
- Parameter groups helper for separate weight decay policies.

Example:
    head = LightweightSegHead(in_channels=384, num_classes=21)
    features = torch.randn(2, 384, 14, 14)            # typical ViT-like map
    out_logits = head(features, output_size=(224, 224))
    print(out_logits.shape)  # -> (2, 21, 224, 224)
"""


class LightweightSegHead(nn.Module):
    """
    Lightweight segmentation head.

    Args:
        in_channels: number of channels from the encoder (e.g. 384 for DINOv3).
        num_classes: segmentation classes.
        hidden_channels: intermediate channel width for the decoder convolutional blocks.
        dropout: dropout probability before final classifier.
        use_aux: if True, produce an auxiliary logits map from an intermediate feature (useful during training).
        upsample_mode: interpolation mode for upsampling ('bilinear' recommended).
        align_corners: align_corners passed to interpolate (only relevant for bilinear).
    """

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 21,
        hidden_channels: int = 256,
        dropout: float = 0.1,
        use_aux: bool = False,
        upsample_mode: str = "bilinear",
        align_corners: Optional[bool] = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.use_aux = use_aux
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        # Reduce / project encoder features
        self.project = ConvBNReLU(in_channels, hidden_channels, kernel_size=1, padding=0)

        # A very light decoder: two conv blocks and a classifier conv
        self.decoder_block1 = ConvBNReLU(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.decoder_block2 = ConvBNReLU(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(hidden_channels // 2, num_classes, kernel_size=1)

        if use_aux:
            # Auxiliary classifier from the projected features (coarser supervision)
            self.aux_classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(
        self,
        features: torch.Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Forward pass.

        Args:
            features: encoder output. Either:
                - (B, C, H, W), or
                - (B, N, C): token sequence (then provide spatial_size=(H, W)).
            spatial_size: required if `features` is (B, N, C) to reshape back to (H, W).
            output_size: final upsample size (height, width). If None, will upsample back to spatial_size inferred from features.
        Returns:
            logits: (B, num_classes, output_h, output_w)
            aux_logits (optional): only if use_aux=True, (B, num_classes, output_h, output_w)
        """
        x = features
        # If token sequence (B, N, C) -> convert to (B, C, H, W)
        if x.dim() == 3:
            if spatial_size is None:
                # try square assumption:
                B, N, C = x.shape
                side = int(math.sqrt(N))
                if side * side != N:
                    raise ValueError(
                        "features is (B, N, C) with N not a perfect square. "
                        "Provide spatial_size=(H, W)."
                    )
                H = W = side
            else:
                H, W = spatial_size
                B, N, C = x.shape
                if N != H * W:
                    raise ValueError(f"Provided spatial_size {spatial_size} does not match N={N}.")
            x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)

        if x.dim() != 4:
            raise ValueError("features must be 4D (B, C, H, W) or 3D (B, N, C)")

        H_in, W_in = x.shape[2], x.shape[3]
        # Project to hidden channels
        proj = self.project(x)  # (B, hidden, H_in, W_in)

        # Decoder
        d = self.decoder_block1(proj)
        d = self.decoder_block2(d)
        d = self.dropout(d)
        logits = self.classifier(d)  # (B, num_classes, H_in, W_in)

        # Determine final output size
        if output_size is None:
            output_size = (H_in, W_in)

        logits_up = F.interpolate(logits, size=output_size, mode=self.upsample_mode, align_corners=self.align_corners)

        if self.use_aux:
            aux = self.aux_classifier(proj)
            aux_up = F.interpolate(aux, size=output_size, mode=self.upsample_mode, align_corners=self.align_corners)
            return logits_up, aux_up

        return logits_up

    def _initialize_weights(self):
        # Kaiming for convs, normal for classifier bias (common pattern)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_param_groups(self, weight_decay: float = 1e-4):
        """
        Return parameter groups for optimizer where biases and batchnorm params typically have no weight decay.
        Use like:
            opt = torch.optim.AdamW(head.get_param_groups() , lr=..., weight_decay=...)
        """
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


class SegmentorHeadv2(nn.Module):
    def __init__(self, input_nc=3, n_layers=4, base_dim=64, num_classes=19, norm_layer='instance', use_bias=False):
        super(SegmentorHeadv2, self).__init__()
        norm_layer = get_norm_layer(norm_layer)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc + 2, base_dim, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(base_dim),
                 nn.PReLU()]

        n_downscaling = 2
        for i in range(n_downscaling):
            mult = 2 ** i
            model += [nn.Conv2d(base_dim * mult, base_dim * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(base_dim * mult * 2),
                      nn.PReLU()]

        mult = 2 ** n_downscaling
        for _ in range(n_layers):
            model += [ResnetBlock(base_dim * mult, norm_layer=norm_layer, use_bias=use_bias, padding_mode='zeros')]

        for i in range(n_downscaling):
            mult = 2 ** (n_downscaling - i)
            model += [nn.ConvTranspose2d(base_dim * mult, int(base_dim * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(base_dim * mult / 2)),
                      nn.PReLU()]

        model += [ASPP(int(base_dim), num_classes, norm_layer)]

        self.model = nn.Sequential(*model)
        self.csg = CatersianGrid()

    def forward(self, x):
        outs, seg_fea = self.model(torch.cat((x, self.csg(x)), 1))
        return outs #, seg_fea