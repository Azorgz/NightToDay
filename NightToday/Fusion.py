from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.filters import median_blur, bilateral_blur
from torch import conv2d
from torch.nn import ModuleList
from torchvision.transforms.functional import gaussian_blur

from . import ThermalPreprocessConfig
from .CrossRAFT import get_wrapper
from .modules import ResnetBlock
from .utilities import get_norm_layer

EPS = 1e-6


# ---------------------------------------------------------
#  RoPE Positional Encoding Block (simple version)
# ---------------------------------------------------------
class RoPE(nn.Module):
    """
    Simple RoPE positional embedding:
    Combines additive + multiplicative sinusoidal encodings.
    """

    def __init__(self,
                 dim: int,
                 height: int,
                 width: int):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width

        self.dim = dim
        self.height = height
        self.width = width
        self._initialized()

    def _initialized(self):
        self.alpha = nn.Parameter(torch.ones(self.dim))
        self.beta = nn.Parameter(torch.ones(self.dim))

        # standard 2D sin/cos base
        y, x = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing="ij")
        coords = torch.stack([x, y], dim=0).float()  # (2, H, W)
        self.register_buffer("coords", coords)

        # frequency bands:
        half = self.dim // 4
        freq = torch.exp(-torch.arange(half) / half * torch.log(torch.tensor(10000.0)))
        self.register_buffer("freq", freq)

    def forward(self, dim, H, W) -> torch.Tensor:
        """
        Returns positional embedding of shape (H*W, dim)
        """
        if dim != self.dim or H != self.height or W != self.width:
            self.dim = dim
            self.height = H
            self.width = W
            self._initialized()
        x = self.coords[0].reshape(-1)[:, None]  # flatten
        y = self.coords[1].reshape(-1)[:, None]

        # apply frequencies
        sinx = torch.sin(x * self.freq)
        cosx = torch.cos(x * self.freq)
        siny = torch.sin(y * self.freq)
        cosy = torch.cos(y * self.freq)

        base = torch.cat([sinx, cosx, siny, cosy], dim=-1)

        # SaPE²: additive + multiplicative modulation
        pe = self.alpha * base + self.beta * (base ** 2)

        return pe  # (H*W, dim)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, D, Hh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N = H'*W', D)
        return x, (Hh, Ww)


class PatchUnEmbed(nn.Module):
    def __init__(self, out_ch: int, embed_dim: int, patch_size: Tuple[int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.out_ch = out_ch
        self.embed_dim = embed_dim
        upscale_level = int(torch.log2(torch.tensor(patch_size[0])) - 1)  # assume square patches for simplicity
        base_dim = max(256, 2 ** upscale_level)
        self.reg_dim = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(embed_dim, base_dim, kernel_size=3, padding=1))
        assert upscale_level <= 8, "Upscale level too high"
        upscale = []
        factor = 1
        for i in range(int(upscale_level)):
            factor = 2 ** i
            upscale += nn.Sequential(
                nn.ConvTranspose2d(base_dim // factor, base_dim // (factor * 2), kernel_size=4, stride=2, padding=1),
                nn.Conv2d(base_dim // (factor * 2), base_dim // (factor * 2), kernel_size=3, padding=1,
                          padding_mode='reflect'),
                nn.GroupNorm(base_dim // (2 ** upscale_level), base_dim // (factor * 2), affine=False),
                nn.PReLU(base_dim // (factor * 2)))
        self.upscale = nn.Sequential(*upscale)
        self.last_conv = nn.Sequential(nn.Conv2d(base_dim // (factor * 2), out_ch, kernel_size=7, padding=3),
                                       nn.Tanh())

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        # x: (B, N, D), grid_size = (Hh, Ww)
        B, N, D = x.shape
        Hh, Ww = grid_size
        x = x.transpose(1, 2).reshape(B, D, Hh, Ww)
        x = self.reg_dim(x)
        x = self.upscale(x)  # (B, base_dim//(factor*2), H, W)
        x = self.last_conv(x)  # (B, out_ch, H, W)
        return x


class TransformerBlockCond(nn.Module):
    """
    A single block: self-attention on x, then cross-attention to cond, then optional FFN.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        if mlp_hidden is None:
            mlp_hidden = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), cond: (B, M, D)
        residual = x
        x1, _ = self.self_attn(x, x, x)
        x1 = self.norm1(residual + self.dropout(x1))

        residual = x1
        x2, _ = self.cross_attn(x1, cond, cond)
        x2 = self.norm2(residual + self.dropout(x2))

        residual = x2
        x3 = self.mlp(x2)
        x3 = self.norm3(residual + self.dropout(x3))
        return x + x3 * 0.1  # scaled residual


class SimpleCondViT(nn.Module):
    def __init__(
            self,
            in_ch: int,
            cond_ch: int = 3,
            embed_dim: int = 256,
            patch_size: Tuple[int, int] = (16, 16),
            num_blocks: int = 4,
            num_heads: int = 4,
            mlp_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size[0])
        self.cond_embed = PatchEmbed(cond_ch, embed_dim, patch_size[1])
        self.pos_enc = RoPE(dim=embed_dim, height=256, width=16)
        self.blocks = nn.ModuleList([
            TransformerBlockCond(embed_dim, num_heads, mlp_hidden)
            for _ in range(num_blocks)
        ])
        self.unembed = PatchUnEmbed(in_ch, embed_dim, patch_size)
        self.act = nn.Tanh()
        self.channel_mixer = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, He, We)
        cond_img: (B, cond_ch, Hc, Wc)
        Returns:
          out: (B, C_in, He, We)  (roughly same resolution as input, but conv transpose might produce slight mismatch)
        """
        x_tok, (Hh, Ww) = self.patch_embed(x)  # (B, N, D)
        cond_tok, _ = self.cond_embed(cond_img)  # (B, M, D)
        # apply positional embedding on input tokens
        B, N, D = x_tok.shape
        bias = self.pos_enc(dim=D, H=Hh, W=Ww).to(x_tok.device)  # (B, N, D)

        # pass through transformer blocks
        x2 = x_tok + bias.unsqueeze(0).expand(B, N, D)  # simple addition of bias as positional encoding
        for blk in self.blocks:
            x2 = blk(x2, cond_tok)

        # reproject to feature map
        out = self.unembed(x2, (Hh, Ww))
        out = self.act(self.channel_mixer(out + x))  # residual
        return out.repeat(1, x.shape[1], 1, 1)  # match input channels


# # ---------------------------------------------------------
# #   Transformer Block with Self + Cross Attention
# # ---------------------------------------------------------
# class CustomTransformerBlock(nn.Module):
#     def __init__(self, dim, heads=8, mlp_ratio=4.0):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
#         self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim),
#         )
#
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#
#     def forward(self, x, cond):
#         # x:   (B, HW, D)
#         # cond:(B, HW, D)
#
#         # self-attention
#         x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#
#         # cross-attention: queries=x, keys/values=cond
#         x = x + self.cross_attn(self.norm2(x), self.norm2(cond), self.norm2(cond))[0]
#
#         # feed-forward
#         x = x + self.mlp(self.norm3(x))
#         return x
#
#
# # ---------------------------------------------------------
# #   Main Model
# # ---------------------------------------------------------
# class SaPE2Transformer(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             cond_channels=3,
#             embed_dim=256,
#             num_blocks=6,
#             patch_size=16,
#             emb_type='SaPE2',
#     ):
#         super().__init__()
#
#         self.patch = patch_size
#         self.dim = embed_dim
#
#         # Embedding = patchify with Conv2d
#         self.input_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size // 4, stride=patch_size // 4)
#
#         self.cond_embed = nn.Conv2d(cond_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#         # De-embedding = unpatchify with TransposeConv2d
#         self.output_proj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size // 4,
#                                               stride=patch_size // 4)
#
#         # Positional embedding will be created at runtime (depends on input size)
#         self.positional_embedding = lambda x, h, w: SaPE2(dim=embed_dim) if emb_type == 'SaPE2' else RoPE()  # placeholder
#
#         #
#         self.blocks = nn.ModuleList([CustomTransformerBlock(embed_dim) for _ in range(num_blocks)])
#
#     def forward(self, x, cond_img):
#         B, C, H, W = x.shape
#
#         # ---- Patch Embedding ----
#         x = self.input_embed(x)  # (B, D, H/P, W/P)
#         cond = self.cond_embed(cond_img)
#
#         Hp, Wp = x.shape[-2:]
#
#         # flatten to tokens
#         x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, D)
#         cond = cond.flatten(2).transpose(1, 2)
#
#         # ---- Generate positional embedding dynamically ----
#         pe = SaPE2(self.dim, Hp, Wp)().to(x.device)
#         x = x + pe.unsqueeze(0)
#         cond = cond + pe.unsqueeze(0)
#
#         # ---- Transformer blocks ----
#         for blk in self.blocks:
#             x = blk(x, cond)
#
#         # ---- De-embed ----
#         x = x.transpose(1, 2).view(B, self.dim, Hp, Wp)
#         x = self.output_proj(x)  # (B, C, H, W)
#
#         return x

class U_ResNetFusion(nn.Module):
    """
    Simple ResNet-based fusion module to combine two feature maps.
    """

    def __init__(self, thermal_preprocessCfg: ThermalPreprocessConfig, input_channel=6, hidden_dim=256,
                 n_enc_layers=4, dropout=0.25, n_downscaling=2, norm_layer='instance', use_bias=True):
        super(U_ResNetFusion, self).__init__()
        self.input_channel = input_channel
        norm_layer = get_norm_layer(norm_layer)
        base_dim = hidden_dim // (2 ** n_downscaling)
        self.hook = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channel, base_dim, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(base_dim),
                 nn.ReLU()]
        self.res_skip = []
        self.count_skip = 0
        for i in range(n_downscaling):
            mult = 2 ** i
            model += [
                nn.Conv2d(base_dim * mult, base_dim * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(base_dim * mult * 2),
                nn.ReLU()]
            self.hook.append(len(model) - 2)  # store index of norm for skip connection
            self.res_skip.append(nn.Sequential(ResnetBlock(base_dim * mult * 2, norm_layer=norm_layer,
                                                           dropout=dropout, use_bias=use_bias)))
        self.res_skip = nn.ModuleList(self.res_skip)
        mult = 2 ** n_downscaling
        for _ in range(n_enc_layers):
            model += [ResnetBlock(base_dim * mult, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]
        self.encoder = nn.ModuleList(model)
        for i, idx in enumerate(self.hook):
            self.encoder[idx].register_forward_hook(lambda model, input, output: self._register_hook(output))
        self.layers = nn.ModuleList([])
        for i in range(n_downscaling):
            mult = 2 ** (n_downscaling - i)
            self.layers.append(nn.Sequential(nn.ConvTranspose2d(base_dim * mult, int(base_dim * mult // 2),
                                                                kernel_size=4, stride=2,
                                                                padding=1, output_padding=0,
                                                                bias=use_bias), self.tanh_n(mult * 2, mult)))

        self.layers.append(nn.Conv2d(int(base_dim * mult // 2), 1,
                                     kernel_size=7, padding=3, padding_mode='reflect'))
        self.final_conv = nn.Sequential(nn.Conv2d(1, 1,
                                                  kernel_size=7, padding=3, padding_mode='reflect'), nn.Tanh())
        self.spatial_aligner = get_wrapper('vis2ir')
        self.thermal_preprocess = MonotonicThermalLUT(thermal_preprocessCfg.bins,
                                                      thermal_preprocessCfg.scene)

    def _register_hook(self, output):
        if len(self.hook) > self.count_skip:
            idx = self.hook[self.count_skip]
            self.count_skip += 1
            setattr(self, f'encoder_hook_{idx}', output)

        else:
            self.count_skip = 0
            self._register_hook(output)

    def tanh_n(self, n1=1.0, n2=None):
        class tanh_n(nn.Module):
            def __init__(self, n_1, n_2):
                super().__init__()
                self.n1 = n_1
                self.n2 = n_2

            def forward(self, x):
                return nn.Tanh()(x / self.n1) * self.n2

        return tanh_n(n1, n2 or n1)

    def forward(self, ir, vis_night, align_first=True, **kwargs):
        ir = self.thermal_preprocess(ir, **kwargs)
        if align_first:
            vis_night = self.spatial_aligner(vis_night, ir).detach()
        # AB = rgb_to_lab(vis_night)[:, 1:] / 128  # extract AB channels and normalize to [-1,1]
        x_feat = torch.cat([ir, vis_night], dim=1)  # concatenate along channel dim
        for layer in self.encoder:
            x_feat = layer(x_feat)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                hook_output = getattr(self, f'encoder_hook_{self.hook[-(i + 1)]}')
                x_feat = x_feat + self.res_skip[-(i + 1)](hook_output)
            x_feat = layer(x_feat)
        out = self.tanh_n(1)(self.final_conv(x_feat))
        # out = torch.cat([out, AB], dim=1)  # combine with AB channels
        # return out, ir, vis_night  # match input channels

        # filtered_ir = torch.sqrt(bilateral_blur(ir*0.5+0.5, (3, 3), 0.1, (1.5, 1.5)) + 1e-6)
        # out = torch.sqrt((self.thermal_postprocess(self.tanh_n(1)(out), p_low=0, p_high=100)*0.5+0.5) * filtered_ir + 1e-6) * 2 - 1
        # out = self.thermal_postprocess(self.tanh_n(1)(out), p_low=0, p_high=100)
        # return out, ir, vis_night  # match input channels
        return self.tanh_n(1)(out).repeat(1, 3, 1, 1), ir, vis_night, None  # match input channels

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.spatial_aligner.train(False)

    @property
    def scene_idx(self):
        return self.thermal_preprocess.scene_idx


class MonotonicThermalLUT(nn.Module):
    """
    Learnable monotonic LUT for thermal re-binning.
    Identity-initialized.
    """

    def __init__(self, bins: int = 2048, scene: int = 8, eps=1e-8):
        super().__init__()
        self.bins = bins
        self.scene = scene
        self.eps = eps

        # Identity initialization:
        # softplus(delta) ≈ constant → cumsum ≈ linear ramp
        init_delta = torch.ones(scene, bins) * 1.0
        self.delta = nn.Parameter(init_delta)
        self.scene_selection = SceneSelector()
        self.scene_idx = None

    def forward(self, x, *args, p_low=2., p_high=100, epoch=0):
        """
        x: IR Tensor of shape (B,1,H,W) or (B,3,H,W)
           assumed normalized to [0,1]
        args: complementary modality for scene selection
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # convert to grayscale
        # Robust normalization to [0,1]
        x = self.robust_norm(x, p_low=p_low, p_high=p_high, eps=self.eps)
        self.scene_idx = torch.ones([x.shape[0], self.scene], device=x.device) / self.scene
        # self.scene_idx = self.naive_scene_selection(x)
        # Build monotonic LUT
        increments = F.softplus(torch.mm(self.scene_idx, self.delta)) + self.eps
        luts = torch.cumsum(increments, dim=1)
        luts = luts / (luts[:, -1:] + self.eps) * 2 - 1  # normalize to [-1,1]

        # Apply LUT
        y = []
        for i, lut in enumerate(luts):
            idx = (x[i][None] * (self.bins - 1)).long().clamp(0, self.bins - 1)
            y.append(lut[idx])

        y = torch.cat(y, 0)
        return y.repeat(1, 3, 1, 1)

    def naive_scene_selection(self, x):
        x_mean_t = x[:, :, ::2].mean(dim=[1, 2, 3])
        x_mean_b = x[:, :, 2::].mean(dim=[1, 2, 3])
        x_std_t = x[:, :, ::2].std(dim=[1, 2, 3])
        x_std = x[:, :, ].std(dim=[1, 2, 3])
        low_lum_t = (x[:, :, ::2] < -0.90).sum(dim=[1, 2, 3]) / torch.tensor([x.shape[0], torch.prod(torch.tensor(x.shape[-2:]))//2])
        cond1 = x_mean_b > x_mean_t * 2
        cond2 = x_std_t > x_std
        cond3 = low_lum_t > 0.1
        out = torch.zeros([x.shape[0], self.scene], device=x.device)
        idx = cond1 + 2 * cond2 + 4 * cond3
        out[:, idx] = 1.
        return out

    def robust_norm(self, x, p_low=0.5, p_high=99.5, eps=1e-6):
        """
        x: (B,1,H,W) or (B,H,W)
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)
        lo = torch.quantile(x_flat, p_low / 100.0, dim=1, keepdim=True)
        hi = torch.quantile(x_flat, p_high / 100.0, dim=1, keepdim=True)
        lo = lo.view(B, 1, 1, 1)
        hi = hi.view(B, 1, 1, 1)

        return ((x - lo) / (hi - lo + eps)).clamp(0, 1)


class SceneSelector(nn.Module):
    def __init__(self,
                 scene: int = 8,
                 embed_dim: int = 64):
        super().__init__()
        self.scene = scene
        self.first_conv = nn.Sequential(nn.Conv2d(3, 3, 5, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(3, 3, 5, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(3, 1, 5, padding=2),
                                        nn.ReLU(),
                                        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(256, embed_dim),
            nn.Linear(embed_dim, scene))

    def forward(self, x, *args):
        """
        x: IR Tensor of shape (B,1,H,W) or (B,3,H,W)
           assumed normalized to [0,1]
        args: complementary modality for scene selection
        """
        if x.shape[1] == 1:
            x_ = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            x_ = x
        else:
            raise NotImplementedError
        x_rs = F.interpolate(x_, (256, 256))
        x_conv = self.first_conv(x_rs)
        x_patches = self.split(x_conv)
        scene_logits = self.classifier(x_patches)
        if args is not None:
            for arg in args:
                if arg.shape[1] == 1:
                    y = arg.repeat(1, 3, 1, 1)
                elif arg.shape[1] == 3:
                    y = arg
                else:
                    raise NotImplementedError
                y_rs = F.interpolate(y, (256, 256))
                y_conv = self.first_conv(y_rs)
                y_patches = self.split(y_conv)
                y_digit = self.classifier(y_patches)
                scene_logits = scene_logits + y_digit

        scene_idx = torch.softmax(scene_logits, dim=-1)  # (B, scene)
        return scene_idx

    def split(self, x: torch.Tensor) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        x_patch_list = []
        for j in range(16):
            j0 = j * 16
            j1 = j0 + 16

            for i in range(16):
                i0 = i * 16
                i1 = i0 + 16
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=1)


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def gradient(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def highlight_mask(vis, threshold=0.9, softness=15.0):
    # vis in [-1,1] → convert to [0,1]
    vis = (vis + 1) * 0.5
    lum = 0.299 * vis[:, 0:1] + \
          0.587 * vis[:, 1:2] + \
          0.114 * vis[:, 2:3]
    return torch.sigmoid((lum - threshold) * softness)


# -----------------------------------------------------------
# Cross Modal Attention Block
# -----------------------------------------------------------

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key   = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, ir_feat, vis_feat):
        Q = self.query(ir_feat)
        K = self.key(vis_feat)
        V = self.value(vis_feat)

        attn = torch.softmax(
            torch.sum(Q * K, dim=1, keepdim=True), dim=-1
        )
        out = ir_feat + self.gamma * attn * V
        return out


# -----------------------------------------------------------
# Illumination Aware Fusion Network
# -----------------------------------------------------------

class IlluminationAwareFusion(nn.Module):
    """
    Unpaired illumination-aware fusion module.
    Produces:
        fake_ir, reflectance, illumination
    """

    def __init__(self, thermal_preprocessCfg: ThermalPreprocessConfig, input_channels=6, hidden_dim=256,
                 n_enc_layers=4, dropout=0.25, n_downscaling=2):
        super().__init__()
        base_dim = hidden_dim // (2 ** n_downscaling)
        # Encoder
        enc1 = nn.Sequential(
            nn.Conv2d(input_channels + 1, base_dim, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.enc = [enc1]
        mult = 1
        for i in range(n_downscaling):
            mult = 2 ** (i + 1)
            self.enc.append(nn.Sequential(
                nn.Conv2d(base_dim * mult // 2, base_dim * mult, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(base_dim * mult),
                nn.ReLU(inplace=True)
            ))
        self.enc = nn.Sequential(*self.enc)

        # Cross modal attention at bottleneck
        self.attn = CrossModalAttention(base_dim * mult)

        # Decoder
        self.dec = []
        for i in range(n_downscaling):
            mult = 2 ** (n_downscaling - i)
            self.dec.append(nn.Sequential(
                nn.ConvTranspose2d(base_dim * mult, base_dim * mult // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(base_dim * mult // 2),
                nn.ReLU(inplace=True)
            ))
        self.dec = nn.Sequential(*self.dec)

        # Dual heads
        self.reflectance_head = nn.Sequential(
            nn.Conv2d(base_dim, 1, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

        self.illumination_head = nn.Sequential(
            nn.Conv2d(base_dim, 1, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
        self.spatial_aligner = get_wrapper('vis2ir')
        self.thermal_preprocess = MonotonicThermalLUT(thermal_preprocessCfg.bins,
                                                      thermal_preprocessCfg.scene)

    # -------------------------------------------------------

    def forward(self, ir, vis_night, align_first=True, **kwargs):
        """
        ir, vis_night ∈ [-1,1]
        """

        # Highlight awareness
        ir = self.thermal_preprocess(ir, **kwargs)
        if align_first:
            vis_night = self.spatial_aligner(vis_night, ir).detach()
        mask = highlight_mask(vis_night)

        # Concatenate mask explicitly
        x = torch.cat([ir, vis_night, mask], dim=1)

        feat = self.enc(x)
        # Split IR and VIS features for attention
        ir_feat = feat
        vis_feat = feat.detach()  # prevent visible highlight domination

        bottleneck = self.attn(ir_feat, vis_feat)

        decoded_feat = self.dec(bottleneck)

        R = self.reflectance_head(decoded_feat) * 0.5 + 1
        I = self.illumination_head(decoded_feat) * 0.5 + 1

        fake_ir = R * I - 1.25  # combine reflectance and illumination, scale to [-1,1]

        return fake_ir.repeat(1, 3, 1, 1), ir, vis_night, I, R, mask