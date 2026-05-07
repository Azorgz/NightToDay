from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_hsv, hsv_to_rgb

from . import ThermalPreprocessConfig
from .CrossRAFT import get_wrapper
from .modules import ResnetBlock, DropInSwinBlock
from .utilities import get_norm_layer

EPS = 1e-6


class U_ResNetFusion(nn.Module):
    """
    Simple ResNet-based fusion module to combine two feature maps.
    """

    def __init__(self, thermal_preprocessCfg: ThermalPreprocessConfig, input_channel=6, hidden_dim=256,
                 n_enc_layers: list = None, dropout=0.25, norm_layer='instance', use_bias=True):
        super(U_ResNetFusion, self).__init__()
        self.input_channel = input_channel
        norm_layer = get_norm_layer(norm_layer)
        n_downscaling = len(n_enc_layers) - 1
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
            self.res_skip.append(nn.Sequential(*[ResnetBlock(base_dim * mult * 2, norm_layer=norm_layer,
                                                           dropout=dropout, use_bias=use_bias)]*n_enc_layers[i]))
            # self.res_skip.append(nn.Sequential(*[[DropInSwinBlock(base_dim * mult)] * n_enc_layers[i]]))
            self.hook.append(len(model) - 2)  # store index of norm for skip connection
        self.res_skip = nn.ModuleList(self.res_skip)
        mult = 2 ** n_downscaling
        for _ in range(n_enc_layers[-1]):
            model += [ResnetBlock(base_dim * mult, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]
            # model += [DropInSwinBlock(base_dim * mult)] * n_enc_layers[-1]
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

        self.layers.append(nn.Conv2d(base_dim, 1,
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

    def forward(self, ir, vis_night, align_first=False, **kwargs):
        ir = self.thermal_preprocess(ir, **kwargs)
        # vis_night = self.vis_preprocess(vis_night, **kwargs)
        if align_first:
            vis_night = self.spatial_aligner(vis_night, ir).detach()
        x_feat = torch.cat([ir, vis_night], dim=1)  # concatenate along channel dim
        for layer in self.encoder:
            x_feat = layer(x_feat)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                hook_output = getattr(self, f'encoder_hook_{self.hook[-(i + 1)]}')
                x_feat = x_feat + self.res_skip[-(i + 1)](hook_output)
            x_feat = layer(x_feat)
        out = self.tanh_n(1)(self.final_conv(x_feat))
        # return out, ir, vis_night  # match input channels
        return out.repeat(1, 3, 1, 1), ir, vis_night

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # self.spatial_aligner.train(False)

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
        self.scene_idx = None
        self.denoiser_module = FastIRDenoiser(in_c=1, base_c=32, num_blocks=3)

    def forward(self, x, *args, p_low=0.5, p_high=100):
        """
        x: IR Tensor of shape (B,1,H,W) or (B,3,H,W)
           assumed normalized to [0,1]
        args: complementary modality for scene selection
        """
        if x.shape[1] == 3:
            HS, x = rgb_to_hsv(x * 0.5 + 0.5).split([2, 1], dim=1)
            x = x * 2 - 1  # normalize to [-1,1]
        else:
            HS = None
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
        y = self.denoiser_module(y)
        y = self.robust_norm(y, p_low=0.0, p_high=100, eps=self.eps) * 2 - 1  # re-normalize to [-1,1]
        if HS is not None:
            y = hsv_to_rgb(torch.cat([HS, y * 0.5 + 0.5], dim=1)) * 2 - 1
        else:
            y = y.repeat(1, 3, 1, 1)
        return y

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


#
# class SceneSelector(nn.Module):
#     def __init__(self,
#                  scene: int = 8,
#                  embed_dim: int = 64):
#         super().__init__()
#         self.scene = scene
#         self.first_conv = nn.Sequential(nn.Conv2d(3, 3, 5, padding=2),
#                                         nn.ReLU(),
#                                         nn.Conv2d(3, 3, 5, padding=2),
#                                         nn.ReLU(),
#                                         nn.Conv2d(3, 1, 5, padding=2),
#                                         nn.ReLU(),
#                                         )
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(1),
#             nn.Linear(256, embed_dim),
#             nn.Linear(embed_dim, scene))
#
#     def forward(self, x, *args):
#         """
#         x: IR Tensor of shape (B,1,H,W) or (B,3,H,W)
#            assumed normalized to [0,1]
#         args: complementary modality for scene selection
#         """
#         if x.shape[1] == 1:
#             x_ = x.repeat(1, 3, 1, 1)
#         elif x.shape[1] == 3:
#             x_ = x
#         else:
#             raise NotImplementedError
#         x_rs = F.interpolate(x_, (256, 256))
#         x_conv = self.first_conv(x_rs)
#         x_patches = self.split(x_conv)
#         scene_logits = self.classifier(x_patches)
#         if args is not None:
#             for arg in args:
#                 if arg.shape[1] == 1:
#                     y = arg.repeat(1, 3, 1, 1)
#                 elif arg.shape[1] == 3:
#                     y = arg
#                 else:
#                     raise NotImplementedError
#                 y_rs = F.interpolate(y, (256, 256))
#                 y_conv = self.first_conv(y_rs)
#                 y_patches = self.split(y_conv)
#                 y_digit = self.classifier(y_patches)
#                 scene_logits = scene_logits + y_digit
#
#         scene_idx = torch.softmax(scene_logits, dim=-1)  # (B, scene)
#         return scene_idx
#
#     def split(self, x: torch.Tensor) -> torch.Tensor:
#         """Split the input into small patches with sliding window."""
#         x_patch_list = []
#         for j in range(16):
#             j0 = j * 16
#             j1 = j0 + 16
#
#             for i in range(16):
#                 i0 = i * 16
#                 i1 = i0 + 16
#                 x_patch_list.append(x[..., j0:j1, i0:i1])
#
#         return torch.cat(x_patch_list, dim=1)
#
#
# # -----------------------------------------------------------
# # Utilities
# # -----------------------------------------------------------
#
# def gradient(x):
#     dx = x[:, :, :, 1:] - x[:, :, :, :-1]
#     dy = x[:, :, 1:, :] - x[:, :, :-1, :]
#     return dx, dy
#
#
# def highlight_mask(vis, threshold=0.95, softness=25.0):
#     # vis in [-1,1] → convert to [0,1]
#     vis = (vis + 1) * 0.5
#     lum = 0.299 * vis[:, 0:1] + \
#           0.587 * vis[:, 1:2] + \
#           0.114 * vis[:, 2:3]
#     col = torch.max(vis, dim=1, keepdim=True)[0] - torch.min(vis, dim=1, keepdim=True)[0]
#     lum = lum * (1 - col)  # boost saturated highlights
#     return torch.sigmoid((lum - threshold) * softness)
#
#
# # -----------------------------------------------------------
# # Attention Blocks
# # -----------------------------------------------------------
#
# class CrossModalAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.query = nn.Conv2d(dim, dim, 1)
#         self.key   = nn.Conv2d(dim, dim, 1)
#         self.value = nn.Conv2d(dim, dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, ir_feat, vis_feat):
#         Q = self.query(ir_feat)
#         K = self.key(vis_feat)
#         V = self.value(vis_feat)
#
#         attn = torch.softmax(
#             torch.sum(Q * K, dim=1, keepdim=True), dim=-1
#         )
#         out = ir_feat + self.gamma * attn * V
#         return out
#
#
# class SelfAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.query = nn.Conv2d(dim, dim, 1)
#         self.key   = nn.Conv2d(dim, dim, 1)
#         self.value = nn.Conv2d(dim, dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, feat):
#         Q = self.query(feat)
#         K = self.key(feat)
#         V = self.value(feat)
#
#         attn = torch.softmax(
#             torch.sum(Q * K, dim=1, keepdim=True), dim=-1
#         )
#         out = feat + self.gamma * attn * V
#         return out


class SimpleGate(nn.Module):
    """
    Replaces standard activations like ReLU/GELU.
    Splits channels in half and multiplies them. Extremely fast.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """ A lightweight channel attention mechanism. """

    def __init__(self, c):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(c, c // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.excite(self.squeeze(x))


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block.
    Uses Depthwise Convolutions and SimpleGates for minimum latency.
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand

        # Spatial feature extraction (Depthwise)
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # Channel feature mixing (Pointwise)
        ffn_channel = c * FFN_Expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = nn.InstanceNorm2d(c)
        self.norm2 = nn.InstanceNorm2d(c)

    def forward(self, x):
        shortcut = x

        # Spatial processing
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = x + shortcut

        # Channel processing
        shortcut = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = x + shortcut

        return x


class detailBlock(nn.Module):
    """ A simple block to extract fine details from the IR image. """

    def __init__(self, c=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv3 = nn.Conv2d(c, 1, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))
        x = self.conv3(x)
        return x


class FastIRDenoiser(nn.Module):
    """ Shallow wide network for fast IR denoising. """

    def __init__(self, in_c=1, base_c=32, num_blocks=3):
        super().__init__()
        self.intro = nn.Conv2d(in_c, base_c, 3, 1, 1)
        self.blocks = nn.Sequential(*[NAFBlock(base_c) for _ in range(num_blocks)])
        self.outro = nn.Conv2d(base_c, in_c, 3, 1, 1)
        self.detail_extractor = detailBlock(c=base_c)
        self.load()

    def forward(self, x):
        shortcut = x
        noise = self.intro(x)
        noise = self.blocks(noise)
        noise = self.outro(noise)
        detail = self.detail_extractor(shortcut)
        x = torch.tanh(shortcut + detail - noise)  # Residual learning + detail enhancement
        return x

    def load(self):
        # Load pretrained weights if available
        try:
            state_dict = torch.load('checkpoints/fast_ir_denoiser_epoch.pth', map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print("Pretrained weights for FastIRDenoiser not found. Using random initialization.")