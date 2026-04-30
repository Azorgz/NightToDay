from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_hsv, hsv_to_rgb

from . import ThermalPreprocessConfig
from .CrossRAFT import get_wrapper
from .modules import ResnetBlock
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
            self.res_skip.append(nn.Sequential(*[DropInSwinBlock(base_dim * mult * 2)]*n_enc_layers[i]))
            self.hook.append(len(model) - 2)  # store index of norm for skip connection
        self.res_skip = nn.ModuleList(self.res_skip)
        mult = 2 ** n_downscaling
        # model += [ResnetBlock(base_dim * mult, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)] * n_enc_layers[-1]
        model += [DropInSwinBlock(base_dim * mult)] * n_enc_layers[-1]
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
        # self.vis_preprocess = MonotonicThermalLUT(thermal_preprocessCfg.bins, 1)

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
        out = self.final_conv(x_feat)
        # return out, ir, vis_night  # match input channels
        return out.repeat(1, 3, 1, 1), ir, vis_night

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
        self.scene_idx = None

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
        # y_ = F.interpolate(self.deconv(self.attn(self.conv(F.interpolate(y, (512, 512))))), (y.shape[-2], y.shape[-1]))
        # y = torch.tanh(y + y_)
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
def window_partition(x, window_size):
    """
    Splits the feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reconstructs the feature map from windows.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Standard Multi-Head Self-Attention applied inside a local window. """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """ The Core Swin Transformer Block. """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # Build attention mask for SW-MSA (Shifted Window)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicSwinBlock(SwinBlock):  # Inheriting from the previous SwinBlock
    def __init__(self, *args, **kwargs):
        super().__init__(*args, input_resolution=(64, 64), **kwargs)
        # Remove the static mask from __init__
        self.attn_mask = None
        self.mask_resolution = None  # To track cached mask size

    def build_attention_mask(self, H, W, device):
        """ Dynamically builds the shifted window mask based on current H, W """
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        # 1. Dynamically extract H and W
        B, C, H, W = x.shape  # Assuming the wrapper passes [B, C, H, W]

        # 2. Reshape for transformer [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = x_flat.view(B, H, W, C)

        # 3. Handle dynamic mask for shifted windows
        if self.shift_size > 0:
            # Check if resolution changed; if so, rebuild mask
            if self.mask_resolution != (H, W) or self.attn_mask is None:
                self.attn_mask = self.build_attention_mask(H, W, x.device)
                self.mask_resolution = (H, W)

            shifted_x = torch.roll(x_flat, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            current_mask = self.attn_mask
        else:
            shifted_x = x_flat
            current_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA with dynamic mask
        attn_windows = self.attn(x_windows, mask=current_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_flat = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_flat = shifted_x

        x_flat = x_flat.view(B, H * W, C)

        # FFN
        x_flat = shortcut + x_flat
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Return to [B, C, H, W]
        return x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class DropInSwinBlock(nn.Module):
    """
    Wrapper to easily drop Swin into CNNs.
    Handles the [B, C, H, W] <---> [B, H, W, C] permutations automatically.
    A standard Swin layer requires two blocks: one standard, one shifted.
    """

    def __init__(self, dim=384, num_heads=8, window_size=8):
        super().__init__()
        # Standard Window Attention
        self.block1 = DynamicSwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0)

        # Shifted Window Attention
        self.block2 = DynamicSwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2)

    def forward(self, x):
        """ Expects input of shape [B, C, H, W] """
        B, C, H, W = x.shape

        # Pass through Swin Blocks
        x = self.block1(x)
        x = self.block2(x)

        # Permute back to [B, C, H, W] for CNNs
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x