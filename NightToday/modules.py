import math

import torch
from kornia.color import rgb_to_lab, lab_to_rgb
from torch import nn
import torch.nn.functional as F

from . import SchedulerConfig
from .utilities import power_iteration


# ------------------------ Modules ------------------------ #
class Get_gradmag_gray(nn.Module):
    "To obtain the magnitude values of the gradients at each position."

    def __init__(self):
        super(Get_gradmag_gray, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).to(torch.device(0))
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda(torch.device(0))

    def forward(self, x):
        x_norm = ((x + 1) / 2).mean(dim=1, keepdim=True)
        x0_v = F.conv2d(x_norm, self.weight_v, padding=1)
        x0_h = F.conv2d(x_norm, self.weight_h, padding=1)
        x_gradmagn = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        return x_gradmagn


class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        super().__init__()
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# -------------------- Embedding Modules --------------------


class RGBToLAB_Embed(nn.Module):
    """Convert RGB input to LAB.
    Input: (B,3,H,W) with RGB in [0,1]
    Output: (B,3,H,W) with LAB in approx range.
    """
    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.rgb_to_lab = rgb_to_lab
        self.embed_L = PatchEmbed(1, dim, patch_size)
        self.embed_AB = PatchEmbed(2, dim, patch_size)

    def forward(self, rgb):
        lab = self.rgb_to_lab(rgb)
        L = lab[:, 0:1] / 50 - 1  # normalize L to [-1,1]
        AB = lab[:, 1:3] / 128  # normalize AB to approx [-1,1]
        feat_L = self.embed_L(L)  # (B, D + 2, H', W')
        feat_AB = self.embed_AB(AB)  # (B, D + 2, H', W')
        out = torch.cat([feat_L, feat_AB], dim=1)  # (B, 2D + 4, H', W')
        return out


class LABToRGB_dEmbed(nn.Module):
    """Convert RGB input to LAB.
    Input: (B,3,H,W) with RGB in [-1,1]
    Output: (B,3,H,W) with LAB in approx range.
    """
    def __init__(self, out_ch, dim, patch_size=16):
        super().__init__()
        self.rgb_to_lab = lab_to_rgb
        self.rec_L = nn.Sequential(
                     nn.ConvTranspose2d(dim, out_ch-2, kernel_size=patch_size, stride=patch_size),
                     nn.Tanh())
        self.rec_AB = nn.Sequential(
                     nn.ConvTranspose2d(dim, 2, kernel_size=patch_size, stride=patch_size),
                     nn.Tanh())

    def forward(self, x):
        L = self.rec_L(x)  # (B,out_ch-2,H,W)
        if L.shape[1] == 1:
            L = L
            T = None
        else:
            L = L[:, 1:2]
            T = L[:, 0:1]

        AB = self.rec_AB(x)  # (B,2,H,W)
        L = (L + 1) * 50  # denormalize L to [0,100]
        AB = AB * 128  # denormalize AB to approx [-128,127]
        lab = torch.cat([L, AB], dim=1)
        out = self.rgb_to_lab(lab)
        if T is not None:
            out = torch.cat([T, out], dim=1)
        return out


class PatchEmbed(nn.Module):
    """Simple patch embed via conv. Converts (B,C,H,W) -> (B, D, H//patch_size, W//patch_size)"""

    def __init__(self, in_ch=3, dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = RopePositionalEncoding2D((256//patch_size)**2)
        self.img_size = 256
        self.patch_size = patch_size

    def forward(self, x):
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            self.positional_encoding = RopePositionalEncoding2D((x.shape[-2] // self.patch_size)**2)
            self.img_size = x.shape[-2]
        return self.positional_encoding(self.proj(x))


class PositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for transformer sequence of patches."""

    def __init__(self, dim, height, width):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, dim, height, width))

    def forward(self, x):
        # x: (B, D, H, W)
        return x + self.pe


class RopePositionalEncoding2D(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        pos_y = torch.arange(H, device=x.device).type(self.inv_freq.type())
        pos_x = torch.arange(W, device=x.device).type(self.inv_freq.type())
        sin_inp_y = torch.einsum("i , j -> i j", pos_y, self.inv_freq)
        sin_inp_x = torch.einsum("i , j -> i j", pos_x, self.inv_freq)
        pos_emb_y = torch.cat([sin_inp_y.sin(), sin_inp_y.cos()], dim=-1)
        pos_emb_x = torch.cat([sin_inp_x.sin(), sin_inp_x.cos()], dim=-1)

        pos_emb = torch.zeros((B, H, W, D), device=x.device)
        pos_emb[..., 0::2] = pos_emb_y.unsqueeze(1).unsqueeze(0).repeat(B, 1, W, 1)
        pos_emb[..., 1::2] = pos_emb_x.unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)
        pos_emb = pos_emb.permute(0, 3, 1, 2)
        return x + pos_emb


# region -------------------- Modules --------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, dropout=0.0, use_bias=False, padding_mode='reflect'):
        super().__init__()
        self.act = nn.PReLU()
        conv_block = [] if padding_mode!='reflect' else [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1 if padding_mode!='reflect' else 0,
                      bias=use_bias, padding_mode=padding_mode),
            norm_layer(dim),
            self.act
        ]
        if dropout:
            conv_block += [nn.Dropout(dropout)]
        if padding_mode == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1 if padding_mode != 'reflect' else 0,
                      bias=use_bias, padding_mode=padding_mode),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)



    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                else torch.FloatTensor
            self.context_var = tensor(*(1, self.n_classes, input.shape[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:, domain, :, :] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.0):
        super().__init__()
        mlp_dim = dim * 4
        self.layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=mlp_dim, dropout=dropout)

    def forward(self, *feats):
        if len(feats) == 1:
            inp = feats[0]  # (seq, B, D)
            return self.layer(inp)
        feat_L, feat_AB = feats
        inp = torch.cat([feat_L, feat_AB], dim=0)  # (seq, B, 2*D)
        out = self.layer(inp.flatten(2).permute(2, 0, 1)).reshape(inp.shape)
        return out[:feat_L.size(0)], out[feat_L.size(0):]


class CrossAttentionFusion(nn.Module):
    """Cross-attention that uses features from appearance stream (night) to modulate structure stream (thermal).
    Inputs are patch feature maps (B, D, H, W) for both streams.
    """
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.lambda_factor = nn.Parameter(torch.ones(dim))

    def forward(self, feat_struct, feat_appear):
        # feat_*: (B, D, H, W) -> flatten
        B, D, H, W = feat_struct.shape
        qs = feat_struct.flatten(2).permute(2, 0, 1)  # (seq, B, D)
        ka = feat_appear.flatten(2).permute(2, 0, 1)
        va = ka
        Q = self.q(qs)  # (seq, B, D)
        K = self.k(ka)
        V = self.v(va)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)  # (seq, B, seq)
        fused = torch.matmul(attn, V)  # (seq, B, D)
        fused = fused + qs * self.lambda_factor
        fused = self.proj(fused)
        fused = fused.permute(1, 2, 0).reshape(B, D, H, W)
        return fused


class TransformerDecoderBlock(nn.Module):
    """Basic Transformer decoder layer with self-attention and cross-attention."""
    def __init__(self, dim, n_heads=4, mlp_dim=None, dropout=0.0):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # tgt and memory: (seq, B, D)
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class ConvBlock(nn.Module):
    """Convolution de base : Conv2d -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Bloc résiduel simple"""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
        nn.Conv2d(ch, ch, 3, 1, 1),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, 1, 1),
        nn.BatchNorm2d(ch)
        )
    def forward(self, x):
        return x + self.block(x)

# endregion


# -----------------------------
# Features Fusion Module (FFM)
# -----------------------------
class FeaturesFusionModule(nn.Module):
    """
    Le FFM prend en entrée les features extraites par le générateur
    à partir de l'image visible nocturne et de l'image thermique,
    et les fusionne pour produire une représentation conjointe.
    """
    def __init__(self, in_ch=256, hidden_dim=2048, n_layers=4):
        super().__init__()
        self.embed_struct = ConvBlock(in_ch, hidden_dim, stride=4, padding=0, kernel=4)
        self.embed_guide = ConvBlock(3, hidden_dim, stride=16, padding=0, kernel=16)
        self.positional_encoding = RopePositionalEncoding2D(hidden_dim//2)
        self.cross_attn = nn.ModuleList([CrossAttentionFusion(hidden_dim)]*n_layers)
        self.final_conv = nn.ConvTranspose2d(hidden_dim, in_ch, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(in_ch)

    def forward(self, struct, guide):
        emb_struct = self.embed_struct(struct)
        emb_struct = emb_struct + self.positional_encoding(emb_struct)  # (B, D, H', W')
        emb_guide = self.embed_guide(guide)
        emb_guide = emb_guide + self.positional_encoding(emb_guide)  # (B, D, H', W')
        for layer in self.cross_attn:
            emb_struct = layer(emb_struct, emb_guide)
        struct_delta = self.final_conv(emb_struct)  # (B, in_ch, H, W)
        return self.norm((struct + struct_delta).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=use_bias)
        self.norm = nn.GroupNorm(out_ch//2, out_ch)
        self.act  = nn.PReLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class SmoothLayer(nn.Module):
    def __init__(self, in_ch, use_bias):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, bias=use_bias)
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=use_bias)
        self.act  = nn.Tanh()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(self.down(x))
        return self.act(x)


# -----------------------------
# Color Guidance Module (CGM)
# -----------------------------
class ColorGuidanceModule(nn.Module):
    """
    Le CGM prend une image visible nocturne (ex: 3 canaux) et produit
    une représentation couleur (feature map) qui sera utilisée pour guider
    le générateur. Le CGM est entraîné conjointement.
    """
    def __init__(self, in_ch=3, feat_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
        ConvBlock(in_ch, feat_ch, kernel=7, padding=3),
        nn.AvgPool2d(2),
        ConvBlock(feat_ch, feat_ch*2),
        nn.AvgPool2d(2),
        ConvBlock(feat_ch*2, feat_ch*4)
        )
        # projection finale -> carte de guidage couleur
        self.proj = nn.Conv2d(feat_ch*4, 3, kernel_size=1)

    def forward(self, x):
        f = self.encoder(x)
        color_map = self.proj(f)
        return color_map  # même spatial size réduite


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.PReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_dim, num_classes, norm_layer):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(in_dim, 64, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(in_dim, 64, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(in_dim, 64, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(in_dim, 64, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_dim, 64, 1, stride=1, bias=False),
                                             nn.PReLU())
        self.conv1 = nn.Conv2d(320, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv_1x1_4(self.dropout(x))

        return out, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


class CatersianGrid(nn.Module):
    """Catersian Grid for 2d tensor.
    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-botton corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)

# -------------------- Segmentation Networks --------------------

class SmallUNet(nn.Module):
    """Lightweight U-Net for segmentation used during training only.
    Produces segmentation logits. You can replace or extend this with any segmentation architecture.
    """

    def __init__(self, in_ch=3, out_ch=19, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1),
                                  nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, return_encoder=False):
        e1 = self.enc1(x)
        p = self.pool(e1)
        e2 = self.enc2(p)
        u = self.up(e2)
        d = self.dec1(u + e1)
        logits = self.out(d)
        if return_encoder:
            # return a compact feature map for transfer losses (use e2)
            return logits, e2
        return logits

# -------------------- Loss Scheduler (controls loss weights over epochs) --------------------


class LossScheduler:
    """Simple scheduler that updates lambda weights over epochs.
    Usage:
    sched = LossScheduler({"lambda_color": (start, end, start_epoch, end_epoch), ...})
    sched.step(epoch)
    current = sched.get("lambda_color")
    """

    def __init__(self, schedules: SchedulerConfig, epoch=0):
        self.schedules = {k: v[0] for k, v in schedules.__dict__.items()}
        self.values = {k: v[-1] for k, v in schedules.__dict__.items()}
        self.current = {k: False for k in schedules.__dict__.keys()}
        self.epoch = epoch
        self.actualize()

    def actualize(self):
        self.current = {k: False for k in self.schedules.keys()}
        for k, switches in self.schedules.items():
            for switch in switches:
                if self.epoch >= switch:
                    self.current[k] = not self.current[k]

    def step(self, epoch):
        self.epoch = epoch
        self.actualize()

    def get(self, name):
        if self.current[name] is True:
            return self.values[name]
        else:
            return 0.0


class Sequential(nn.Sequential):
    """Custom Sequential module that can handle multiple inputs and outputs."""

    def forward(self, x):
        for module in self._modules.values():
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x

