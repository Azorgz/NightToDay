# -------------------- Generator Networks --------------------

"""Generator architectures for Night-to-Day and Day-to-Night translation.
Includes:
- Transformer-based dual encoder + decoder
- Transformer-based mono encoder + decoder
- ResNet-based encoder + decoder"""

import torch
from torch import nn

from .modules import LABToRGB_dEmbed, TransformerDecoderBlock, CrossAttentionFusion, TransformerEncoderBlock, \
    RGBToLAB_Embed, PatchEmbed, ResnetBlock, FeaturesFusionModule, Upsample, SmoothLayer
from .utilities import get_norm_layer


class TransformerEncoderDual(nn.Module):
    """Generator with two encoders (thermal, night) + cross-attention fusion + decoder.
    Produces day-visible image.
    """
    def __init__(self, dim=256, n_enc_layers=4, patch_size=16, n_heads=4):
        """Dual encoder that processes RGB night image + thermal image in LAB space
        args:
            dim: feature dimension
            patch_size: patch size for embedding
            n_enc_layers: number of transformer encoder layers
            n_heads: number of attention heads
        """
        super().__init__()
        self.patch_size = patch_size
        # Embeddings
        self.embed_T = PatchEmbed(3, dim, patch_size)
        self.embed_N = RGBToLAB_Embed(dim, patch_size)

        # Positional encodings (assume input size multiples are known; we'll support dynamic by conv -> flatten)
        # For safety, do not create fixed pos enc; use layernorm instead for stability
        self.norm_T = nn.LayerNorm(dim)
        self.norm_N = nn.LayerNorm(dim)

        # Transformer stacks
        self.encoder_T = nn.ModuleList([TransformerEncoderBlock(dim, n_heads) for _ in range(n_enc_layers)])
        self.encoder_N = nn.ModuleList([TransformerEncoderBlock(2*dim, n_heads) for _ in range(n_enc_layers)])

        self.fusion_L = CrossAttentionFusion(dim)
        self.fusion_AB = CrossAttentionFusion(dim)

    def forward(self, x, y, *args):
        x_struct, x_appear = x, y  # x_struct: (B,3,H,W), x_appear: (B,3,H,W)
        feat_L = self.embed_T(x_struct)  # (B, D, H', W')
        feat_LAB = self.embed_N(x_appear)  # (B, 2*D + 2, H', W')
        B, D, Hs, Ws = feat_L.shape

        seq_L = feat_L.flatten(2).permute(2, 0, 1)  # (seq, B, D + 2)
        seq_LAB = feat_LAB.flatten(2).permute(2, 0, 1)  # (seq, B, 2D + 2)
        first_part = (self.norm_N(seq_LAB[:, :, :D]) + self.norm_T(seq_L)) / 2
        second_part = seq_LAB[:, :, D + 2:]
        seq_LAB = torch.cat([first_part, second_part], dim=2)

        for layer in self.encoder_T:
            seq_L = layer(seq_L)
        for layer in self.encoder_N:
            seq_LAB = layer(seq_LAB)

        feat_L = seq_L[..., :-2].permute(1, 2, 0).reshape(B, D, Hs, Ws)
        feat_LAB = seq_LAB[..., :-2].permute(1, 2, 0).reshape(B, 2*D, Hs, Ws)

        feat_L = self.fusion_L(feat_L, feat_LAB[:, :D])
        feat_AB = self.fusion_AB(feat_LAB[:, D:], feat_L)
        return feat_L, feat_AB


class TransformerEncoderMono(nn.Module):
    """Generator with two encoders (thermal, night) + cross-attention fusion + decoder.
    Produces day-visible image.
    """
    def __init__(self, dim=256, n_enc_layers=4, patch_size=16, n_heads=4):
        """Single encoder that processes RGB day image in LAB space
        args:
            dim: feature dimension
            patch_size: patch size for embedding
            n_enc_layers: number of transformer encoder layers
            n_heads: number of attention heads
        """
        super().__init__()
        self.patch_size = patch_size
        # Embeddings
        self.LAB_embed = RGBToLAB_Embed(dim, patch_size)

        # Positional encodings (assume input size multiples are known; we'll support dynamic by conv -> flatten)
        # For safety, do not create fixed pos enc; use layernorm instead for stability
        self.norm = nn.LayerNorm(dim)

        # Transformer stacks
        self.encoder_L = nn.ModuleList([TransformerEncoderBlock(dim, n_heads) for _ in range(n_enc_layers)])
        self.encoder_AB = nn.ModuleList([TransformerEncoderBlock(2*dim, n_heads) for _ in range(n_enc_layers)])

        self.fusion_L = CrossAttentionFusion(dim)
        self.fusion_AB = CrossAttentionFusion(dim)

    def forward(self, x):
        # x_struct: (B,1,H,W), x_appear: (B,3,H,W)
        feats = self.LAB_embed(x)  # (B, 2*D, H', W')
        B, D, Hs, Ws = feats.shape
        D = D//2

        seq_L = feats[:, :D].flatten(2).permute(2, 0, 1)  # (seq, B, D)
        seq_LAB = feats.flatten(2).permute(2, 0, 1)

        for layer in self.encoder_L:
            seq_L = layer(seq_L)
        for layer in self.encoder_AB:
            seq_LAB = layer(seq_LAB)

        feat_L = seq_L.permute(1, 2, 0).reshape(B, D, Hs, Ws)
        feat_LAB = seq_LAB.permute(1, 2, 0).reshape(B, 2*D, Hs, Ws)

        feat_L = self.fusion_L(feat_L, feat_LAB[:, :D])
        feat_AB = self.fusion_AB(feat_LAB[:, D:], feat_L)
        return feat_L, feat_AB


class TransformerDecoder(nn.Module):
    """Decoder with transformer decoder blocks and final reconstruction layer.
    Produce either LAB day image or LAB night Image with fused thermal and visible L layer
    """

    def __init__(self, out_ch=3, dim=256, patch_size=4, n_dec_layers=4, n_heads=4):
        super().__init__()
        # --- Decoder stack ---
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(dim, n_heads) for _ in range(n_dec_layers)
        ])

        # Final reconstruction
        self.to_image = LABToRGB_dEmbed(out_ch, dim, patch_size)

    def forward(self, feats):
        # --- Transformer Decoder ---
        feat_L, feat_AB = feats  # (B, D, H, W)
        B, D, Hs, Ws = feat_L.shape
        seq_L = feat_L.flatten(2).permute(2, 0, 1)
        seq_AB = feat_AB.flatten(2).permute(2, 0, 1)
        for dec in self.decoder:
            seq_L = dec(seq_L, seq_AB)

        out = seq_L.permute(1, 2, 0).reshape(B, D, Hs, Ws)
        out = self.to_image(out)
        return out


class ResnetGenEncoder(nn.Module):
    def __init__(self, input_channel, hidden_dim=256, n_enc_layers=4, dropout=0, n_downscaling=2,
                 norm_layer='instance', use_bias=True):
        super(ResnetGenEncoder, self).__init__()
        norm_layer = get_norm_layer(norm_layer)
        base_dim = hidden_dim // (2 ** n_downscaling)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channel, base_dim, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(base_dim),
                 nn.PReLU()]
        for i in range(n_downscaling):
            mult = 2 ** i
            model += [nn.Conv2d(base_dim * mult, base_dim * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(base_dim * mult * 2),
                      nn.PReLU()]
        mult = 2 ** n_downscaling
        for _ in range(n_enc_layers):
            model += [ResnetBlock(base_dim * mult, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, x, *args):
        x_feat = self.model(x)
        return x_feat


class ResnetGenDecoder(nn.Module):
    def __init__(self, output_channel, hidden_dim=256, n_dec_layers=5, dropout=0, n_downscaling=2
                 , norm_layer='instance', use_bias=True):
        super(ResnetGenDecoder, self).__init__()
        norm_layer = get_norm_layer(norm_layer)
        model = []
        mult = 2 ** n_downscaling
        base_dim = hidden_dim // mult

        for _ in range(n_dec_layers):
            model += [ResnetBlock(hidden_dim, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]

        for i in range(n_downscaling):
            mult = 2 ** (n_downscaling - i)
            dim_out = int(base_dim * mult / 2)

            # model += [Upsample(base_dim * mult, int(base_dim * mult / 2), use_bias=use_bias)]
            model += [nn.ConvTranspose2d(base_dim * mult, int(base_dim * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      nn.GroupNorm(base_dim//2, dim_out),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(base_dim, output_channel, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

