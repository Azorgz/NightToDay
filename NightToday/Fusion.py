import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from kornia.filters import median_blur, bilateral_blur
from kornia.geometry import PyrDown, PyrUp
from kornia.morphology import dilation, erosion
from torch import conv2d
from torch.nn import AvgPool2d

from NightToday import ThermalPreprocessConfig
from NightToday.CrossRAFT import get_wrapper
from NightToday.modules import ResnetBlock
from NightToday.utilities import get_norm_layer

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


# ---------------------------------------------------------
#  SaPE² Positional Encoding Block (simple version)
# ---------------------------------------------------------
class SaPE2(nn.Module):
    """
    SaPE^2 approximate implementation (PyTorch).
    Based on: "A 2D Semantic-Aware Position Encoding for Vision Transformers"
    (Xi Chen et al., arXiv:2505.09466v1), §3 (equations (9)-(16)).

    Inputs:
      - x: token features, either shape (B, N, C) or (B, H, W, C).
      - expects H, W to be provided when x is flat (N = H*W).
    Outputs:
      - bias: attention bias tensor of shape (B, N, N). Add this to attention logits.

    Key hyperparams:
      - dim: input token dimension C
      - pos_dim: dimension of positional embeddings per axis (so final pos vector is pos_dim*2)
      - max_pos: maximum integer position index for learnable position table (>= expected cumulative sums)
      - mode: 'Q' or 'K' (paper allows applying SaPE^2 to Q or K; implementation supports both; behavior identical here)
      - normalize_gates: whether to normalize gate weights when aggregating (True recommended)

    Note: This implementation follows the paper's algorithmic description. Some low-level choices
    (exact normalization, whether bias = -||p_i - p_j|| or -||p_i - p_j||^2) are implemented here
    as reasonable defaults; they can be changed easily.
    """

    def __init__(self,
                 dim: int,
                 pos_dim: int = 64,
                 max_pos: int = 64,
                 normalize_gates: bool = True,
                 apply_sigmoid_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.max_pos = max_pos
        self.normalize_gates = normalize_gates
        self.scale = apply_sigmoid_scale

        # projections used to compute semantic gates along each axis (shared across axes)
        # paper indicates gate from inner product of local projections; we follow that.
        self.q_proj = nn.Linear(dim, pos_dim, bias=False)
        self.k_proj = nn.Linear(dim, pos_dim, bias=False)

        # learnable integer position embeddings for positions 0..max_pos (per axis)
        # we store one table and reuse it for both axes (paper interpolates integer embeddings).
        self.pos_table = nn.Parameter(torch.randn(max_pos + 1, pos_dim))  # +1 to allow floor+1

        # small linear to fuse two axis vectors into same output dim if desired (optional)
        # Here we keep final position vector dim = pos_dim * 2 (concat axis vectors).
        # Optionally you can project down to match model dim.
        # self.fuse = nn.Linear(pos_dim * 2, dim)  # optional

    def forward(self,
                x: torch.Tensor,
                H: Optional[int] = None,
                W: Optional[int] = None) -> torch.Tensor:

        """
        x: (B, N, C) or (B, H, W, C)
        returns:
          bias: (B, N, N) float tensor. Add this to attention logits.
        """
        orig_shape = x.shape
        if x.ndim == 4:
            # (B, H, W, C) -> flatten to (B, N, C)
            B, H, W, C = x.shape
            N = H * W
            x_flat = x.view(B, N, C)
        elif x.ndim == 3:
            B, N, C = x.shape
            assert H is not None and W is not None and H * W == N, "Provide H and W when passing (B,N,C)"
            x_flat = x
        else:
            raise ValueError("x must be shape (B,H,W,C) or (B,N,C)")

        # reshape to grid for axis operations
        x_grid = x_flat.view(B, H, W, C)  # (B, H, W, C)

        # compute projections for gates
        q = self.q_proj(x_grid)  # (B, H, W, pos_dim)
        k = self.k_proj(x_grid)  # (B, H, W, pos_dim)

        # We'll compute gates along horizontal axis (rows) and vertical axis (cols) separately,
        # then produce per-patch axis-specific position vectors and concatenate.

        # ---------- horizontal axis (rows): length W per row ----------
        # compute pairwise dot products along width for each row:
        # Prepare shapes: (B, H, W, d). We'll compute (B, H, W, W) = dot over last dim.
        d = q.shape[-1]
        scale = (d ** 0.5)
        # compute pairwise dot along width: efficient using einsum
        # gates_logits_h[b,h,i,j] = <q[b,h,i,:], k[b,h,j,:]> / scale
        gates_logits_h = torch.einsum("bhid,bhjd->bhij", q, k) / (scale + EPS)  # (B, H, W, W)

        # apply sigmoid to get gates in (0,1). Paper uses sigmoid gating.
        gates_h = torch.sigmoid(self.scale * gates_logits_h)  # (B, H, W, W)

        # compute prefix sums along width: prefix[b,h,j] = sum_{t=0..j} gates_h[b,h,ref, t]
        # For each *reference* position i, we want position values relative to every j.
        # We implement paper's "sum gate values" idea by computing cumulative sums along axis j index.
        prefix_h = gates_h.cumsum(dim=-1)  # (B, H, W, W)

        # position value r_h[i->j] = prefix_h[..., j] - prefix_h[..., i]  (sum gates between i+1..j)
        # This yields continuous relative position (can be negative if j < i).
        # Compute difference for all i,j pairs:
        # pref_j = prefix_h.unsqueeze(3)  # (B,H,W,1,W)
        # pref_i = prefix_h.unsqueeze(2)  # (B,H,1,W,W)
        # r_h = prefix[..., j] - prefix[..., i] along the indexing consistent dimension
        # We'll compute r_h[b,h,i,j] = prefix_h[b,h,j] - prefix_h[b,h,i]
        # prefix_h_trans = prefix_h  # (B,H,W,W) where index -1 is j; need broadcast
        # r_h = prefix_h[:, :, None, :, :]  # (B, H, 1, W, W)?? simpler compute via outer diff
        # simpler: compute prefix_h_at_j and prefix_h_at_i
        # prefix_at_j = prefix_h  # shape (B,H,W,W) with last dim j
        # prefix_at_i = prefix_h.transpose(-2, -1)  # now last dim corresponds to i
        # Actually easier: use broadcast: r_h[b,h,i,j] = prefix_h[b,h,j] - prefix_h[b,h,i]
        r_h = prefix_h.unsqueeze(-2) - prefix_h.unsqueeze(-1)  # (B,H,W,W)
        # Now r_h is the continuous relative position values along width.

        # ---------- vertical axis (columns): length H per column ----------
        # Compute pairwise along height: we need (B, W, H, H) essentially
        # q_t = q.permute(0, 3, 2, 1)  # (B, W, pos_dim, H) intermediate; easier: transpose H/W dims
        # simpler: permute to (B, W, H, d)
        q_v = q.permute(0, 2, 1, 3)  # (B, W, H, d) where indexing is column-major
        k_v = k.permute(0, 2, 1, 3)  # (B, W, H, d)
        # want (B, W, H, H) = dot over last dim between positions along H.
        gates_logits_v = torch.einsum("bwhd,bwjd->bwhj", q_v, k_v) / (scale + EPS)  # (B, W, H, H)
        gates_v = torch.sigmoid(self.scale * gates_logits_v)  # (B, W, H, H)
        prefix_v = gates_v.cumsum(dim=-1)  # (B, W, H, H)
        # r_v[b,w,i,j] = prefix_v[b,w,j] - prefix_v[b,w,i]
        r_v = prefix_v.unsqueeze(-2) - prefix_v.unsqueeze(-1)  # (B, W, H, H) -> after broadcasts -> (B, W, H, H)

        # We need r_v in shape (B, H, W, W) or an equivalent mapping to per-patch (i,j) pairs.
        # Let's rearrange: currently r_v indexed by (B, W, i, j) where i,j along height positions.
        # For patch pair with coordinates (h1,w1) and (h2,w2), the vertical relative value is:
        # r_v[b, w1, h1, h2]  (we can gather by appropriate indexing)
        # and horizontal relative value is r_h[b,h1,w1,w2]

        # Build per-pair axis relative positions for all patch pairs:
        # For convenience create grids of indices and then gather (vectorized).
        # Generate r_h_per_pair: shape (B, H, W, H, W) where dims are (b, h1, w1, h2, w2)
        # but storing full 5D is large; we will reshape to (B, N, N) at the end.

        # Compute horizontal relative positions for each ordered pair (i->j):
        # r_h_by_pair[b, h1, w1, h2, w2] = r_h[b, h1, w1, w2]  (doesn't depend on h2)
        # So we can expand r_h to (B, H, W, 1, W) and broadcast over h2 dimension:
        r_h_exp = r_h.unsqueeze(3)  # (B, H, W, 1, W)

        # Compute vertical relative positions expanded similarly:
        # r_v currently (B, W, H, H) so r_v[b, w1, h1, h2]
        # we want r_v_by_pair[b, h1, w1, h2, w2] = r_v[b, w1, h1, h2]
        r_v_exp = r_v.permute(0, 2, 1, 3).unsqueeze(4)  # (B, H, W, H, 1) after permute and unsqueeze

        # Now assemble per-pair continuous relative scalar positions along each axis:
        # horizontal scalar = r_h_exp[..., w2]
        # vertical scalar   = r_v_exp[..., h2]
        # final shape (B, H, W, H, W)
        # We will flatten to (B, N, N)
        # Concatenate axis scalars into two scalars per pair later used to get embeddings.
        # For memory-efficiency we'll directly compute embeddings per pair by interpolation.

        # Flatten coords
        # For positions i flatten index p = h1 * W + w1; for j flatten q = h2 * W + w2
        # r_h_exp: (B, H, W, 1, W) -> after broadcast becomes (B, H, W, H, W)
        # r_v_exp: (B, H, W, H, 1) -> broadcast to (B, H, W, H, W)
        r_h_full = r_h_exp.expand(-1, -1, -1, H, -1)  # (B, H, W, H, W)
        r_v_full = r_v_exp.expand(-1, -1, -1, -1, W)  # (B, H, W, H, W)

        # Flatten pair dims to (B, N, N)
        r_h_flat = r_h_full.reshape(B, H * W, H * W)  # (B, N, N)
        r_v_flat = r_v_full.reshape(B, H * W, H * W)  # (B, N, N)

        # Clamp continuous positions into [0, max_pos - 1 + small frac]
        def interp_embeddings(pos_scalar: torch.Tensor):
            """
            pos_scalar: (B, N, N) continuous >= maybe negative. We'll shift to non-negative by offsetting
            by maximum negative value per-batch if needed. In the paper sums are non-negative for j>=i;
            but here we allow negative if j<i; to make embeddings work, we shift by +max_pos/2.
            """
            B_, NN1, NN2 = pos_scalar.shape
            # shift so values fall into 0..max_pos-1 range (simple heuristic)
            # compute min and max, then affine map to [0, max_pos-1]
            p_min = pos_scalar.amin(dim=(1, 2), keepdim=True)  # (B,1,1)
            p_max = pos_scalar.amax(dim=(1, 2), keepdim=True)
            # avoid degenerate mapping
            span = (p_max - p_min).clamp(min=1e-3)
            # affine map positions to [0, max_pos - 1 - eps]
            scaled = (pos_scalar - p_min) / span * (self.max_pos - 1 - 1e-3)
            scaled = scaled.clamp(0.0, self.max_pos - 1 - 1e-6)
            # floor and frac
            flo = torch.floor(scaled).long()  # (B,N,N)
            frac = (scaled - flo.float()).unsqueeze(-1)  # (B,N,N,1)
            # gather embeddings for flo and flo+1
            # pos_table: (max_pos+1, pos_dim)
            emb0 = F.embedding(flo.clamp(0, self.max_pos), self.pos_table)  # (B,N,N,pos_dim)
            emb1 = F.embedding((flo + 1).clamp(0, self.max_pos), self.pos_table)  # (B,N,N,pos_dim)
            emb = emb0 * (1.0 - frac) + emb1 * (frac)  # linear interpolation
            return emb  # (B,N,N,pos_dim)

        # get interpolated embeddings for both axes
        emb_h = interp_embeddings(r_h_flat)  # (B, N, N, pos_dim)
        emb_v = interp_embeddings(r_v_flat)  # (B, N, N, pos_dim)

        # Now aggregate per-pair embeddings into per-reference patch positional vectors
        # Paper suggests weighting contributions by gate values between reference i and partner j.
        # We already have gates matrices for horizontal and vertical; build combined gate per pair:
        # We need gates expanded to (B,N,N). Build gates_h_flat and gates_v_flat similarly as we did for r_*.
        gates_h_full = gates_h.unsqueeze(3).expand(-1, -1, -1, H, -1).reshape(B, H * W, H * W)  # (B,N,N)
        gates_v_full = gates_v.permute(0, 2, 1, 3).unsqueeze(4).expand(-1, -1, -1, -1, W).reshape(B, H * W,
                                                                                                  H * W)  # (B,N,N)
        # combine axis gates (product or average). Paper uses gates along axis separately; we combine by mean here:
        combined_gates = (gates_h_full + gates_v_full) * 0.5  # (B,N,N)

        if self.normalize_gates:
            # normalize weights across j for each reference i
            norm = combined_gates.sum(dim=-1, keepdim=True) + EPS  # (B, N, 1)
            weights = combined_gates / norm
        else:
            weights = combined_gates

        # weights: (B, N, N); emb_h/emb_v: (B, N, N, pos_dim)
        # compute per-reference axis vectors: sum_j weights[i,j] * emb[..., j]
        weights_unsq = weights.unsqueeze(-1)  # (B,N,N,1)
        pv_h = (weights_unsq * emb_h).sum(dim=2)  # (B, N, pos_dim)
        pv_v = (weights_unsq * emb_v).sum(dim=2)  # (B, N, pos_dim)

        # final per-patch position vector: concat axis vectors (paper aggregates neighbors from both axes).
        pos_vec = torch.cat([pv_h, pv_v], dim=-1)  # (B, N, pos_dim*2)

        # Now form pairwise distances between pos_vecs to produce scalar bias per pair:
        # d_ij = ||pos_i - pos_j||_2
        # bias = -d_ij  (negative distance as bias; paper uses Euclidean distance based bias)
        # compute efficiently:
        # pos_vec: (B, N, D)
        D = pos_vec.shape[-1]
        # compute squared norms
        norms = (pos_vec ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
        # pairwise squared distances: (B, N, N) = norms_i + norms_j^T - 2 * pos @ pos^T
        pairwise = norms + norms.transpose(1, 2) - 2.0 * (pos_vec @ pos_vec.transpose(-1, -2))  # (B,N,N)
        # stabilize small negative
        pairwise = pairwise.clamp(min=0.0)
        d = torch.sqrt(pairwise + EPS)  # Euclidean distance
        bias = -d  # negative distance as bias; you can use -d**2 if you prefer squared distance.

        # Return (B, N, N). If you use multi-head attention, expand this to (B, num_heads, N, N) outside.
        return bias


class SaPE2PosBias(nn.Module):
    """
    Implements the positional bias construction exactly as in the SaPE² paper:
    - Horizontal gating → r_h
    - Vertical gating   → r_v
    - Combine into full (B, N, N) bias matrix
    """

    def __init__(self,
                 dim: int,
                 pos_dim: int = 64,
                 max_pos: int = 64,
                 normalize_gates: bool = True,
                 apply_sigmoid_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.max_pos = max_pos
        self.normalize_gates = normalize_gates
        self.scale = apply_sigmoid_scale

        # projections used to compute semantic gates along each axis (shared across axes)
        # paper indicates gate from inner product of local projections; we follow that.
        self.q_proj = nn.Linear(dim, pos_dim, bias=False)
        self.k_proj = nn.Linear(dim, pos_dim, bias=False)

        # learnable integer position embeddings for positions 0..max_pos (per axis)
        # we store one table and reuse it for both axes (paper interpolates integer embeddings).
        self.pos_table = nn.Parameter(torch.randn(max_pos + 1, pos_dim))  # +1 to allow floor+1

        # small linear to fuse two axis vectors into same output dim if desired (optional)
        # Here we keep final position vector dim = pos_dim * 2 (concat axis vectors).
        # Optionally you can project down to match model dim.
        # self.fuse = nn.Linear(pos_dim * 2, dim)  # optional

    def forward(self, x: torch.Tensor,
                H: Optional[int] = None,
                W: Optional[int] = None) -> torch.Tensor:
        """
        x: (B, N, C) or (B, H, W, C)
        returns:
          bias: (B, N, N) float tensor. Add this to attention logits.
        """
        if x.ndim == 4:
            # (B, H, W, C) -> flatten to (B, N, C)
            B, H, W, C = x.shape
            N = H * W
            x_flat = x.view(B, N, C)
        elif x.ndim == 3:
            B, N, C = x.shape
            assert H is not None and W is not None and H * W == N, "Provide H and W when passing (B,N,C)"
            x_flat = x
        else:
            raise ValueError("x must be shape (B,H,W,C) or (B,N,C)")

        # reshape to grid for axis operations
        x_grid = x_flat.view(B, H, W, C)  # (B, H, W, C)

        # compute projections for gates
        q = self.q_proj(x_grid)  # (B, H, W, pos_dim)
        k = self.k_proj(x_grid)  # (B, H, W, pos_dim)

        # We'll compute gates along horizontal axis (rows) and vertical axis (cols) separately,
        # then produce per-patch axis-specific position vectors and concatenate.

        # ---------- horizontal axis (rows): length W per row ----------
        # compute pairwise dot products along width for each row:
        # Prepare shapes: (B, H, W, d). We'll compute (B, H, W, W) = dot over last dim.
        d = q.shape[-1]
        scale = (d ** 0.5)

        # ---------------------------------------------------------------
        # 1) Horizontal gating  (gates_h)
        # ---------------------------------------------------------------
        # Compare each position (h, w_q) with all (h, w_k)
        # Result: (B, H, W, W)
        gates_logits_h = torch.einsum("bhid,bhjd->bhij", q, k) / (scale + EPS)  # (B, H, W, W)

        gates_h = torch.sigmoid(self.scale * gates_logits_h)  # (B, H, W, W)

        # Cumulative prefix sum along horizontal key dimension
        prefix_h = gates_h.cumsum(dim=-1)  # (B, H, W, W)

        # Pairwise differences along horizontal axis
        # prefix_h.unsqueeze(-2): (B, H, W, 1, W)
        # prefix_h.unsqueeze(-1): (B, H, W, W, 1)
        r_h = prefix_h.unsqueeze(-2) - prefix_h.unsqueeze(-1)  # (B, H, W, W, W)

        # Now expand over vertical dimension (height)
        # We need (B, H, W, H, W)
        # r_h = r_h.unsqueeze(3).expand(B, H, W, H, W)  # (B, H, W, H, W)

        # ---------------------------------------------------------------
        # 2) Vertical gating  (gates_v)
        # ---------------------------------------------------------------
        # Re-index q,k to interchange H<->W (column-major behavior)
        q_v = q.permute(0, 2, 1, 3)  # (B, W, H, D)
        k_v = k.permute(0, 2, 1, 3)  # (B, W, H, D)

        # Compare (w, h_q) with (w, h_k)
        gates_logits_v = torch.einsum(
            "bwhd,bwjd->bwhj", q_v, k_v
        ) / (self.scale + self.eps)
        gates_v = torch.sigmoid(self.scale * gates_logits_v)  # (B, W, H, H)

        prefix_v = gates_v.cumsum(dim=-1)  # (B, W, H, H)
        r_v = prefix_v.unsqueeze(-2) - prefix_v.unsqueeze(-1)  # (B, W, H, H, H)

        # Move back to (B, H, W, H, W)
        r_v = (
            r_v.permute(0, 2, 1, 3, 4)  # (B, H, W, H, H)
            # .unsqueeze(-1)  # (B, H, W, H, H, 1)
            # .expand(B, H, W, H, W)  # (B, H, W, H, W)
        )

        # ---------------------------------------------------------------
        # 3) Combine horizontal + vertical
        # ---------------------------------------------------------------
        r = r_h + r_v  # (B, H, W, H, W)

        # ---------------------------------------------------------------
        # 4) Flatten to (B, N, N)
        # ---------------------------------------------------------------
        pos = r.reshape(B, N, N)  # (B, N, N)

        # ---------------------------------------------------------------
        # 5) Add a head dimension → (B, 1, N, N)
        # ---------------------------------------------------------------
        pos = pos.unsqueeze(1)

        return pos  # (B, 1, N, N)


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
                                                      thermal_preprocessCfg.scene,
                                                      thermal_preprocessCfg.naive_train_first,
                                                      thermal_preprocessCfg.start_training)

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
        ir = self.thermal_preprocess(ir, vis_night, **kwargs)
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
        x = ir.mean(dim=1, keepdim=True)
        out = self.final_conv(x_feat) #+ self.extract_hf(x)/2
        return self.tanh_n(1)(out).repeat(1, vis_night.shape[1], 1, 1), ir, vis_night  # match input channels
#
    def extract_hf(self, x):
        k_s = 7
        x1 = conv2d(x, weight=torch.ones(1, 1, k_s, k_s, device=x.device)/k_s**2, padding=k_s//2)
        return x - x1  # back to [-1,1]

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

    def __init__(self, bins: int = 2048, scene: int = 8,
                 naive_train_first: bool = True, start_training: int = 0, eps=1e-8):
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
        self.naive_train = naive_train_first
        self.start_training = start_training

    def forward(self, x, *args, epoch=0):
        """
        x: IR Tensor of shape (B,1,H,W) or (B,3,H,W)
           assumed normalized to [0,1]
        args: complementary modality for scene selection
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # convert to grayscale
        # Robust normalization to [0,1]
        x = self.robust_norm(x, p_low=2., p_high=99.5, eps=self.eps)
        if epoch > self.start_training:
            self.scene_idx = self.scene_selection(x, *args)  # (B, scene) long tensor
        elif self.naive_train:
            self.scene_idx = self.naive_scene_selection(x)
        else:
            idx = torch.zeros([x.shape[0], self.scene], device=x.device)
            idx[0] = 1.
            self.scene_idx = idx
            # Build monotonic LUT
        increments = F.softplus(torch.mm(self.scene_idx, self.delta)) + self.eps
        luts = torch.cumsum(increments, dim=1)
        luts = luts / (luts[:, -1] + self.eps) * 2 - 1  # normalize to [-1,1]

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
        low_lum_t = (x[:, :, 2::] < -0.95).sum(dim=[1, 2, 3]) / (x[:, :, 2::]>=-1).sum(dim=[1, 2, 3])
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