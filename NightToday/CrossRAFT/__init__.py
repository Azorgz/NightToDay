import os
import socket
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn.functional import interpolate

from .models.basic_blocks import back_warp
from .models.cross_raft import CrossRAFT
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_wrapper(direction: Literal['ir2vis', 'vis2ir'], **kwargs):
    model = CrossRAFT(adapter=True)
    BASE_DIR = Path(__file__).resolve().parent

    if 'laptop' in socket.gethostname():
        path = BASE_DIR / 'checkpoints' / 'checkpoint-10000.ckpt'
    else:
        path = Path('/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/CrossRAFT/checkpoint-10000.ckpt')

    path = str(path)
    state_dict = torch.load(path, weights_only=True)['state_dict']
    model.load_state_dict(state_dict)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.direction = direction
            self.model = model.eval()
            self.ST = back_warp

        def forward(self, img_vis, img_ir):
            if self.direction == 'ir2vis':
                img_tgt, img_src = img_vis, img_ir
            else:
                img_tgt, img_src = img_ir, img_vis
            shape = img_tgt.shape[-2:]
            # img_tgt_ = interpolate(img_tgt, (256, 256)).to(torch.float32)
            # img_src_ = interpolate(img_src, (256, 256)).to(torch.float32)
            flow = self.model(img_tgt, img_src)['flow']
            # flow = interpolate(flow, size=shape, mode='bilinear', align_corners=True)
            image_proj = self.ST(img_src, flow)
            return image_proj

    return Model()
