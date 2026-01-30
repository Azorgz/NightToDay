import os
import socket
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
    path = os.getcwd() + '/NightToday/CrossRAFT/' if 'laptop' in socket.gethostname() else '/NightToday/CrossRAFT/'
    state_dict = torch.load(path + 'checkpoints/checkpoint-10000.ckpt', weights_only=True)['state_dict']
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
            img_tgt_ = interpolate(img_tgt, (256, 256)).to(torch.float32)
            img_src_ = interpolate(img_src, (256, 256)).to(torch.float32)
            flow = self.model(img_tgt_, img_src_)['flow']
            flow = interpolate(flow, size=shape, mode='bilinear', align_corners=True)
            image_proj = self.ST(img_src, flow)
            return image_proj

    return Model()
