import os
from typing import Literal

import torch
from torch import nn

from .models.basic_blocks import back_warp
from .models.cross_raft import CrossRAFT
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_wrapper(direction: Literal['ir2vis', 'vis2ir'], **kwargs):
    model = CrossRAFT(adapter=True)
    state_dict = torch.load(os.getcwd() + '/NightToday/CrossRAFT/checkpoints/checkpoint-10000.ckpt',
                            weights_only=True)['state_dict']
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
            flow = self.model(img_tgt*0.5+0.5, img_src*0.5+0.5)['flow']
            image_proj = self.ST(img_src, flow)
            return image_proj

    return Model()
