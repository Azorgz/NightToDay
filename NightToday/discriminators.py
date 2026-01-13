import functools
import numpy as np
import torch.nn as nn
import torch
from torch import tensor

from .modules import SNConv2d
from .utilities import gkern_2d


class TemporalDiscriminator(nn.Module):
    def __init__(self, input_nc, base_dim=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(TemporalDiscriminator, self).__init__()
        self.model = NLayerDiscriminatorSN(input_nc * 2, base_dim, n_layers, norm_layer)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        outs = self.model(x)
        return outs


class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, base_dim=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminatorSN, self).__init__()
        self.input_nc = input_nc
        self.grad_filter = tensor([0., 0., 0., -1., 0., 1., 0., 0., 0.], dtype=torch.float32).view(1, 1, 3, 3)
        self.dsamp_filter = tensor([1], dtype=torch.float32).view(1, 1, 1, 1)
        self.blur_filter = tensor(gkern_2d(nchannels=input_nc), dtype=torch.float32)

        self.model_rgb = self.model(input_nc, base_dim, n_layers, norm_layer)
        self.model_gray = self.model(1, base_dim, n_layers, norm_layer)
        self.model_grad = self.model(2, base_dim, n_layers - 1, norm_layer)

    def to(self, *args, **kwargs):
        self.grad_filter = self.grad_filter.to(*args, **kwargs)
        self.dsamp_filter = self.dsamp_filter.to(*args, **kwargs)
        self.blur_filter = self.blur_filter.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def model(self, input_nc, base_dim, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequences = [[
            SNConv2d(input_nc, base_dim, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequences += [[
                SNConv2d(base_dim * nf_mult_prev, base_dim * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequences += [[
            SNConv2d(base_dim * nf_mult_prev, base_dim * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.PReLU(),
            SNConv2d(base_dim * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, x):
        blurred = torch.nn.functional.conv2d(x, self.blur_filter, groups=self.input_nc, padding=2)
        if self.input_nc == 1:
            gray = blurred
        elif self.input_nc == 3:
            gray = x.mean(dim=1, keepdim=True)
            gray = (.299 * x[:, 0, :, :] + .587 * x[:, 1, :, :] + .114 * x[:, 2, :, :]).unsqueeze_(1)
        else:
            gray = (.299 * x[:, 0, :, :] + .587 * x[:, 1, :, :] + .114 * x[:, 2, :, :]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2, -1))
        gradient = torch.cat([dx, dy], 1)

        outs1 = self.model_rgb(blurred)
        outs2 = self.model_gray(gray)
        outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3


class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, x):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(x)
            if i == 0:
                x = output
                continue
            predictions.append(output[:, -1, :, :])
            if i != len(layers) - 1:
                x = output[:, :-1, :, :]
        return predictions
