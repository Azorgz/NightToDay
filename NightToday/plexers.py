#### PLEXERS
import os
import socket
from os.path import isfile
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn.functional import interpolate

from . import GenConfig, TrainConfig, SegConfig, DiscrConfig
from .Fusion import U_ResNetFusion, IlluminationAwareFusion
from .LETNet import LETNet
from .discriminators import NLayerDiscriminatorSN
from .generators import ResnetGenEncoder, ResnetGenDecoder, ResnetBlock
from .modules import Sequential
from .segmentors import SegmentorHeadv2
from .utilities import weights_init


class Plexer(nn.Module):
    """
    Base Plexer class for multiple networks.
    attributes:
        names_domains: dict mapping domain names to indices
    """

    def __init__(self, names, device: torch.device, split_optimizers: bool = False):
        super(Plexer, self).__init__()
        self.names_domains = {name: i for i, name in enumerate(names)}
        self.names = []
        self.networks = []
        self.device = device
        self.split_optimizers = split_optimizers

    def __len__(self):
        return len(self.networks)

    def train(self, *args, mode: bool = True):
        super().train(mode=mode)
        for net in self.networks:
            # if isinstance(net, SimpleCondViT) or isinstance(net, U_ResNetFusion):
            #     net.train(mode=mode)
            #     for p in net.parameters():
            #         p.requires_grad = True
            # else:
            net.train(mode=False)
                # for p in net.parameters():
                #     p.requires_grad = False
        for arg in args if args else range(len(self.networks)):
            if arg < len(self.networks):
                self.networks[arg].train(mode=mode)
                if hasattr(self.networks[arg], 'patch_encoder'):
                    self.networks[arg].patch_encoder.train(False)
            else:
                pass

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', self.device)
        for net in self.networks:
            net.to(device)

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def init_optimizers(self, opt, lr, betas):
        if self.split_optimizers:

            optimizers = {name: opt(net.parameters(), lr=lr, betas=betas)
                          for name, net in zip(self.names, self.networks)}
        else:
            optimizers = [opt((p for net in self.networks for p in net.parameters() if p.requires_grad),
                              lr=lr, betas=betas)]
        setattr(self, 'optimizers', optimizers)

    def zero_grads(self, *args):
        if args is None or len(args) == 0 or not self.split_optimizer:
            for opt in self.optimizers:
                opt.zero_grad()
        else:
            for dom in args:
                if dom < len(self.optimizers):
                    self.optimizers[self.names[dom]].zero_grad()

    def step_grads(self, *args):
        if args is None or len(args) == 0 or not self.split_optimizer:
            for opt in self.optimizers:
                opt.step()
        else:
            for d in args:
                if d < len(self.optimizers):
                    self.optimizers[self.names[d]].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers.values():
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + f'{self.names[i]}.pth'
            torch.save(net.state_dict(), filename)

    def get_weights(self):
        weights = {}
        for i, net in enumerate(self.networks):
            weights[self.names[i]] = net.state_dict()
        return weights

    def load_split_weights(self, save_path: str | list):
        if not isinstance(save_path, list):
            save_path = [save_path] * len(self.networks)
        for i, (net, path) in enumerate(zip(self.networks, save_path)):
            filename = path + '.pth'
            if isfile(filename):
                net.load_state_dict(torch.load(filename))
        self.to(device=self.device)

    def load_weights(self, weights: dict):
        for i, net in enumerate(self.networks):
            if i < len(self.names):
                if self.names[i] in weights:
                    net.load_state_dict(weights[self.names[i]], strict=False)
        self.to(device=self.device)


class G_Plexer(Plexer):
    """Generator Plexer for multiple domains using shared Transformer blocks.
    """

    def __init__(self, names, opt: GenConfig, training_cfg: TrainConfig, device: torch.device):
        super(G_Plexer, self).__init__(names, device, training_cfg.split_optimizers)
        self.input_size = opt.input_size
        self.fusion_first = opt.fusion_first
        self.opt = opt
        encoders = [ResnetGenEncoder] * 2
        decoders = [ResnetGenDecoder] * 2  # for _ in range(len(self.names_domains))]
        enc_args = [(3, opt.hidden_dim, opt.n_enc_layers, opt.dropout, opt.downscaling),
                    (3 if opt.fusion_first else 6, opt.hidden_dim, opt.n_enc_layers, opt.dropout, opt.downscaling)]
        dec_args = [(3, opt.hidden_dim, opt.n_dec_layers, opt.dropout, opt.downscaling),
                    (3 if opt.fusion_first else 6, opt.hidden_dim, opt.n_dec_layers, opt.dropout, opt.downscaling)]
        block_shared = ResnetBlock
        shenc_args = (opt.n_shared_layers, opt.hidden_dim, nn.BatchNorm2d)
        fus = opt.fus
        if not hasattr(fus, 'type'):
            fus.type = 'IAware'
        if fus.type == 'UResNet':
            fusion_module = U_ResNetFusion
        else:
            fusion_module = IlluminationAwareFusion
        self.fusion = fusion_module(hidden_dim=fus.hidden_dim, n_enc_layers=fus.n_enc_layers, dropout=fus.dropout,
                                     n_downscaling=fus.n_downscaling, thermal_preprocessCfg=fus.preprocess_thermal)
        self.encoders = [encoder(*enc_arg).train(False) for encoder, enc_arg in zip(encoders, enc_args)]
        self.decoders = [decoder(*dec_arg).train(False) for decoder, dec_arg in zip(decoders, dec_args)]
        self.networks: list = self.encoders + self.decoders + [self.fusion]
        self.names = ([f'GenEnc_{dom}' for dom, i in zip(self.names_domains, range(2))] +
                      [f'GenDec_{dom}' for dom, i in zip(self.names_domains, range(2))] + ['Fusion'])

        if opt.n_shared_layers > 0:
            self.shared_encoder = Sequential(*[block_shared(*shenc_args[1:])] * shenc_args[0])
            self.networks.append(self.shared_encoder)
            self.names.append('GenShared')
        else:
            self.shared_encoder = nn.Identity()
        self.to(self.device)
        self.ori_shape = None
        self.train()
        self.init_optimizers(torch.optim.Adam, lr=training_cfg.lr_G, betas=training_cfg.betas_G)

    def encode(self, x, *args, from_: str = None, **kwargs):
        assert from_ in self.names_domains, f"Unknown source domain: {from_}"
        return_im = False
        if len(args) and self.fusion_first:
            x, ir, n, *other = self.fusion(x, *args, **kwargs)
            return_im = True
        elif len(args) and not self.fusion_first:
            x = torch.cat((x, *args), dim=1)
            n = args[0]
            return_im = True
        self.ori_shape = x.shape
        scale = 2**self.opt.downscaling
        # input_size = self.input_size if isinstance(self.input_size, (list, tuple)) else (self.input_size, self.input_size)
        input_size = (-1, -1)
        if input_size[0] < 0:
            input_size = self.ori_shape[-2]//scale*scale, self.ori_shape[-1]//scale*scale
        else:
            if input_size[0] / scale != input_size[0] // scale:
                input_size[0] = input_size[0] // scale * scale
            if input_size[1] / scale != input_size[1] // scale:
                input_size[1] = input_size[1] // scale * scale
        x = interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        output = self.encoders[self.names_domains[from_]](x)
        output = self.shared_encoder(output)
        if return_im:
            return output, x, ir, n, *other
        return output

    def clean_IR(self, ir):
        return self.fusion.thermal_preprocess(ir)

    def decode(self, encoded, to_: str = None):
        assert to_ in self.names_domains, f"Unknown target domain: {to_}"
        out = self.decoders[self.names_domains[to_]](encoded)
        return interpolate(out, size=self.ori_shape[-2:], mode='bilinear', align_corners=False)

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) + '\n' + repr(d) + '\n' + \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) + '\n' + \
            'Number of parameters per Encoder: %d' % e_params + '\n' + \
            'Number of parameters per Decoder: %d' % d_params


class D_Plexer(Plexer):
    def __init__(self, names, opt: DiscrConfig, training_cfg: TrainConfig, device: torch.device):
        super(D_Plexer, self).__init__(names, device, training_cfg.split_optimizers)
        discriminators = NLayerDiscriminatorSN
        if opt.fusion_first:
            discr_args = [{'input_nc': 3, 'base_dim': opt.base_dim, 'n_layers': opt.n_layers},
                          {'input_nc': 3, 'base_dim': opt.base_dim, 'n_layers': opt.n_layers}]
        else:
            discr_args = [{'input_nc': 3, 'base_dim': opt.base_dim, 'n_layers': opt.n_layers}] * 3
        self.networks = [discriminators(**model_arg) for model_arg in discr_args]
        self.names = [f'D_{dom}' for dom in self.names_domains]
        self.apply(weights_init)
        self.to(self.device)
        self.init_optimizers(torch.optim.Adam, lr=training_cfg.lr_D, betas=training_cfg.betas_D)

    def forward(self, x, from_: str = None):
        assert from_ in self.names_domains, f"Unknown source domain: {from_}"
        return self.networks[self.names_domains[from_]](x)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) + '\n' + \
            'Created %d Discriminators' % len(self.networks) + '\n' + \
            'Number of parameters per Discriminator: %d' % t_params


class S_Plexer(Plexer):
    _stage: Literal['update_D', 'update_TN', 'train_D', 'freeze_all', 'freeze_seg', 'continue']

    def __init__(self, names, opt: SegConfig, training_cfg: TrainConfig, device: torch.device):
        super(S_Plexer, self).__init__(names, device, training_cfg.split_optimizers)
        self.Scheduler = opt.training_schedule
        self.num_classes = opt.num_classes
        self.epoch = 0
        if not opt.type == 'LETNet':
            model = SegmentorHeadv2
            model_args = [(3, opt.n_layers, opt.base_dim, opt.num_classes, 'instance'),
                          (3 if opt.fusion_first else 6, opt.n_layers, opt.base_dim, opt.num_classes, 'instance')]
        else:
            model = LETNet
            model_args = [(opt.num_classes, 3), (opt.num_classes, 3 if opt.fusion_first else 6)]

        self.networks = [model(*model_arg) for model_arg in model_args]
        self.names = [f'S_{dom}' for dom in self.names_domains]
        BASE_DIR = Path(__file__).resolve().parent.parent
        for net, name in zip(self.networks, self.names):
            if 'laptop' in socket.gethostname():
                path = BASE_DIR / 'checkpoints' / f'{name}.pth'
            else:
                path = f'/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/{name}.pth'
            net.load_state_dict(torch.load(path), strict=False)
        self.to(self.device)
        self.init_optimizers(torch.optim.Adam, lr=training_cfg.lr_S, betas=training_cfg.betas_D)
        self.freeze = True

    def forward(self, x, *args, from_: str = None):
        assert from_ in self.names_domains, f"Unknown source domain: {from_}"
        if len(args):
            x = torch.cat((x, *args), dim=1)
        return self.networks[self.names_domains[from_]](x)

    def stage_update(self):
        '''
        Determine the current training stage based on the epoch and scheduler settings.
        :return:
            'update_D' : Update Day seg-map regarding fake T, train D normally and TN with pseudo TN image and D labels
            'update_TN': Update TN mask regarding fake T, train D normally and TN with pseudo TN image and D labels and real TN images
            'train_D'  : Train Discriminator normally
            'freeze_all': Freeze all networks (before training starts)
            'freeze_seg': Freeze segmentation network (after training ends)
        '''
        if self.Scheduler.start_epoch <= self.epoch < self.Scheduler.end_epoch:
            if self.Scheduler.updateGT_D_start_epoch <= self._epoch < self.Scheduler.updateGT_TN_start_epoch:
                self.stage = 'update_D'
            elif self.Scheduler.updateGT_TN_start_epoch <= self._epoch:
                self.stage = 'update_TN'
            else:
                self.stage = 'train'
        elif self.epoch < self.Scheduler.start_epoch:
            self.stage = 'freeze_all'
        else:
            self.stage = 'trained'

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        for param_group_a in self.optimizers[dom_a].param_groups:
            param_group_a['lr'] = new_lr
            print('Learning rate of SegA is: %.4f.' % param_group_a['lr'])

        for param_group_b in self.optimizers[dom_b].param_groups:
            # print(param_group_b['lr'])
            param_group_b['lr'] = new_lr
            print('Learning rate of SegB is: %.4f.' % param_group_b['lr'])

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, value):
        self._stage = value

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        if self.Scheduler.start_epoch <= self._epoch <= self.Scheduler.end_epoch:
            self.freeze = False
        else:
            self.freeze = True
        self.stage_update()

    @property
    def freeze(self):
        return self._freeze

    @freeze.setter
    def freeze(self, value):
        self._freeze = value
        if self._freeze:
            self.train()
            for net in self.networks:
                net.eval()
        else:
            for net in self.networks:
                net.train()

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) + '\n' + \
            'Created %d Segmentors' % len(self.networks) + '\n' + \
            'Number of parameters per Segmentor: %d' % t_params
