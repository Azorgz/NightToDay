"""
Image-to-Image Generative Adversarial Transformer (Dual-Stream)

- Two generators (G_TN->D and inverse optional) with dual encoders:
  - Encoder_T (thermal) for structure
  - Encoder_N (night-visible) for color/appearance
  - Cross-attention fusion
  - Decoder to day-visible image

- Two discriminators (D_D, D_TN) patch-based
- Losses: adversarial (LSGAN), cycle/identity (optional), color consistency (YCbCr chroma),
  gradient (Sobel), optional perceptual (VGG feature loss if enabled)

Usage: open this file in the editor and adapt patch sizes / transformer sizes to your input resolution.
"""
import math
import os
import socket
from collections import OrderedDict
from functools import partial
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from ImagesCameras import ImageTensor
from kornia.augmentation import RandomCrop
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.contrib import connected_components
from kornia.morphology import dilation
from torch import Tensor
from torch.nn.functional import interpolate, normalize

from . import OptImage2ImageGATConfig
from . import get_config
from .losses import GANLoss, SSIM_Loss, TVLoss, StructuralGradientLoss, \
    FakeIRPersonLoss, BiasCorrLoss, ColorLoss, CondGradRepaLoss, AdaptativeColAttentionLoss, SemEdgeLoss, \
    ThermalLoss, SharpFusionLoss, IlluminationAwareFusionLoss, PixelConsistencyLoss, \
    TrafLighLumiLoss_TN, ForegroundContourLoss
from .modules import LossScheduler, Get_gradmag_gray
from .plexers import G_Plexer, D_Plexer, S_Plexer
from .utilities import UpdateVisGT, UpdateIRGTv1, UpdateIRGTv2, AttackImages, get_disk_kernel, \
    determine_color_N
from .visualizers import Visualizer


# ------------------------ Main GAT Class ------------------------ #


class Image2ImageGAT_Dual(nn.Module):
    """
    Full training-ready module for thermal+night -> day translation.
    """
    _partial_train_net: dict[str, list[int]]
    model_name = 'ResNet'

    def __init__(self, opt: OptImage2ImageGATConfig | str | Path | dict | None = None,
                 *args, trainable: bool = False, **kwargs):
        # region initialization sequence
        # If building from checkpoint, load config from the checkpoint given by resume_epoch
        super().__init__()
        checkpoint = self.initialization(opt, *args, **kwargs)
        self.names_domains = self.opt.model.names_domains
        self.mode = self.opt.model.mode if trainable else 'test'
        self.opt.model.gen.fusion_first = self.opt.model.fusion_first
        self.opt.model.discr.fusion_first = self.opt.model.fusion_first
        self.opt.model.seg.fusion_first = self.opt.model.fusion_first

        if self.mode == 'test':
            self.opt.model.gen.input_size = -1
            self.input_size = -1
        else:
            self.input_size = self.opt.model.gen.input_size
        # endregion

        # region Networks
        self.netG = G_Plexer(self.names_domains, self.opt.model.gen, self.opt.training, self.device)
        self.netD = D_Plexer(self.names_domains, self.opt.model.discr, self.opt.training, self.device)
        self.netS = S_Plexer(self.names_domains, self.opt.model.seg, self.opt.training, self.device)
        self.load(self.opt.training.resume_epoch, checkpoint=checkpoint)
        # endregion

        # region Inputs / Outputs buffers
        self.set_input()
        # endregion

        # region Train parameters, criterion and losses
        if self.mode == 'train' and trainable:
            p_color = self.opt.training.pedestrian_color
            if isinstance(p_color, list):
                if len(p_color) == 1:
                    p_color = [determine_color_N(mcolors.CSS4_COLORS[p_color[0]])] * 2
                self.pedestrian_color = (Tensor(mcolors.to_rgb(mcolors.CSS4_COLORS[p_color[0]])),
                                         Tensor(mcolors.to_rgb(mcolors.CSS4_COLORS[p_color[1]])))
            else:
                self.pedestrian_color = (None, None)
            self.checkpoint_dir = self.opt.training.checkpoint_dir if 'laptop' in socket.gethostname() else '/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/'
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.visualize_dir = self.opt.training.visualize_dir
            os.makedirs(self.visualize_dir, exist_ok=True)
            self.visualizer = Visualizer(self.opt.training)
            # Training functions
            self.att_input = AttackImages(device=self.device)
            # criteria
            self.get_gradmag = Get_gradmag_gray()
            self.used_losses = []
            self.sum_lambdas = -1.0
            self.GANLoss = GANLoss(gan_type=self.opt.training.gan_type, device=self.device)
            self.L1 = nn.SmoothL1Loss()
            self.L1_sum = nn.SmoothL1Loss(reduction='sum')
            self.downsample = torch.nn.AvgPool2d(3, stride=2)

            self.criterion_gan = lambda d, r, p_r, f, v: self.GANLoss(d, r, p_r, f, v)
            self.criterion_id = lambda y, t: self.L1(self.downsample(y), self.downsample(t))
            self.criterion_cycle = lambda rec, real: nn.SmoothL1Loss(beta=0.5)(rec, real) + self.criterion_ssim(rec,
                                                                                                                real) / self.lambda_cycle
            self.criterion_latent = lambda y, t: self.L1(y, t.detach())
            self.criterion_ssim = lambda x, y: SSIM_Loss()((x + 1) / 2, (y + 1) / 2) * self.lambda_ssim
            self.criterion_tv = TVLoss(TVLoss_weight=1)
            self.criterion_color = ColorLoss
            self.criterion_thermal = ThermalLoss
            self.criterion_att = lambda rec, fake: (self.criterion_cycle(rec, self.real_TN) +
                                                    self.criterion_cycle(fake, self.fake_D.detach()))
            self.criterion_semEdge = partial(SemEdgeLoss, num_classes=self.netS.num_classes)
            self.criterion_sharpness = SharpFusionLoss()
            self.criterion_scene_id = nn.CrossEntropyLoss()
            self.criterion_cgr = lambda f_d, seg_t, r_t: CondGradRepaLoss(f_d,
                                                                          seg_t.detach() if seg_t is not None else seg_t,
                                                                          self.get_gradmag(f_d),
                                                                          self.get_gradmag(r_t.detach()))
            self.criterion_aca = lambda r_v, f_v, f_v_m, f_v_f: AdaptativeColAttentionLoss(r_v, f_v.detach(),
                                                                                           f_v_m.detach() if f_v_m is not None else f_v_m,
                                                                                           f_v_f,
                                                                                           4, 100000)
            self.criterion_contour = ForegroundContourLoss
            self.criterion_sga = StructuralGradientLoss(8, 0.8)
            self.criterion_IRClsDis = FakeIRPersonLoss
            self.criterion_bc = BiasCorrLoss
            self.criterion_tll = TrafLighLumiLoss_TN
            self.criterion_tlc = PixelConsistencyLoss
            self.criterion_illum = IlluminationAwareFusionLoss()

            # Losses storage
            self.initialize_losses()
            self.epoch = self.opt.training.start_epoch
            self.losses_scheduler = LossScheduler(self.opt.training.loss_scheduler, epoch=self.epoch)
            # Loss Weights
            for lam in self.losses_scheduler.current.keys():
                setattr(self, f'{lam}', self.losses_scheduler.get(f'{lam}'))
            self.often_weight = torch.ones(self.netS.num_classes, device=self.device)
            self.class_weight = torch.ones(self.netS.num_classes, device=self.device)
            self.max_value = 7
            self.often_balance = True

            # Partial training setup
            self.set_partial_train()

        # endregion

    # region ------------------------ Setup Functions ------------------------ #

    def initialization(self, opt, *args, **kwargs) -> dict | None:
        checkpoint = None
        if isinstance(opt, (str, Path)):
            # Load config from yaml/json if given, else from checkpoint
            if 'yaml' in opt or 'yml' in opt or 'json' in opt:
                self.opt = get_config(opt)
            else:
                checkpoint = torch.load(opt, weights_only=False, map_location='cpu')
                self.opt = get_config()
                self.opt.model = checkpoint['config'].model
        else:
            checkpoint = opt
            self.opt = get_config()
            if isinstance(checkpoint, dict):
                # Checkpoint case
                self.opt.model = checkpoint['config'].model
            else:  # None case
                self.mode = 'train'
                if self.opt.model.build_from_checkpoint and self.opt.training.resume:
                    checkpoint = self.load(self.opt.training.resume_epoch, return_checkpoint=True)
                    self.opt.model = checkpoint['config'].model
                else:
                    checkpoint = None
        self.device = self.opt.device
        return checkpoint

    def save(self, epoch):
        checkpoint = {'epoch': epoch,
                      'config': self.opt}
        for net_label in ['G', 'D', 'S']:
            net = getattr(self, f'net{net_label}')
            if self.opt.training.split_weights:
                self.save_network(net, epoch)
            else:
                checkpoint[net_label] = self.get_weights(net)
        if not self.opt.training.split_weights:
            save_filename = f'{epoch}_net_{self.model_name}'
            save_path = os.path.join(self.checkpoint_dir, save_filename)
            torch.save(checkpoint, save_path)

    def save_network(self, network, epoch):
        save_filename = f'{epoch}_net_'
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        return network.save(save_path)

    @staticmethod
    def get_weights(network) -> OrderedDict:
        return network.get_weights()

    def load(self, epoch: str | int | dict, return_checkpoint: bool = False,
             checkpoint: dict = None) -> OrderedDict | None:
        if self.opt.training.resume or self.opt.model.mode == 'test':
            if not self.opt.training.split_weights:
                if checkpoint is None:
                    assert isinstance(epoch, (str, int)), "When loading full checkpoints, epoch must be str or int."
                    save_filename = f'{epoch}_net_{self.model_name}'
                    path = (os.getcwd() + '/checkpoints/NightToday/') if 'laptop' in socket.gethostname() else \
                        '/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/'
                    save_path = os.path.join(path, save_filename)
                    checkpoint = torch.load(save_path, weights_only=False, map_location='cpu')
                    if return_checkpoint:
                        return checkpoint
                for net_label in ['G', 'D', 'S'] if self.mode == 'train' else ['G']:
                    net = getattr(self, f'net{net_label}')
                    net.load_weights(checkpoint[net_label])
            else:
                for net_label in ['G', 'D', 'S'] if self.mode == 'train' else ['G']:
                    net = getattr(self, f'net{net_label}')
                    if isinstance(epoch, dict):
                        epoch_net = {k: e for k, e in epoch.items() if net_label in k}
                    else:
                        epoch_net = epoch
                    self._load_network(net, epoch_net)

    def _load_network(self, network, epoch):
        if isinstance(epoch, (str, int)):
            save_filename = f'{epoch}_net_'
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        elif isinstance(epoch, dict):
            save_filename = [f'{e}_net_{network_label}' for network_label, e in epoch.items()]
            save_path = [os.path.join(self.checkpoint_dir, fn) for fn in save_filename]
        else:
            raise ValueError("epoch must be str, int, or dict.")
        network.load_split_weights(save_path)

    def set_partial_train(self):
        if self.opt.training.split_optimizers is False:
            self.partial_train_net = {'G': [], 'D': [], 'S': []}
        elif self.opt.training.partial_train is not None:
            self.partial_train_net = self.opt.training.partial_train
        else:
            self.partial_train_net = {'G': [i for i in range(len(self.netG.optimizers))],
                                      'D': [i for i in range(len(self.netD.optimizers))],
                                      'S': [i for i in range(len(self.netS.optimizers))]}

    def set_input(self, *args, **kwargs):
        setattr(self, 'real_D', kwargs.get('D', ImageTensor.rand(1, 3, 4, 4)).to(self.device))
        setattr(self, 'real_D_T', kwargs.get('D_T', ImageTensor.rand(1, 3, 4, 4)).to(self.device)
        if kwargs.get('D_T', ImageTensor.rand(1, 3, 4, 4)) is not None else None)
        setattr(self, 'real_T', kwargs.get('T', ImageTensor.rand(1, 3, 4, 4)).to(self.device))
        setattr(self, 'real_N', kwargs.get('N', ImageTensor.rand(1, 3, 4, 4)).to(self.device))
        setattr(self, 'real_TN', None)
        setattr(self, 'remapped_T', None)
        setattr(self, 'segMask_D', kwargs.get('seg_D', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'segMask_TN', kwargs.get('seg_TN', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'edges_D', kwargs.get('edges_D', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'edges_TN', kwargs.get('edges_TN', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'TL_collection', kwargs.get('TL_collection', {}))
        setattr(self, 'input_size', self.real_D.shape[-2:])
        setattr(self, 'segMask_D_update', None)
        setattr(self, 'segMask_TN_update', None)
        setattr(self, 'pred_real_D', None)
        setattr(self, 'pred_real_T', None)
        setattr(self, 'pred_real_N', None)
        setattr(self, 'fake_D', None)
        setattr(self, 'fake_T', None)
        setattr(self, 'fake_TN', None)
        setattr(self, 'rec_D', None)
        setattr(self, 'rec_T', None)
        setattr(self, 'D_com', None)
        setattr(self, 'T_com', None)
        setattr(self, 'N_com', None)
        setattr(self, 'remapped_T_com', None)
        setattr(self, 'fake_T_com', None)
        setattr(self, 'TN_com', None)
        setattr(self, 'rec_D_com', None)
        setattr(self, 'fake_T_com', None)
        setattr(self, 'fake_D_com', None)
        setattr(self, 'fake_TN_com', None)
        setattr(self, 'rec_TN_com', None)
        setattr(self, 'fake_D_day', None)
        setattr(self, 'att_rec_D', None)

    def initialize_losses(self):
        setattr(self, 'loss_D', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_G', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_S', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_cycle', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_id', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_color', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_color_day', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_grad', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_sga', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_tv', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_ds', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_latent', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_mean_std', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_seg', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_att', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_fus', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_scale_robustness', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_mean_var', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_milo', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_thermal', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_sharpness', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_trafficlight', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_contour', {k: 0. for k in self.names_domains})

    def set_pedestrians_color(self):
        if self.pedestrian_color[0] is None:
            return
        color = self.pedestrian_color[0].view(1, 3, 1, 1).to(self.segMask_D.device) * 2 - 1
        border = self.pedestrian_color[1].view(1, 3, 1, 1).to(self.segMask_D.device) * 2 - 1
        pedestrians = (self.segMask_D == 11) * normalize((self.real_D * 0.5 + 0.5).mean(1, keepdim=True))
        contours = dilation((pedestrians > 0).float(), kernel=torch.ones(3, 3, device=pedestrians.device)) - (
                pedestrians > 0).float()
        colors = color * pedestrians.expand(pedestrians.shape[0], 3, *pedestrians.shape[-2:]) + border * contours
        colors = colors / self.real_D.flatten(1).max(1).values if self.real_D.flatten(1).max(1).values > 0 else colors
        self.real_D = self.real_D * ((self.segMask_D != 11).float() - contours).expand(-1, 3,
                                                                                       *self.real_D.shape[-2:]) + colors

    def grenner_vegetation(self):
        vegetation_mask = (self.segMask_D == 8).float()
        sky_mask = (self.segMask_D == 10).float()
        real_D = (self.real_D * 0.5 + 0.5)
        greener_veg = torch.cat([real_D[:, 0:1], (real_D[:, 1:2] * 1.05).clamp(0, 1), real_D[:, 2:3]], dim=1) * 2 - 1
        bluer_sky = torch.cat([(real_D[:, 0:1] * 0.98).clamp(0, 1), (real_D[:, 1:2] * 0.98).clamp(0, 1),
                               (real_D[:, 2:3] * 1.02).clamp(0, 1)], dim=1) * 2 - 1
        self.real_D = self.real_D * (
                    1 - vegetation_mask - sky_mask) + greener_veg * vegetation_mask + bluer_sky * sky_mask

    # endregion

    # region ------------------------ Inference Function -------------------- #
    @torch.no_grad()
    @torch.no_grad()
    def forward(self, thermal, night=None, return_fused_IR=False, align_first=True):
        inputs = (thermal * 2 - 1, night * 2 - 1) if night is not None else (thermal * 2 - 1,)
        outputs = self.netG.encode(*inputs, from_=self.T, align_first=align_first)
        if len(inputs) == 2:
            encoded_TN, fused_IR, _, _ = outputs
        else:
            encoded_TN = outputs
            fused_IR = None
        fake_D = self.netG.decode(encoded_TN, to_=self.D)
        if return_fused_IR:
            return fake_D * 0.5 + 0.5, fused_IR * 0.5 + 0.5 if fused_IR is not None else None
        return fake_D * 0.5 + 0.5

    @torch.no_grad()
    def split(self, img, patch_size=256):
        h, w = img.shape[-2:]
        mask = 0 * img[:, 0, :, :]
        n_patch_h = h // patch_size + int(h % patch_size != 0)
        overlap_h = (n_patch_h * patch_size - h) // (n_patch_h - 1)
        n_patch_w = w // patch_size + int(w % patch_size != 0)
        overlap_w = (n_patch_w * patch_size - w) // (n_patch_w - 1)

        patch_list = []
        mask_list = []
        j_ = 0
        i_ = 0
        for j in range(n_patch_h):
            j0 = j_ * 1.
            j1 = j0 + patch_size
            j_ = min(j_ + patch_size - overlap_h, h)
            j_ = j_ if j_ + patch_size <= h else h - patch_size

            for i in range(n_patch_w):
                i0 = i_ * 1.
                i1 = i0 + patch_size
                i_ = min(i_ + patch_size - overlap_w, w)
                i_ = i_ if i_ + patch_size <= w else w - patch_size
                patch_list.append(img[..., j0:j1, i0:i1])
                m = mask.clone()
                m[..., j0:j1, i0:i1] = 1.
                mask_list.append(m)
        masks = torch.stack(mask_list, dim=1)
        masks_normed = masks / (masks.sum(dim=1, keepdim=True) + 1e-6)

        return torch.cat(patch_list, dim=0), masks_normed

    @torch.no_grad()
    def merge(self, patches, masks):
        b, n, h, w = masks.shape
        masks = masks[:, :, None].repeat(1, 1, patches.shape[1], 1, 1)
        output = torch.zeros_like(masks, device=masks.device).repeat(1, patches.shape[1], 1, 1)
        for i in range(n):
            valid = masks[:, i] > 0
            output[valid] = masks[:, i][valid] * patches[i // b, :, :, :][valid]

        return output

    # endregion

    # region ------------------------ Training Functions --------------------- #
    def optimize_parameters(self, *args, epoch=None, **kwargs):
        """Combined train step that applies scheduler (if attached), trains segmentation nets first,
        then runs discriminator and generator updates. Returns aggregated metrics dictionary.
        """
        self.epoch = epoch if epoch is not None else self.epoch
        self.set_input(**kwargs)
        self.set_pedestrians_color()
        self.grenner_vegetation()
        self.initialize_losses()
        # apply scheduler
        if hasattr(self, 'loss_scheduler'):
            self.loss_scheduler.step(self.epoch)

        for lam in self.losses_scheduler.current.keys():
            setattr(self, f'{lam}', self.losses_scheduler.get(f'{lam}'))

        # G_A and G_B
        self.netG.zero_grads(), self.netS.zero_grads()
        self.backward_G()
        self.netG.step_grads(*self.partial_train_net['G'])
        if self.lambda_seg > 0.0:
            self.netS.step_grads(*self.partial_train_net['S'])
        self.netD.zero_grads(*self.partial_train_net['D'])
        self.backward_D()
        self.netD.step_grads(*self.partial_train_net['D'])

    def backward_G(self):
        self.pred_real_D = self.netD(self.real_D, from_=self.D)
        self.pred_real_T = self.netD(self.real_T, from_=self.T)

        encoded_D = self.netG.encode(self.real_D, from_=self.D)
        encoded_TN, self.fake_TN, self.remapped_T, self.real_N, *other = self.netG.encode(self.real_T, self.real_N,
                                                                                          from_=self.T, epoch=self.epoch)

        # region Identity "auto-encode" loss
        if self.lambda_id > 0:
            # Same encoder and decoder should recreate image
            id_D = self.netG.decode(encoded_D, to_=self.D)
            self.loss_id[self.D] += self.compute_loss('id', id_D, self.real_D)
            id_TN = self.netG.decode(encoded_TN, to_=self.T)
            self.loss_id[self.T] += self.compute_loss('id', id_TN, self.fake_TN)
        # endregion

        # region GAN loss
        """D_T(G_T(D))"""
        self.fake_T = self.netG.decode(encoded_D, to_=self.T)
        # self.fake_T = self.netG.fusion.thermal_postprocess(self.fake_T)
        self.loss_G[self.T] += self.compute_loss('gan', partial(self.netD, from_=self.T),
                                                 self.remapped_T, self.pred_real_T, self.fake_T, False, loss_name='G')
        """D_N(G_N(T))"""
        self.loss_G[self.N] += self.compute_loss('gan', partial(self.netD, from_=self.T),
                                                 self.remapped_T, self.pred_real_T, self.fake_TN, False, loss_name='G')
        """D_D(G_D(T))"""
        self.fake_D = self.netG.decode(encoded_TN, to_=self.D)
        self.loss_G[self.D] += self.compute_loss('gan', partial(self.netD, from_=self.D), self.real_D,
                                                 self.pred_real_D, self.fake_D, False, loss_name='G')
        # endregion

        # region Cycle loss
        #  Forward
        rec_encoded_D = self.netG.encode(self.fake_T, from_=self.T)
        self.rec_D = self.netG.decode(rec_encoded_D, self.D)
        self.loss_cycle[self.D] += self.compute_loss('cycle', self.rec_D, self.real_D)
        # Backward
        rec_encoded_TN = self.netG.encode(self.fake_D, from_=self.D)
        self.rec_TN = self.netG.decode(rec_encoded_TN, self.T)
        self.rec_T = self.rec_TN
        self.loss_cycle[self.T] += self.compute_loss('cycle', self.rec_T, self.fake_TN,
                                                     loss_name='cycle', criterion_lambda='thermal')
        # endregion

        # region Cycle loss on Latent Space
        if self.lambda_latent > 0:
            self.loss_latent[self.D] += self.compute_loss('latent', rec_encoded_D, encoded_D)
            self.loss_latent[self.T] += self.compute_loss('latent', rec_encoded_TN, encoded_TN)
        # endregion

        # region Fusion Loss
        self.loss_sharpness[self.T] += self.compute_loss('sharpness', self.fake_TN, self.real_N, self.real_T)
        gray_N = .299 * self.real_N[:, 0:1, :, :] + .587 * self.real_N[:, 1:2, :, :] + .114 * self.real_N[:, 2:3, :, :]
        self.loss_fus[self.N] += self.compute_loss('cycle', self.rec_T, -gray_N.repeat(1, 3, 1, 1),
                                                   loss_name='fus', criterion_lambda='fus')
        self.loss_fus[self.T] += self.compute_loss('cycle', self.rec_T, self.remapped_T,
                                                   loss_name='fus', criterion_lambda='fus')
        if not (None in other):
            self.loss_fus[self.T] += self.compute_loss('illum', *other, self.remapped_T, self.fake_TN,
                                                   loss_name='fus', criterion_lambda='illumination_aware')
        # endregion

        # region Total Variation loss
        self.loss_tv[self.T] += self.compute_loss('tv', self.fake_T)
        self.loss_tv[self.N] += self.compute_loss('tv', self.fake_TN)
        self.loss_tv[self.D] += self.compute_loss('tv', self.fake_D)
        # endregion

        # region Segmentation Distillation Knowledge
        rand_size, seg_IR = self.backward_S()
        # endregion

        # region ACL
        # First step : Learning to translate Day color traffic lights to Thermal traffic lights
        if self.lambda_trafficlight > 0.0:
            self.D_com, self.T_com, self.N_com, segMask_com, contourMask, weights = self.merge_TL()
            total_mask = segMask_com | contourMask
            encoded_TN, self.TN_com, self.remapped_T_com, *_ = self.netG.encode(self.T_com, self.N_com,
                                                                               from_=self.T, align_first=False)
            # self.fake_T_com = self.fake_T * (~total_mask) + self.TN_com * total_mask
            self.fake_T_com = self.netG.decode(self.netG.encode(self.D_com, from_=self.D), to_=self.T).detach()
            # self.rec_D_com = self.netG.decode(self.netG.encode(self.fake_T_com, from_=self.T), to_=self.D)
            self.fake_D_com = self.netG.decode(encoded_TN, to_=self.D)
            self.rec_TN_com = self.netG.decode(self.netG.encode(self.fake_D_com, from_=self.D), to_=self.T)
            self.loss_trafficlight[self.N] += self.compute_loss('tll', self.N_com, self.remapped_T_com, self.TN_com,
                                                                self.rec_TN_com, self.D_com, self.fake_D_com,
                                                                self.fake_T_com,
                                                                segMask_com, contourMask, weights,
                                                                self.segMask_TN_update,
                                                                loss_name='trafficlight',
                                                                criterion_lambda='trafficlight')
        # endregion

        # region Structure-Gradient Alignment loss
        self.loss_sga[self.D] += self.compute_loss('sga', self.edges_D, self.get_gradmag(self.fake_T))
        self.loss_sga[self.D] += self.compute_loss('IRClsDis', self.segMask_D,
                                                   self.fake_T.mean(dim=1, keepdim=True),
                                                   criterion_lambda='ssim', loss_name='sga')
        self.loss_sga[self.D] += self.compute_loss('bc', self.segMask_D, self.segMask_TN_update,
                                                   self.fake_T, self.real_D, self.remapped_T,
                                                   self.rec_D, self.edges_D, self.get_gradmag(self.fake_T),
                                                   criterion_lambda='bc', loss_name='sga')
        self.loss_sga[self.T] += self.compute_loss('sga', self.get_gradmag(self.fake_TN), self.get_gradmag(self.fake_D))
        self.loss_sga[self.T] += self.compute_loss('IRClsDis', self.segMask_TN_update if
        self.segMask_TN_update is not None else self.segMask_TN,
                                                   self.fake_TN.mean(dim=1, keepdim=True),
                                                   criterion_lambda='ssim', loss_name='sga')
        # endregion

        # region Scale Robustness Loss
        if self.lambda_scale_robustness > 0.0:
            size = self.input_size
            h_ds, w_ds = size[0] // 2, size[1] // 2
            random_crop = RandomCrop((h_ds, w_ds))
            real_T_prep = self.real_T[..., size[0] // 5:-size[0] // 5, size[1] // 5:-size[1] // 5] * 0.5 + 0.5
            real_N_prep = self.real_N[..., size[0] // 5:-size[0] // 5, size[1] // 5:-size[1] // 5] * 0.5 + 0.5
            fake_D_prep = self.fake_D[..., size[0] // 5:-size[0] // 5, size[1] // 5:-size[1] // 5] * 0.5 + 0.5

            input_to_crop = torch.cat([real_T_prep, real_N_prep, fake_D_prep], dim=1)
            real_T_ds, real_N_ds, fake_D_ds = random_crop(input_to_crop).split([3, 3, 3], dim=1)
            fake_TN_encoded, *_ = self.netG.encode(real_T_ds, real_N_ds, from_=self.T, align_first=False)
            fake_D = self.netG.decode(fake_TN_encoded, to_=self.D)
            self.loss_scale_robustness[self.T] += self.compute_loss('cycle', fake_D, fake_D_ds,
                                                                    loss_name='scale_robustness',
                                                                    criterion_lambda='scale_robustness')
        # endregion

        # region Domain-specific losses include CGR loss and ACA loss.
        if self.netS.stage in ['freeze_all', 'trained']:
            self.loss_ds[self.T] += self.compute_loss('cgr', self.fake_D, self.segMask_TN_update,
                                                      self.fake_TN, loss_name='ds')
            self.loss_ds[self.D] += self.compute_loss('aca', self.segMask_D_update, encoded_D,
                                                      self.segMask_TN_update, rec_encoded_TN, loss_name='ds',
                                                      criterion_lambda='cgr')
        # endregion

        # region Attacks stability loss
        if self.lambda_att > 0.0:
            # att_T, att_N = self.att_input(self.real_T, self.real_N, balance=0.5, epsilon=0.25)
            # fake_D_att = self.netG.decode(self.netG.encode(att_T, att_N, from_=self.T, align_first=False)[0],
            #                               to_=self.D)
            # rec_T = self.netG.decode(self.netG.encode(fake_D_att, from_=self.D), to_=self.T)
            # self.loss_att[self.T] += self.compute_loss('att', rec_T, fake_D_att)
            att_fake_T = self.att_input(self.fake_T.mean(1, keepdim=True), epsilon=torch.rand(1) / 20).repeat(1, 3, 1,
                                                                                                              1)
            self.att_rec_D = self.netG.decode(self.netG.encode(att_fake_T, from_=self.T, align_first=False), to_=self.D)
            self.loss_att[self.T] += self.compute_loss('cycle', self.att_rec_D, self.real_D, loss_name='att',
                                                       criterion_lambda='att')
        # endregion

        # region Color/Thermal loss
        self.loss_color[self.T] += self.compute_loss('color', self.fake_D, self.real_N, self.segMask_TN_update,
                                                     weights=self.class_weight)
        self.loss_color[self.D] += self.compute_loss('color', self.rec_D, self.real_D, self.segMask_D,
                                                     weights=self.class_weight)
        self.loss_thermal[self.T] += self.compute_loss('thermal', self.fake_TN, self.remapped_T, self.real_N,
                                                       self.segMask_TN_update, weights=self.class_weight)
        self.loss_contour[self.T] += self.compute_loss('contour', self.fake_D, self.segMask_TN_update)

        if self.real_D_T is not None:
            encoded_TD, _, _, real_D, *_ = self.netG.encode(self.real_D_T, self.real_D, from_=self.T, epoch=self.epoch)
            encoded_D = self.netG.encode(real_D, from_=self.D).detach()
            self.fake_D_day = self.netG.decode(encoded_TD, to_=self.D)
            self.loss_color_day[self.T] += self.compute_loss('latent', encoded_TD, encoded_D, loss_name='color_day',
                                                             criterion_lambda='color_day')
            mask_proj = (real_D.mean(dim=1, keepdim=True) == 0.5).float() * (real_D.std(dim=1, keepdim=True) == 0).float()
            mask_lum = (real_D.mean(dim=1, keepdim=True) < 0.95).float() * (1-mask_proj)
            self.loss_color_day[self.D] += self.compute_loss('cycle', self.fake_D_day*mask_lum,
                                                             real_D*mask_lum, loss_name='color_day',
                                                             criterion_lambda='color_day')
        # endregion
        # combined loss
        self.sum_losses().backward()

    # def backward_G(self):
    #     encoded_D = self.netG.encode(self.real_D, from_=self.D)
    #     encoded_TN, self.fake_TN, self.remapped_T, self.real_N = self.netG.encode(self.real_T, self.real_N,
    #                                                                               from_=self.T, epoch=self.epoch)
    #     self.create_TN()
    #     self.pred_real_D = self.netD(self.real_D, from_=self.D)
    #     self.pred_real_T = self.netD(self.real_TN, from_=self.T)
    #
    #     # region Identity "auto-encode" loss
    #     if self.lambda_id > 0:
    #         # Same encoder and decoder should recreate image
    #         id_D = self.netG.decode(encoded_D, to_=self.D)
    #         self.loss_id[self.D] += self.compute_loss('id', id_D, self.real_D)
    #         id_TN = self.netG.decode(encoded_TN, to_=self.T)
    #         self.loss_id[self.T] += self.compute_loss('id', id_TN, self.fake_TN)
    #     # endregion
    #
    #     # region GAN loss
    #     """D_T(G_T(D))"""
    #     self.fake_T = self.netG.decode(encoded_D, to_=self.T)
    #     # self.fake_T = self.netG.fusion.thermal_postprocess(self.fake_T)
    #     self.loss_G[self.T] += self.compute_loss('gan', partial(self.netD, from_=self.T),
    #                                              self.real_TN, self.pred_real_T, self.fake_T, False, loss_name='G')
    #     """D_N(G_N(T))"""
    #     self.loss_G[self.N] += self.compute_loss('gan', partial(self.netD, from_=self.T),
    #                                              self.real_TN, self.pred_real_T, self.fake_TN, False, loss_name='G')
    #     """D_D(G_D(T))"""
    #     self.fake_D = self.netG.decode(encoded_TN, to_=self.D)
    #     self.loss_G[self.D] += self.compute_loss('gan', partial(self.netD, from_=self.D), self.real_D,
    #                                              self.pred_real_D, self.fake_D, False, loss_name='G')
    #     # endregion
    #
    #     # region Cycle loss
    #     #  Forward
    #     rec_encoded_D = self.netG.encode(self.fake_T, from_=self.T)
    #     self.rec_D = self.netG.decode(rec_encoded_D, self.D)
    #     self.loss_cycle[self.D] += self.compute_loss('cycle', self.rec_D, self.real_D)
    #     # Backward
    #     rec_encoded_TN = self.netG.encode(self.fake_D, from_=self.D)
    #     self.rec_TN = self.netG.decode(rec_encoded_TN, self.T)
    #     # self.rec_T = self.rec_TN
    #     self.loss_cycle[self.T] += self.compute_loss('cycle', self.rec_TN, self.fake_TN)
    #     # endregion
    #
    #     # region Cycle loss on Latent Space
    #     if self.lambda_latent > 0:
    #         self.loss_latent[self.D] += self.compute_loss('latent', rec_encoded_D, encoded_D)
    #         self.loss_latent[self.T] += self.compute_loss('latent', rec_encoded_TN, encoded_TN)
    #     # endregion
    #
    #     # region Fusion Loss
    #     self.loss_sharpness[self.T] += self.compute_loss('sharpness', self.fake_TN, self.real_N, self.real_T)
    #     gray_N = .299 * self.real_N[:, 0:1, :, :] + .587 * self.real_N[:, 1:2, :, :] + .114 * self.real_N[:, 2:3, :, :]
    #     self.loss_fus[self.N] += self.compute_loss('cycle', self.rec_TN[:, :1].repeat(1, 3, 1, 1),
    #                                                -gray_N.repeat(1, 3, 1, 1), loss_name='fus', criterion_lambda='fus')
    #     self.loss_fus[self.T] += self.compute_loss('cycle', self.rec_TN[:, :1].repeat(1, 3, 1, 1),
    #                                                self.remapped_T, loss_name='fus', criterion_lambda='fus')
    #     # endregion
    #
    #     # region Total Variation loss
    #     self.loss_tv[self.T] += self.compute_loss('tv', self.fake_T)
    #     self.loss_tv[self.N] += self.compute_loss('tv', self.fake_TN)
    #     self.loss_tv[self.D] += self.compute_loss('tv', self.fake_D)
    #     # endregion
    #
    #     # region Segmentation Distillation Knowledge
    #     rand_size, seg_IR = self.backward_S()
    #     # endregion
    #
    #     # region ACL
    #     if self.lambda_trafficlight_l > 0.0:
    #         self.D_com, self.T_com, self.N_com, segMask_com, contourMask, weights = self.merge_TL()
    #         encoded_TN, self.TN_com, self.remapped_T_com, _ = self.netG.encode(self.T_com, self.N_com,
    #                                                                            from_=self.T, align_first=False)
    #         self.fake_T_com = self.netG.decode(self.netG.encode(self.D_com, from_=self.D), to_=self.T).detach()
    #         self.fake_D_com = self.netG.decode(encoded_TN, to_=self.D)
    #         self.rec_TN_com = self.netG.decode(self.netG.encode(self.fake_D_com, from_=self.D), to_=self.T)
    #         self.loss_trafficlight[self.N] += self.compute_loss('tll2', self.N_com, self.remapped_T_com, self.TN_com,
    #                                                             self.rec_TN_com, self.D_com, self.fake_D_com,
    #                                                             self.fake_T_com,
    #                                                             segMask_com, contourMask, weights,
    #                                                             self.segMask_TN_update,
    #                                                             loss_name='trafficlight',
    #                                                             criterion_lambda='trafficlight_f')
    #     # endregion
    #
    #     # region Structure-Gradient Alignment loss
    #     self.loss_sga[self.D] += self.compute_loss('sga', self.edges_D, self.get_gradmag(self.fake_T))
    #     self.loss_sga[self.D] += self.compute_loss('IRClsDis', self.segMask_D,
    #                                                self.fake_T[:, :1],
    #                                                criterion_lambda='ssim', loss_name='sga')
    #     self.loss_sga[self.D] += self.compute_loss('bc', self.segMask_D, self.segMask_TN_update,
    #                                                self.fake_T, self.real_D, self.remapped_T,
    #                                                self.rec_D, self.edges_D, self.get_gradmag(self.fake_T),
    #                                                criterion_lambda='bc', loss_name='sga')
    #     self.loss_sga[self.T] += self.compute_loss('sga', self.get_gradmag(self.fake_TN), self.get_gradmag(self.fake_D))
    #     self.loss_sga[self.T] += self.compute_loss('IRClsDis', self.segMask_TN_update if
    #                                                self.segMask_TN_update is not None else self.segMask_TN,
    #                                                self.fake_TN[:, :1], criterion_lambda='ssim', loss_name='sga')
    #     # endregion
    #
    #     # region Scale Robustness Loss
    #     if self.lambda_scale_robustness > 0.0:
    #         size = self.input_size
    #         h_ds, w_ds = size[0] // 2, size[1] // 2
    #         random_crop = RandomCrop((h_ds, w_ds))
    #         real_T_prep = self.real_T[..., size[0] // 5:-size[0] // 5, size[1] // 5:-size[1] // 5] * 0.5 + 0.5
    #         real_N_prep = self.real_N[..., size[0] // 5:-size[0] // 5, size[1] // 5:-size[1] // 5] * 0.5 + 0.5
    #         fake_D_prep = self.fake_D[..., size[0] // 5:-size[0] // 5, size[1] // 5:-size[1] // 5] * 0.5 + 0.5
    #
    #         input_to_crop = torch.cat([real_T_prep, real_N_prep, fake_D_prep], dim=1)
    #         real_T_ds, real_N_ds, fake_D_ds = random_crop(input_to_crop).split([3, 3, 3], dim=1)
    #         fake_TN_encoded, _, *_ = self.netG.encode(real_T_ds, real_N_ds, from_=self.T, align_first=False)
    #         fake_D = self.netG.decode(fake_TN_encoded, to_=self.D)
    #         self.loss_scale_robustness[self.T] += self.compute_loss('cycle', fake_D, fake_D_ds,
    #                                                                 loss_name='scale_robustness',
    #                                                                 criterion_lambda='scale_robustness')
    #     # endregion
    #
    #     # region Domain-specific losses include CGR loss and ACA loss.
    #     if self.netS.stage in ['freeze_all', 'trained']:
    #         self.loss_ds[self.T] += self.compute_loss('cgr', self.fake_D, self.segMask_TN_update,
    #                                                   self.fake_TN, loss_name='ds')
    #         self.loss_ds[self.D] += self.compute_loss('aca', self.segMask_D_update, encoded_D,
    #                                                   self.segMask_TN_update, rec_encoded_TN, loss_name='ds',
    #                                                   criterion_lambda='cgr')
    #     # endregion
    #
    #     # region Attacks stability loss
    #     if self.lambda_att > 0.0:
    #         # att_T, att_N = self.att_input(self.real_T, self.real_N, balance=0.5, epsilon=0.25)
    #         # fake_D_att = self.netG.decode(self.netG.encode(att_T, att_N, from_=self.T, align_first=False)[0],
    #         #                               to_=self.D)
    #         # rec_T = self.netG.decode(self.netG.encode(fake_D_att, from_=self.D), to_=self.T)
    #         # self.loss_att[self.T] += self.compute_loss('att', rec_T, fake_D_att)
    #         att_fake_T = self.att_input(self.fake_T.mean(1, keepdim=True), epsilon=torch.rand(1) / 20).repeat(1, 3, 1,
    #                                                                                                           1)
    #         self.att_rec_D = self.netG.decode(self.netG.encode(att_fake_T, from_=self.T, align_first=False), to_=self.D)
    #         self.loss_att[self.T] += self.compute_loss('cycle', self.att_rec_D, self.real_D, loss_name='att',
    #                                                    criterion_lambda='att')
    #     # endregion
    #
    #     # region Color/Thermal loss
    #     self.loss_color[self.T] += self.compute_loss('color', self.fake_D, self.fake_TN, self.segMask_TN_update,
    #                                                  weights=self.class_weight)
    #     self.loss_color[self.D] += self.compute_loss('color', self.rec_D, self.real_D, self.segMask_D,
    #                                                  weights=self.class_weight)
    #     self.loss_thermal[self.T] += self.compute_loss('thermal', self.fake_TN, self.remapped_T, self.real_N,
    #                                                    self.segMask_TN_update, weights=self.class_weight)
    #     self.loss_contour[self.T] += self.compute_loss('contour', self.fake_D, self.segMask_TN_update)
    #
    #     if self.real_D_T is not None and self.lambda_color_day > 0.0:
    #         encoded_TD, _, _, real_D = self.netG.encode(self.real_D_T, self.real_D, from_=self.T, epoch=self.epoch)
    #         encoded_D = self.netG.encode(real_D, from_=self.D).detach()
    #         self.fake_D_day = self.netG.decode(encoded_TD, to_=self.D)
    #         self.loss_color_day[self.T] += self.compute_loss('latent', encoded_TD, encoded_D, loss_name='color_day',
    #                                                          criterion_lambda='color_day')
    #         mask_proj = (real_D.mean(dim=1, keepdim=True) == 0.5).float() * (
    #                     real_D.std(dim=1, keepdim=True) == 0).float()
    #         mask_lum = (real_D.mean(dim=1, keepdim=True) < 0.95).float() * (1 - mask_proj)
    #         self.loss_color_day[self.D] += self.compute_loss('cycle', self.fake_D_day * mask_lum,
    #                                                          real_D * mask_lum, loss_name='color_day',
    #                                                          criterion_lambda='color_day')
    #     # endregion
    #
    #     # combined loss
    #     self.sum_losses().backward()

    def backward_D(self):
        #  D_Thermal
        D = partial(self.netD, from_=self.T)
        # self.loss_D[self.T] += self.compute_loss('gan', D, self.remapped_T, self.pred_real_T,
        #                                          self.fake_T, True, loss_name='D')
        self.loss_D[self.T] += self.compute_loss('gan', D, self.real_T, self.pred_real_T,
                                                 self.fake_T, True, loss_name='D')
        #  D_Day
        D = partial(self.netD, from_=self.D)
        self.loss_D[self.D] += self.compute_loss('gan', D, self.real_D, self.pred_real_D,
                                                 self.fake_D, True, loss_name='D')
        # combined loss
        self.sum_losses().backward()

    def backward_S(self) -> tuple[int, Tensor]:
        """Random size for segmentation network training. Then, retain original image size."""
        stage = self.netS.stage
        if stage == 'freeze_all':
            rand_size = self.input_size[0]
            self.segMask_D_update = self.segMask_D
            self.segMask_TN_update = self.segMask_TN
            return rand_size, self.segMask_TN
        else:
            rand_scale = torch.randint(8, 20, (1, 1))
            rand_size = int(rand_scale.item() * self.input_size[0] / 16)

            if stage == 'train':
                real_D_s = interpolate(self.real_D, size=rand_size, mode='bilinear', align_corners=False)
                segMask_D_s = interpolate(self.segMask_D.float(), size=rand_size, mode='nearest')
                real_D_pred_seg = self.netS(real_D_s, from_=self.D)
                self.criterion_seg = self.update_class_criterion(segMask_D_s.long())
                self.loss_S[self.D] += self.compute_loss('seg', real_D_pred_seg,
                                                         segMask_D_s.long().squeeze(1), loss_name='S')
                self.segMask_D_update = segMask_D_s.long().detach()
                self.segMask_TN_update = self.segMask_TN
                IR_pred_seg = self.segMask_TN

            elif stage == 'update_D':
                # Start updating D seg labels and train Thermal Seg with pseudo TIR images and Day labels
                real_D_s = interpolate(self.real_D, size=rand_size, mode='bilinear', align_corners=False)
                fake_D_s = interpolate(self.fake_D, size=rand_size, mode='bilinear', align_corners=False)
                fake_T_s = interpolate(self.fake_T, size=rand_size, mode='bilinear', align_corners=False)
                fake_TN_s = interpolate(self.fake_TN, size=rand_size, mode='bilinear', align_corners=False)
                segMask_D_s = interpolate(self.segMask_D.float(), size=rand_size, mode='nearest').long()
                segMask_TN_s = interpolate(self.segMask_TN.float(), size=rand_size, mode='nearest').long()
                real_D_pred_seg = self.netS(real_D_s, from_=self.D)
                fake_TN_pred_seg = self.netS(fake_TN_s, from_=self.T)
                fake_D_pred_seg_d = self.netS(fake_D_s.detach(), from_=self.D)
                fake_T_pred_seg_d = self.netS(fake_T_s.detach(), from_=self.T)

                self.segMask_D_update = UpdateVisGT(fake_TN_s.detach(), segMask_D_s, 0.25).long()
                self.criterion_seg = self.update_class_criterion(self.segMask_D_update)
                ####
                self.loss_S[self.D] += self.compute_loss('seg', real_D_pred_seg,
                                                         self.segMask_D_update.squeeze(1), loss_name='S')
                self.loss_S[self.D] += self.compute_loss('semEdge', real_D_pred_seg,
                                                         self.segMask_D_update, loss_name='S')
                self.loss_seg[self.D] += self.compute_loss('seg', fake_T_pred_seg_d,
                                                           self.segMask_D_update.squeeze(1))
                mask_uncertain = segMask_TN_s == 255
                self.segMask_TN_update = (UpdateIRGTv1(fake_TN_pred_seg.detach(), fake_D_pred_seg_d,
                                                       255 * torch.ones_like(segMask_D_s), fake_TN_s) *
                                          mask_uncertain + ~mask_uncertain * segMask_TN_s)
                IR_pred_seg = fake_D_pred_seg_d

            elif stage == 'update_TN':
                real_D_s = interpolate(self.real_D, size=rand_size, mode='bilinear', align_corners=False)
                fake_T_s = interpolate(self.fake_T, size=rand_size, mode='bilinear', align_corners=False)
                fake_D_s = interpolate(self.fake_D, size=rand_size, mode='bilinear', align_corners=False)
                fake_TN_s = interpolate(self.fake_TN, size=rand_size, mode='bilinear', align_corners=False)
                segMask_D_s = interpolate(self.segMask_D.float(), size=rand_size, mode='nearest').long()
                segMask_TN_s = interpolate(self.segMask_TN.float(), size=rand_size, mode='nearest').long()
                real_D_pred_seg = self.netS(real_D_s, from_=self.D)
                fake_D_pred_seg_d = self.netS(fake_D_s.detach(), from_=self.D)
                fake_T_pred_seg_d = self.netS(fake_T_s.detach(), from_=self.T)
                fake_TN_pred_seg_d = self.netS(fake_TN_s.detach(), from_=self.T)
                self.segMask_D_update = UpdateVisGT(fake_T_s.detach(), segMask_D_s, 0.25).long()
                self.criterion_seg = self.update_class_criterion(self.segMask_D_update)
                self.loss_S[self.D] += self.compute_loss('seg', real_D_pred_seg,
                                                         self.segMask_D_update.squeeze(1), loss_name='S')
                self.loss_S[self.D] += self.compute_loss('semEdge', real_D_pred_seg,
                                                         self.segMask_D_update.squeeze(1), loss_name='S')
                self.loss_seg[self.D] += self.compute_loss('seg', fake_T_pred_seg_d,
                                                           self.segMask_D_update.squeeze(1))
                mask_uncertain = segMask_TN_s == 255
                self.segMask_TN_update = (UpdateIRGTv2(fake_TN_pred_seg_d.detach(), fake_D_pred_seg_d, segMask_TN_s,
                                          fake_TN_s, prob_th=0.9).long() * mask_uncertain + ~mask_uncertain * segMask_TN_s)
                self.criterion_seg = self.update_class_criterion(self.segMask_TN_update)
                self.loss_seg[self.T] += self.compute_loss('seg', fake_TN_pred_seg_d.squeeze(1),
                                                           self.segMask_TN_update.squeeze(1))
                IR_pred_seg = fake_D_pred_seg_d

            else:
                fake_TN_s = interpolate(self.fake_TN, size=rand_size, mode='bilinear', align_corners=False)
                fake_D_s = interpolate(self.fake_D, size=rand_size, mode='bilinear', align_corners=False)
                segMask_D_s = interpolate(self.segMask_D.float(), size=rand_size, mode='nearest').long()
                segMask_TN_s = interpolate(self.segMask_TN.float(), size=rand_size, mode='nearest').long()
                fake_D_pred_seg = self.netS(fake_D_s, from_=self.D)
                fake_TN_pred_seg = self.netS(fake_TN_s, from_=self.T)
                fake_D_pred_seg_d = self.netS(fake_D_s.detach(), from_=self.D)

                self.segMask_D_update = UpdateVisGT(fake_TN_s.detach(), segMask_D_s, 0.25).long()
                self.criterion_seg = self.update_class_criterion(self.segMask_D_update)
                self.loss_seg[self.D] += self.compute_loss('seg', fake_TN_pred_seg,
                                                           self.segMask_D_update.squeeze(1))
                mask_uncertain = segMask_TN_s == 255
                self.segMask_TN_update = (UpdateIRGTv2(fake_TN_pred_seg.detach(), fake_D_pred_seg_d, segMask_TN_s,
                                                       fake_TN_s) * mask_uncertain + ~mask_uncertain * segMask_TN_s)
                segMask_TN_update_s = interpolate(self.segMask_TN_update.float(), size=rand_size, mode='nearest').long()
                self.criterion_seg = self.update_class_criterion(segMask_TN_update_s)
                self.loss_seg[self.T] = self.compute_loss('seg', fake_D_pred_seg,
                                                          segMask_TN_update_s.squeeze(1))
                IR_pred_seg = fake_D_pred_seg_d
            return rand_size, IR_pred_seg

    def update_class_criterion(self, labels):
        # labels: (N, H, W)
        flat = labels.long().view(-1)
        device = labels.device
        # Count occurrences across the whole batch
        vals, count = flat.unique(return_counts=True)
        if 255 in vals:
            ignore_index = ((vals == 255) + (vals == 19)).nonzero(as_tuple=True)[0]
            vals = torch.cat((vals[:ignore_index], vals[ignore_index + 1:]))
            count = torch.cat((count[:ignore_index], count[ignore_index + 1:]))
        absent_count = []
        for i in range(self.netS.num_classes):
            if i not in vals:
                count = torch.cat((count, torch.tensor([0], device=device)))
                vals = torch.cat((vals, torch.tensor([i], device=device)))
                absent_count.append(i)
        weight = torch.ones(self.netS.num_classes, device=device)
        # Small-objective mask: classes with < 32×32 pixels per image on average
        n = labels.size(0)
        small_mask = count < (32 * 32 * n)
        weight[small_mask] = self.max_value
        # Often-balance: classes missing from the batch
        often = torch.ones(self.netS.num_classes, device=device)
        if self.often_balance:
            for i in absent_count:
                often[i] = self.max_value
            self.often_weight = 0.9 * self.often_weight + 0.1 * often
        else:
            self.often_weight = often
        # Final class weights
        self.class_weight = weight * self.often_weight
        return nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=255)

    def compute_loss(self, criterion: str, *inputs, criterion_lambda: str | float = None, loss_name: str = '',
                     **kwargs):
        """Compute a specific loss with given inputs and apply its lambda weight."""
        if criterion_lambda is None:
            criterion_lambda = criterion
        if not loss_name:
            loss_name = criterion
        if isinstance(criterion_lambda, float):
            if criterion_lambda == 0:
                return 0.0
        elif criterion_lambda in self.lambdas:
            criterion_lambda = self.lambdas[criterion_lambda]
            if criterion_lambda == 0:
                return 0.0
        else:
            return 0.0
        criterion = getattr(self, f'criterion_{criterion}', None)
        if criterion is None:
            raise ValueError(f"Criterion {criterion} not found.")
        self.used_losses = loss_name
        self.sum_lambdas += criterion_lambda
        return criterion(*inputs, **kwargs) * criterion_lambda

    def sum_losses(self) -> Tensor:
        sum_losses: Tensor = Tensor(
            sum(sum([v for v in self.losses[used_losses].values()]) for used_losses in self.used_losses))
        sum_lambdas = self.sum_lambdas
        self.used_losses = []
        self.sum_lambdas = -1.0  # reset
        return sum_losses / (sum_lambdas if sum_lambdas > 0 else 1.0)

    def merge_TL(self, nb=1):
        B, C, H, W = self.real_D.shape
        D = self.real_D * 0.5 + 0.5
        segMask_D = self.segMask_D.clone()
        T, N = self.real_T * 0.5 + 0.5, self.real_N * 0.5 + 0.5
        segMask_TN = interpolate(self.segMask_TN_update.clone().float(), size=T.shape[-2:], mode='nearest')
        already_present_TL = (segMask_TN == 6).float()
        contour_mask = torch.zeros_like(segMask_TN)
        weights = torch.zeros_like(segMask_TN)
        if already_present_TL.sum() > 0:
            # check the rectangle shape of the mask
            labels = connected_components(already_present_TL.float())

            for b in range(B):
                uniques = labels[b].unique(return_counts=True)
                for i, (label, count) in enumerate(zip(uniques[0], uniques[1])):
                    if i == 0:
                        continue
                    mask_ = labels[b] == label
                    if count < 75:
                        segMask_TN[b][mask_] = 255
                    else:
                        ys, xs = (mask_[0]).nonzero().permute(1, 0)
                        y0, y1 = ys.min(), ys.max()
                        x0, x1 = xs.min(), xs.max()
                        rect_area = (y1 - y0 + 1) * (x1 - x0 + 1)
                        ratio_xy = (y1 - y0 + 1) / (x1 - x0 + 1)
                        if rect_area / count > 0.95 and 1.2 < ratio_xy < 5.:
                            color = determine_color_N(self.real_N[b, :, y0:y1 + 1, x0:x1 + 1])
                            TL = self.TL_collection[color]
                            fake_light = TL['D'][torch.randint(0, len(TL['D']), [1])].to(D.device)
                            if count < 250:
                                fake_light = dilation(fake_light, torch.ones((3, 3), device=fake_light.device))
                            fake_light = interpolate(fake_light, (y1 - y0 + 1, x1 - x0 + 1))
                            mask_road = (interpolate(self.segMask_TN_update[b:b + 1].float(), self.fake_D.shape[-2:],
                                                     mode='nearest') == 0).float()
                            road_mean = ((self.fake_D[b:b + 1] * 0.5 + 0.5) * mask_road).mean()
                            fake_light_norm = (fake_light - fake_light.min()) / (
                                        fake_light.max() - fake_light.min() + 1e-6)
                            fake_light = (fake_light_norm * (1 - road_mean + 1e-5) + road_mean).clamp(0, 1)
                            D[b:b + 1, :, y0:y1 + 1, x0:x1 + 1] = fake_light
                            nb -= 1
                            disk = get_disk_kernel(radius=max(int(math.sqrt(rect_area)), 3), device=fake_light.device)
                            x0 = (x1 + x0) // 2 - disk.shape[-1] // 2
                            x1 = x0 + disk.shape[-1]
                            y0 = (y1 + y0) // 2 - disk.shape[-2] // 2
                            y1 = y0 + disk.shape[-2]
                            if x0 < 0:
                                disk = disk[:, -x0:]
                                x0 = 0
                            if y0 < 0:
                                disk = disk[-y0:, :]
                                y0 = 0
                            if x1 > W:
                                disk = disk[:, :W - x1 + disk.shape[-1]]
                                x1 = W
                            if y1 > H:
                                disk = disk[:H - y1 + disk.shape[-2], :]
                                y1 = H
                            contour_mask[b:b + 1, :, y0:y1, x0:x1] = disk
                            contour_mask -= mask_.float()
                            weights += mask_[None].float() * 5.
                        else:
                            segMask_TN[b][labels[b] == label] = 255
        if nb > 0:
            for i in range(nb):
                rand_idx = torch.rand(1)
                if rand_idx < 0.40:
                    color = 'green'
                elif rand_idx > 0.60:
                    color = 'red'
                else:
                    color = 'orange'
                TL = self.TL_collection[color]
                TL_D = TL['D'][torch.randint(0, len(TL['D']), [1])].to(D.device)
                idx = torch.randint(0, len(TL['T']), [1])
                TL_T = TL['T'][idx].to(T.device)
                TL_N = TL['N'][idx].to(T.device)
                min_scale = 9 / TL_T.shape[-1]
                scale_size = max(torch.randint(25, 200, (1,)).item() / 100, min_scale)
                TL_T_ = interpolate(TL_T, scale_factor=scale_size, mode='bilinear', align_corners=False)
                TL_N_ = interpolate(TL_N, scale_factor=scale_size, mode='bilinear', align_corners=False)
                TL_D_ = TL_D.match_shape(TL_T_)
                valid_pos_TN = (segMask_TN < 2).float() + (segMask_TN == 8).float() + (segMask_TN == 10).float()
                valid_pos_D = (segMask_D < 2).float() + (segMask_D == 8).float() + (segMask_D == 10).float()
                valid_pos = valid_pos_TN * valid_pos_D
                valid_pos[..., :TL_N_.shape[-2], :], valid_pos[..., -TL_N_.shape[-2]:, :] = 0, 0
                valid_pos[..., -TL_N_.shape[-1]:], valid_pos[..., :TL_N_.shape[-1]] = 0, 0
                if valid_pos.sum() == 0:
                    y_idxs, x_idxs = torch.randint(0, T.shape[-2] - TL_N_.shape[-2], (1,)), torch.randint(0,
                                                                                                          T.shape[-1] -
                                                                                                          TL_N_.shape[
                                                                                                              -1],
                                                                                                          (1,))
                    y = y_idxs.item()
                    x = x_idxs.item()
                else:
                    y_idxs, x_idxs = valid_pos.nonzero()[:, 2:].permute(1, 0)
                    idx = torch.randint(0, len(y_idxs), (1,))
                    y = y_idxs[idx].item()
                    x = x_idxs[idx].item()
                y = max(min(N.shape[-2] - TL_N_.shape[-2] // 2 - 1, y), TL_N_.shape[-2] // 2 + 1)
                x = max(min(N.shape[-1] - TL_N_.shape[-1] // 2 - 1, x), TL_N_.shape[-1] // 2 + 1)
                y1_TD = y + TL_T_.shape[-2] // 2
                x1_TD = x + TL_T_.shape[-1] // 2
                x0_TD = x1_TD - TL_T_.shape[-1]
                y0_TD = y1_TD - TL_T_.shape[-2]

                TL_D_ = TL_D_ ** (torch.rand(1).item() * 0.4 + 0.8)
                T[:, :, y0_TD:y1_TD, x0_TD:x1_TD] = TL_T_
                N[:, :, y0_TD:y1_TD, x0_TD:x1_TD] = TL_N_
                D[:, :, y0_TD:y1_TD, x0_TD:x1_TD] = TL_D_
                segMask_TN[:, :, y0_TD:y1_TD, x0_TD:x1_TD] = 6

                # y1_N = y + TL_N_.shape[-2] // 2
                # x1_N = x + TL_N_.shape[-1] // 2
                # x0_N = x1_N - TL_N_.shape[-1]
                # y0_N = y1_N - TL_N_.shape[-2]
                # real = N[:, :, y0_N:y1_N, x0_N:x1_N]
                # if TL_T.shape[-1] != TL_N_.shape[-1]:
                #     mask_ori = torch.zeros_like(TL_N_[:, :1])
                #     mask_ori[:, :, TL_N_.shape[-2] // 2 - TL_T_.shape[-2] // 2:TL_N_.shape[-2] // 2 + TL_T_.shape[-2] // 2,
                #     TL_N_.shape[-1] // 2 - TL_T_.shape[-1] // 2:TL_N_.shape[-1] // 2 + TL_T_.shape[-1] // 2] = 1.0
                # else:
                #     mask_ori = torch.ones_like(TL_N_[:, :1])
                # TL_N_mean = (TL_N_.mean(1, keepdim=True) * mask_ori).sum(dim=[1, 2, 3]) / mask_ori.sum()
                # C_intensity = TL_N_.max(1, keepdim=True)[0] - TL_N_.min(1, keepdim=True)[0]
                # mask = (TL_N_.mean(1, keepdim=True) + mask_ori + C_intensity).max(1, keepdim=True)[0].clamp(0, 1)
                # N[:, :, y0_N:y1_N, x0_N:x1_N] = TL_N_ * mask + real * (1 - mask)
                contour_mask[:, :, y0_TD:y1_TD, x0_TD:x1_TD] = 1
                weights[:, :, y0_TD:y1_TD, x0_TD:x1_TD] = 0.5

        contour_mask = (dilation(contour_mask.float(),
                                 get_disk_kernel(3, contour_mask.device)) - (segMask_TN == 6).float()).bool()

        return ((D * 2 - 1).detach(), (T * 2 - 1).detach(), (N * 2 - 1).detach(), (segMask_TN == 6).detach(),
                contour_mask.detach(), weights.detach())

    def create_TN(self):
        L = self.real_T.mean(1, keepdim=True)
        AB = rgb_to_lab(self.real_N * 0.5 + 0.5)[:, 1:] / 128
        self.real_TN = torch.cat([L, AB], dim=1)

    # endregion

    # region ------------------------ Training Helpers ----------------------- #
    def visualize_current_results(self, save=False):
        visuals = {'real_D': (self.D_com * 0.5 + 0.5 if self.D_com is not None else self.real_D * 0.5 + 0.5),
                   'real_T': (self.T_com * 0.5 + 0.5 if self.T_com is not None else self.real_T * 0.5 + 0.5),
                   'real_N': (self.N_com * 0.5 + 0.5 if self.N_com is not None else self.real_N * 0.5 + 0.5),
                   'fake_T': (self.fake_T_com * 0.5 + 0.5 if self.fake_T_com is not None else self.fake_T * 0.5 + 0.5),
                   'remapped_T': (
                       self.remapped_T_com * 0.5 + 0.5 if self.remapped_T_com is not None else self.remapped_T * 0.5 + 0.5),
                   'fake_TN': (self.TN_com * 0.5 + 0.5 if self.TN_com is not None else self.fake_TN * 0.5 + 0.5),
                   'rec_D': (self.fake_D_day * 0.5 + 0.5 if self.fake_D_day is not None else self.rec_D * 0.5 + 0.5),
                   'rec_T': (self.rec_TN_com * 0.5 + 0.5 if self.rec_TN_com is not None else self.rec_TN * 0.5 + 0.5),
                   'fake_D': (self.fake_D_com * 0.5 + 0.5 if self.fake_D_com is not None else self.fake_D * 0.5 + 0.5)}
        out = {lab: ImageTensor(im[0]) for lab, im in visuals.items() if im is not None}
        # out = {lab: ImageTensor(im[0], colorspace='LAB' if ('fake_T' in lab or 'rec_T' in lab) else 'RGB') for lab, im in visuals.items() if im is not None}
        out = self.visualizer.display_current_results(out)
        if save:
            self.visualizer.save_current_results({'Training': out}, self.epoch)

    def get_current_errors(self):
        return OrderedDict(
            [(f'{key}', {k: round(float(v), 4) for k, v in value.items()}) for key, value in self.losses.items()])

    # endregion

    # region ------------------------ Properties ----------------------------- #

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.netS.epoch = value

    @property
    def D(self):
        return self.names_domains[0]

    @property
    def T(self):
        return self.names_domains[1]

    @property
    def N(self):
        return self.names_domains[2]

    @property
    def losses(self):
        return {loss.replace('loss_', ''): getattr(self, f'{loss}') for loss in self.__dict__ if
                loss.startswith('loss_')}

    @property
    def lambdas(self):
        return {lam.replace('lambda_', ''): getattr(self, f'{lam}') for lam in self.__dict__ if
                lam.startswith('lambda_')}

    @property
    def used_losses(self):
        return self._used_losses

    @used_losses.setter
    def used_losses(self, value):
        if isinstance(value, list):
            self._used_losses = value
        elif isinstance(value, str):
            if value not in self._used_losses:
                if value not in self.losses:
                    raise ValueError(f"Loss {value} not found in losses.")
                self._used_losses.append(value)
        else:
            raise ValueError("used_losses should be a list or a single string.")

    @property
    def sum_lambdas(self):
        return self._sum_lambdas

    @sum_lambdas.setter
    def sum_lambdas(self, value):
        if value < 0:
            self._sum_lambdas = 0.0
        else:
            self._sum_lambdas = value

    @property
    def partial_train_net(self):
        return self._partial_train_net

    @partial_train_net.setter
    def partial_train_net(self, value):
        if isinstance(value, dict):
            self._partial_train_net = value
        elif isinstance(value, list):
            self._partial_train_net = {'G': [i for i in value if i < len(self.netG.names)],
                                       'D': [i for i in value if i < len(self.netD)],
                                       'S': [i for i in value if i < len(self.netS)]}
        else:
            raise ValueError("partial_train_net should be a list of integers or a dict of list.")
        self.netG.train(*self.partial_train_net['G'])
        self.netD.train(*self.partial_train_net['D'])
        self.netS.train(*self.partial_train_net['S'])
    # endregion
