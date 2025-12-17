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
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from ImagesCameras import ImageTensor
from torch import Tensor
from torch.nn.functional import interpolate, relu
from . import get_config
from NightToday import OptImage2ImageGATConfig
from NightToday.losses import GANLoss, SSIM_Loss, TVLoss, StructuralGradientLoss, \
    FakeIRPersonLoss, BiasCorrLoss, ColorLoss, CondGradRepaLoss, AdaptativeColAttentionLoss, SemEdgeLoss, \
    ThermalLoss, SharpFusionLoss
from NightToday.modules import LossScheduler, Get_gradmag_gray
from NightToday.plexers import G_Plexer, D_Plexer, S_Plexer
from NightToday.utilities import UpdateVisGT, UpdateIRGTv1, UpdateIRGTv2, AttackImages, get_FG_MergeMask
from NightToday.visualizers import Visualizer


# ------------------------ Main GAT Class ------------------------ #


class Image2ImageGAT_Dual(nn.Module):
    """
    Full training-ready module for thermal+night -> day translation.
    """
    _partial_train_net: dict[str, list[int]]

    def __init__(self, opt: OptImage2ImageGATConfig | str | Path | dict | None = None,
                 *args, trainable: bool = True, **kwargs):
        # region initialization sequence
        # If building from checkpoint, load config from the checkpoint given by resume_epoch
        super().__init__()
        checkpoint = self.initialization(opt, *args, **kwargs)
        self.opt.model.gen.fusion_first = self.opt.model.fusion_first
        self.opt.model.discr.fusion_first = self.opt.model.fusion_first
        self.opt.model.seg.fusion_first = self.opt.model.fusion_first
        self.model_name = self.opt.model.gen.type
        self.names_domains = self.opt.model.names_domains
        self.mode = self.opt.model.mode if trainable else 'test'
        self.checkpoint_dir = self.opt.training.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.visualize_dir = self.opt.training.visualize_dir
        os.makedirs(self.visualize_dir, exist_ok=True)
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
            self.criterion_mean_var = lambda f_tn, r_t: relu(torch.abs(f_tn.mean() - r_t.mean() - 0.1)) + relu(
                torch.abs(f_tn.std() - r_t.std()) - 0.05)
            self.criterion_latent = lambda y, t: self.L1(y, t.detach())
            self.criterion_ssim = lambda x, y: SSIM_Loss()((x + 1) / 2, (y + 1) / 2) * self.lambda_ssim
            self.criterion_tv = TVLoss(TVLoss_weight=1)
            self.criterion_color = ColorLoss
            self.criterion_thermal = ThermalLoss
            self.criterion_att = lambda rec, fake: self.criterion_cycle(rec, self.real_TN) + self.criterion_cycle(fake,
                                                                                                                  self.fake_D.detach())
            self.criterion_detail = lambda f_d, r_t: self.L1(self.get_gradmag(f_d), self.get_gradmag(r_t.detach()))
            # self.criterion_milo = lambda f_d, r_t, m=None: MILO_Loss(self.device)(f_d, r_t.detach(), m)
            self.criterion_semEdge = partial(SemEdgeLoss, num_classes=self.netS.num_classes)
            self.criterion_sharpness = SharpFusionLoss()
            self.criterion_cgr = lambda f_d, seg_t, r_t: CondGradRepaLoss(f_d,
                                                                          seg_t.detach() if seg_t is not None else seg_t,
                                                                          self.get_gradmag(f_d),
                                                                          self.get_gradmag(r_t.detach()))
            self.criterion_aca = lambda r_v, f_v, f_v_m, f_v_f: AdaptativeColAttentionLoss(r_v, f_v.detach(),
                                                                                           f_v_m.detach() if f_v_m is not None else f_v_m,
                                                                                           f_v_f,
                                                                                           4, 100000) if isinstance(
                f_v_f, Tensor) else sum(
                [AdaptativeColAttentionLoss(r_v, f_v[i].detach(), f_v_m.detach(), f_v_f[i], 4, 100000)] for i in
                range(len(f_v_f))) / len(f_v_f)
            self.criterion_sga = StructuralGradientLoss(8, 0.8)
            self.criterion_IRClsDis = FakeIRPersonLoss
            self.criterion_bc = BiasCorrLoss

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
            if 'yaml' in opt or 'yml' in opt or 'json' in opt:
                self.opt = get_config(opt)
        else:
            checkpoint = opt
            self.opt = get_config()
            if isinstance(checkpoint, (str, Path)) and os.path.isfile(checkpoint):
                checkpoint = torch.load(checkpoint, weights_only=False, map_location='cpu')
                self.opt.model = checkpoint['config']
            elif isinstance(checkpoint, dict):
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
            save_filename = f'{epoch}_net_{self.opt.model.gen.type}'
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
                    save_filename = f'{epoch}_net_{self.opt.model.gen.type}'
                    save_path = os.path.join(self.opt.training.checkpoint_dir, save_filename)
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
        setattr(self, 'real_T', kwargs.get('T', ImageTensor.rand(1, 3, 4, 4)).to(self.device))
        setattr(self, 'real_N', kwargs.get('N', ImageTensor.rand(1, 3, 4, 4)).to(self.device))
        setattr(self, 'real_TN', torch.cat([self.real_T, self.real_N], dim=1))
        setattr(self, 'segMask_D', kwargs.get('seg_D', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'segMask_TN', kwargs.get('seg_TN', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'edges_D', kwargs.get('edges_D', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'edges_TN', kwargs.get('edges_TN', ImageTensor.rand(1, 1, 4, 4)).to(self.device))
        setattr(self, 'segMask_D_update', None)
        setattr(self, 'segMask_TN_update', None)
        setattr(self, 'pred_real_D', None)
        setattr(self, 'pred_real_T', None)
        setattr(self, 'fake_D', None)
        setattr(self, 'fake_T', None)
        setattr(self, 'rec_D', None)
        setattr(self, 'rec_T', None)

    def initialize_losses(self):
        setattr(self, 'loss_D', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_G', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_S', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_cycle', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_id', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_color', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_grad', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_sga', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_tv', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_ds', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_latent', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_mean_std', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_seg', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_att', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_scale_robustness', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_mean_var', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_milo', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_thermal', {k: 0. for k in self.names_domains})
        setattr(self, 'loss_sharpness', {k: 0. for k in self.names_domains})

    # endregion

    # region ------------------------ Inference Function -------------------- #
    @torch.no_grad()
    def forward(self, data, return_fused_IR=False):
        thermal = data.get('T', ImageTensor.rand(1, 3, 256, 256)).to(self.device)
        night = data.get('N', ImageTensor.rand(1, 3, 256, 256)).to(self.device)
        encoded_TN, fused_IR, _ = self.netG.encode(thermal, night, from_=self.T)
        fake_D = self.netG.decode(encoded_TN, to_=self.D)
        if return_fused_IR:
            return fake_D*0.5+0.5, fused_IR*0.5+0.5
        return fake_D*0.5+0.5

    # endregion

    # region ------------------------ Training Functions --------------------- #
    def optimize_parameters(self, *args, epoch=None, **kwargs):
        """Combined train step that applies scheduler (if attached), trains segmentation nets first,
        then runs discriminator and generator updates. Returns aggregated metrics dictionary.
        """
        self.epoch = epoch if epoch is not None else self.epoch
        self.set_input(**kwargs)
        self.initialize_losses()
        # apply scheduler
        if hasattr(self, 'loss_scheduler'):
            self.loss_scheduler.step(self.epoch)

        for lam in self.losses_scheduler.current.keys():
            setattr(self, f'{lam}', self.losses_scheduler.get(f'{lam}'))
        self.pred_real_D = self.netD(self.real_D, from_=self.D)
        self.pred_real_T = self.netD(self.real_T, from_=self.T)
        self.pred_real_N = self.netD(self.real_N, from_=self.N) if not self.opt.model.fusion_first else None

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
        encoded_D = self.netG.encode(self.real_D, from_=self.D)
        encoded_TN, self.real_TN, self.real_N = self.netG.encode(self.real_T, self.real_N, from_=self.T)

        # region Fusion Loss
        if self.opt.model.fusion_first:
            self.loss_sharpness[self.T] += self.compute_loss('sharpness', self.real_TN, self.real_N, self.real_T)
        # endregion

        # region Identity "auto-encode" loss
        if self.lambda_id > 0:
            # Same encoder and decoder should recreate image
            id_D = self.netG.decode(encoded_D, to_=self.D)
            self.loss_id[self.D] += self.compute_loss('id', id_D, self.real_D)
            id_TN = self.netG.decode(encoded_TN, to_=self.T)
            # self.loss_id[self.T] += self.compute_loss('id', id_TN, self.real_T)
            self.loss_id[self.T] += self.compute_loss('id', id_TN, self.real_TN)
        # endregion

        # region GAN loss
        """D_T(G_T(D))"""
        self.fake_TN = self.netG.decode(encoded_D, to_=self.T)
        self.fake_T, self.fake_N = (self.fake_TN[:, :3], self.fake_TN[:, 3:]) if not self.opt.model.fusion_first else (
        self.fake_TN, None)

        self.loss_G[self.T] += self.compute_loss('gan', partial(self.netD, from_=self.T), self.real_T,
                                                 self.pred_real_T, self.fake_T, False, loss_name='G')
        """D_N(G_N(T))"""
        if self.fake_N is not None:
            self.loss_G[self.N] += self.compute_loss('gan', partial(self.netD, from_=self.N), self.real_N,
                                                     self.pred_real_N, self.fake_N, False, loss_name='G')
        else:
            self.loss_G[self.N] += self.compute_loss('gan', partial(self.netD, from_=self.T), self.real_T,
                                                     self.pred_real_T,
                                                     self.real_TN if self.fake_N is None else self.fake_N
                                                     , False, loss_name='G')
        """D_D(G_D(T))"""
        self.fake_D = self.netG.decode(encoded_TN, to_=self.D)
        self.loss_G[self.D] += self.compute_loss('gan', partial(self.netD, from_=self.D), self.real_D,
                                                 self.pred_real_D, self.fake_D, False, loss_name='G')
        # endregion

        # region Cycle loss
        #  Forward
        if self.fake_N is not None:
            rec_encoded_D, *_ = self.netG.encode(self.fake_T, self.fake_N, from_=self.T)
        else:
            rec_encoded_D = self.netG.encode(self.fake_T, from_=self.T)
        self.rec_D = self.netG.decode(rec_encoded_D, self.D)
        self.loss_cycle[self.D] += self.compute_loss('cycle', self.rec_D, self.real_D)
        # Backward
        rec_encoded_TN = self.netG.encode(self.fake_D, from_=self.D)
        self.rec_TN = self.netG.decode(rec_encoded_TN, self.T)
        self.rec_T, self.rec_N = (self.rec_TN[:, :3], self.rec_TN[:, 3:]) if self.rec_TN.shape[1] == 6 else (
        self.rec_TN, None)
        self.loss_cycle[self.T] += self.compute_loss('cycle', self.rec_T, self.real_TN, loss_name='cycle')
        if self.rec_N is not None:
            self.loss_cycle[self.N] += self.compute_loss('cycle', self.rec_N, self.real_N, loss_name='cycle')
        # endregion

        # region Cycle loss on Latent Space
        if self.lambda_latent > 0:
            self.loss_latent[self.D] += self.compute_loss('latent', rec_encoded_D, encoded_D)
            self.loss_latent[self.T] += self.compute_loss('latent', rec_encoded_TN, encoded_TN)
        # endregion

        # region Total Variation loss
        self.loss_tv[self.T] += self.compute_loss('tv', self.fake_T)
        self.loss_tv[self.N] += self.compute_loss('tv', self.real_TN) if self.fake_N is None else (
            self.compute_loss('tv', self.fake_N))
        self.loss_tv[self.D] += self.compute_loss('tv', self.fake_D)
        # endregion

        # region Segmentation Distillation Knowledge
        rand_size = self.backward_S()
        # endregion

        # region Structure-Gradient Alignment loss
        self.loss_sga[self.D] += self.compute_loss('sga', self.edges_D, self.get_gradmag(self.fake_T))
        self.loss_sga[self.D] += self.compute_loss('IRClsDis', self.segMask_D,
                                                   self.fake_T.mean(dim=1, keepdim=True),
                                                   criterion_lambda='ssim', loss_name='sga')
        self.loss_sga[self.D] += self.compute_loss('bc', self.segMask_D, self.fake_T, self.real_D,
                                                   self.rec_D, self.edges_D, self.get_gradmag(self.fake_T),
                                                   criterion_lambda='ssim', loss_name='sga')
        self.loss_sga[self.T] += self.compute_loss('sga', self.get_gradmag(self.real_TN), self.get_gradmag(self.fake_D))
        self.loss_sga[self.T] += self.compute_loss('IRClsDis', self.segMask_TN_update if
        self.segMask_TN_update is not None else self.segMask_TN,
                                                   self.real_TN.mean(dim=1, keepdim=True),
                                                   criterion_lambda='ssim', loss_name='sga')
        # endregion

        # region Scale Robustness Loss
        # if self.netS.stage in ['update_D', 'update_TN', 'trained']:
        #     if torch.rand(1) > 0.5:
        #         real_D_s = interpolate(self.real_D, size=[128, 128], mode='bilinear', align_corners=False)
        #         real_TN_s = interpolate(self.real_TN, size=[128, 128], mode='bilinear', align_corners=False)
        #         fake_TN_s = self.netG.decode(self.netG.encode(real_D_s, from_=self.D), to_=self.T)
        #         fake_D_s = self.netG.decode(self.netG.encode(real_TN_s, from_=self.T), to_=self.D)
        #         fake_D = interpolate(self.fake_D, size=[128, 128], mode='bilinear', align_corners=False)
        #         fake_TN = interpolate(self.fake_TN, size=[128, 128], mode='bilinear', align_corners=False)
        #     else:
        #         real_D_s = interpolate(self.real_D, size=[384, 384], mode='bilinear', align_corners=False)
        #         real_TN_s = interpolate(self.real_TN, size=[384, 384], mode='bilinear', align_corners=False)
        #         fake_TN_s = interpolate(self.netG.decode(self.netG.encode(real_D_s, from_=self.D), to_=self.T),
        #                                 size=[256, 256], mode='bilinear', align_corners=False)
        #         fake_D_s = interpolate(self.netG.decode(self.netG.encode(real_TN_s, from_=self.T), to_=self.D),
        #                                size=[256, 256], mode='bilinear', align_corners=False)
        #         fake_D = self.fake_D
        #         fake_TN = self.fake_TN
        #
        #     self.loss_scale_robustness[self.D] = self.compute_loss('cycle', fake_TN_s, fake_TN.detach())
        #     self.loss_scale_robustness[self.T] = self.compute_loss('cycle', fake_D_s, fake_D.detach())
        # endregion

        # region Domain-specific losses include CGR loss and ACA loss.
        # if self.netS.stage in ['freeze_all', 'trained']:
        self.loss_ds[self.T] += self.compute_loss('cgr', self.fake_D, self.segMask_TN_update,
                                                  self.real_TN, loss_name='ds')
        self.loss_ds[self.D] += self.compute_loss('aca', self.segMask_D_update, encoded_D,
                                                  self.segMask_TN_update, rec_encoded_TN, loss_name='ds',
                                                  criterion_lambda='cgr')
        # endregion

        # region Attacks stability loss
        if self.lambda_att > 0.0:
            if self.opt.model.fusion_first:
                self.loss_att[self.T] += self.compute_loss('cycle', self.real_TN,
                                                           -self.real_N.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1),
                                                           loss_name='att', criterion_lambda='att')
                self.loss_att[self.T] += self.compute_loss('cycle', self.real_TN, self.real_T,
                                                           loss_name='att', criterion_lambda='att')

        #     # att_T, att_N = self.att_input(self.real_T, self.real_N, balance=0.5, epsilon=0.25)
        #     # fake_D_att = self.netG.decode(self.netG.encode(att_T, att_N, from_=self.T), to_=self.D)
        #     # rec_T = self.netG.decode(self.netG.encode(fake_D_att, from_=self.D), to_=self.T)
        #     # self.loss_att[self.T] += self.compute_loss('att', rec_T, fake_D_att)
        #     att_T, att_N = self.att_input(self.fake_T, self.fake_N, balance=0.5, epsilon=0.5)
        #     rec_D_att = self.netG.decode(self.netG.encode(att_T,  att_N, from_=self.T), to_=self.D)
        #     self.loss_att[self.T] += self.compute_loss('color', rec_D_att, self.real_D, self.segMask_D,
        #                                                loss_name='att', criterion_lambda='att')
        #     self.loss_att[self.T] += self.compute_loss('cycle', rec_D_att, self.real_D,
        #                                                loss_name='att', criterion_lambda='att')
        # endregion

        # region ACL
        # if self.netS.stage in ['trained']: #'update_TN',
        #     fake_D_Mask = interpolate(fake_D_pred_seg_d.float(), size=[256, 256], mode='bilinear')
        #     ##########Fake_IR_Composition, OAMix-TIR
        #     valid, FG_Mask, FG_FakeTN, FG_RealVis, HL_Mask, ComIR_Light_Mask = \
        #         get_FG_MergeMask(self.segMask_D, fake_D_Mask, self.real_D, self.fake_TN.detach())
        #     if valid:
        #         IR_com = self.get_IR_Com(FG_Mask, FG_FakeTN, self.real_TN.detach(),
        #                                       self.segMask_TN_update.detach(), HL_Mask)
        #         fake_D_com = self.netG.decode(self.netG.encode(self.IR_com, self.T), self.D)
        #         loss_ACL_B += self.criterionPixCon(fake_D_com, FG_RealVis, FG_FakeTN[:, 3:])
        #         Com_RealVis = out_FG_RealVis + out_FG_RealVis_flip
        #         ###Traffic Light Luminance Loss
        #         loss_tll = self.criterionTLL(self.fake_A, self.SegMask_B_update.detach(), self.real_B.detach(),
        #                                      self.gpu_ids[0])
        #         ####Traffic light color loss
        #         loss_TLight_color = self.criterionTLC(self.real_B, self.fake_A, self.SegMask_B_update.detach(), \
        #                                               Com_RealVis, ComIR_Light_Mask, HL_Mask, self.gpu_ids[0])
        #         loss_TLight_appe = loss_tll + loss_TLight_color
        #         ####Appearance consistency loss of domain B
        #         self.loss_AC[self.DB] = loss_ACL_B + loss_ACL_B_flip + self.criterionComIR(FakeIR_FG_Mask,
        #                                                                                    FakeIR_FG_Mask_flip, \
        #                                                                                    self.SegMask_B_update.detach(),
        #                                                                                    self.IR_com, self.fake_A_IR_com,
        #                                                                                    self.gpu_ids[0])
        #
        #         FakeVis_FG_Mask, FakeVis_FG_Mask_flip, _ = self.get_FG_MergeMaskVis(fake_A_Mask, self.SegMask_A.detach(),
        #                                                                             self.gpu_ids[0])
        #         self.Vis_com = (torch.ones_like(FakeVis_FG_Mask) - FakeVis_FG_Mask - FakeVis_FG_Mask_flip).mul(
        #             self.real_A) + \
        #                        FakeVis_FG_Mask.mul(self.fake_A) + FakeVis_FG_Mask_flip.mul(
        #             torch.flip(self.fake_A.detach(), dims=[3]))
        #         ###########
        #
        #         encoded_Vis_com = self.netG.encode(self.Vis_com, self.DA)
        #         self.fake_B_Vis_com = self.netG.decode(encoded_Vis_com, self.DB)
        #
        #         if torch.sum(FakeVis_FG_Mask) > 0.0:
        #             loss_ACL_A = self.criterionPixCon(self.fake_B_Vis_com, self.real_B, FakeVis_FG_Mask,
        #                                               self.opt.ssim_winsize)
        #         else:
        #             loss_ACL_A = 0.0
        #
        #         if torch.sum(FakeVis_FG_Mask_flip) > 0.0:
        #             loss_ACL_A_flip = self.criterionPixCon(self.fake_B_Vis_com, torch.flip(self.real_B, dims=[3]),
        #                                                    FakeVis_FG_Mask_flip, self.opt.ssim_winsize)
        #         else:
        #             loss_ACL_A_flip = 0.0
        #         ####Appearance consistency loss of domain A
        #         self.loss_AC[self.DA] = loss_ACL_A + loss_ACL_A_flip
        #         ##############################
        # endregion

        # region Color/Thermal loss
        self.loss_color[self.T] += self.compute_loss('color', self.fake_D, self.real_N, self.segMask_TN_update,
                                                     weights=self.class_weight)
        self.loss_color[self.D] += self.compute_loss('color', self.rec_D, self.real_D, self.segMask_D,
                                                     weights=self.class_weight)
        self.loss_thermal[self.T] += self.compute_loss('thermal', self.real_TN, self.real_T, self.real_N,
                                                       self.segMask_TN_update, weights=self.class_weight)
        # endregion

        # region Perception Loss (MILO)
        # mask_wo_sign_wo_road = (self.segMask_TN_update!=0) * (self.segMask_TN_update!=7) \
        #     if self.segMask_TN_update is not None else None
        # self.loss_milo[self.T] += self.compute_loss('milo', self.real_TN, self.real_T, mask_wo_sign_wo_road)
        # self.loss_milo[self.D] += self.compute_loss('milo', self.rec_D, self.real_D)
        # endregion

        # combined loss
        self.sum_losses().backward()

    def backward_D(self):
        #  D_Night
        if self.pred_real_N is not None:
            D = partial(self.netD, from_=self.N)
            self.loss_D[self.N] += self.compute_loss('gan', D, self.real_N, self.pred_real_N,
                                                     self.fake_N, True, loss_name='D')
        #  D_Thermal
        D = partial(self.netD, from_=self.T)
        self.loss_D[self.T] += self.compute_loss('gan', D, self.real_T, self.pred_real_T,
                                                 self.fake_T, True, loss_name='D')
        #  D_Day
        D = partial(self.netD, from_=self.D)
        self.loss_D[self.D] += self.compute_loss('gan', D, self.real_D, self.pred_real_D,
                                                 self.fake_D, True, loss_name='D')
        # combined loss
        self.sum_losses().backward()

    def backward_S(self) -> int:
        """Random size for segmentation network training. Then, retain original image size."""
        stage = self.netS.stage
        if stage == 'freeze_all':
            rand_size = 256
            self.segMask_D_update = self.segMask_D
            self.segMask_TN_update = self.segMask_TN
            return rand_size
        else:
            rand_scale = torch.randint(8, 20, (1, 1))
            rand_size = int(rand_scale.item() * 16)

            real_D_s = interpolate(self.real_D, size=rand_size, mode='bilinear', align_corners=False)
            real_T_s = interpolate(self.real_T, size=rand_size, mode='bilinear', align_corners=False)
            real_TN_s = interpolate(self.real_TN, size=rand_size, mode='bilinear', align_corners=False)
            fake_TN_s = interpolate(self.fake_TN, size=rand_size, mode='bilinear', align_corners=False)
            fake_D_s = interpolate(self.fake_D, size=rand_size, mode='bilinear', align_corners=False)

            if stage == 'train':
                segMask_D_s = interpolate(self.segMask_D, size=rand_size, mode='nearest')
                real_D_pred_seg = self.netS(real_D_s, from_=self.D)
                self.criterion_seg = self.update_class_criterion(segMask_D_s.long())
                self.loss_S[self.D] += self.compute_loss('seg', real_D_pred_seg,
                                                         segMask_D_s.long().squeeze(1), loss_name='S')
                self.segMask_D_update = segMask_D_s.long().detach()
                self.segMask_TN_update = self.segMask_TN

            elif stage == 'update_D':
                # Start updating D seg labels and train Thermal Seg with pseudo TIR images and Day labels
                segMask_D_s = interpolate(self.segMask_D, size=rand_size, mode='nearest').long()
                segMask_TN_s = interpolate(self.segMask_TN, size=rand_size, mode='nearest').long()
                real_D_pred_seg = self.netS(real_D_s, from_=self.D)
                real_T_pred_seg = self.netS(real_T_s if self.opt.model.fusion_first else real_TN_s, from_=self.T)
                fake_D_pred_seg_d = self.netS(fake_D_s.detach(), from_=self.D)
                fake_TN_pred_seg_d = self.netS(fake_TN_s.detach(), from_=self.T)

                self.segMask_D_update = UpdateVisGT(real_TN_s.detach(), segMask_D_s, 0.25).long()
                self.criterion_seg = self.update_class_criterion(self.segMask_D_update)
                ####
                self.loss_S[self.D] += self.compute_loss('seg', real_D_pred_seg,
                                                         self.segMask_D_update.squeeze(1), loss_name='S')
                self.loss_S[self.D] += self.compute_loss('semEdge', real_D_pred_seg,
                                                         self.segMask_D_update, loss_name='S')
                self.loss_seg[self.D] += self.compute_loss('seg', fake_TN_pred_seg_d,
                                                           self.segMask_D_update.squeeze(1))
                mask_uncertain = segMask_TN_s == 255
                self.segMask_TN_update = (UpdateIRGTv1(real_T_pred_seg.detach(), fake_D_pred_seg_d,
                                                       255 * torch.ones_like(segMask_D_s), real_T_s) *
                                          mask_uncertain + ~mask_uncertain * segMask_TN_s)

            elif stage == 'update_TN':
                segMask_D_s = interpolate(self.segMask_D, size=rand_size, mode='nearest').long()
                segMask_TN_s = interpolate(self.segMask_TN, size=rand_size, mode='nearest').long()
                real_D_pred_seg = self.netS(real_D_s, from_=self.D)
                real_T_pred_seg = self.netS(real_T_s if self.opt.model.fusion_first else real_TN_s, from_=self.T)
                fake_D_pred_seg_d = self.netS(fake_D_s.detach(), from_=self.D)
                fake_TN_pred_seg_d = self.netS(fake_TN_s.detach(), from_=self.T)
                self.segMask_D_update = UpdateVisGT(fake_TN_s.detach(), segMask_D_s, 0.25).long()
                self.criterion_seg = self.update_class_criterion(self.segMask_D_update)
                self.loss_S[self.D] += self.compute_loss('seg', real_D_pred_seg,
                                                         self.segMask_D_update.squeeze(1), loss_name='S')
                self.loss_S[self.D] += self.compute_loss('semEdge', real_D_pred_seg,
                                                         self.segMask_D_update.squeeze(1), loss_name='S')
                self.loss_seg[self.D] += self.compute_loss('seg', fake_TN_pred_seg_d,
                                                           self.segMask_D_update.squeeze(1))
                mask_uncertain = segMask_TN_s == 255
                self.segMask_TN_update = (UpdateIRGTv2(fake_TN_pred_seg_d.detach(), fake_D_pred_seg_d, segMask_TN_s,
                                                       real_T_s, prob_th=0.9).long() * mask_uncertain + ~mask_uncertain * segMask_TN_s)
                self.criterion_seg = self.update_class_criterion(self.segMask_TN_update)
                self.loss_seg[self.T] += self.compute_loss('seg', real_T_pred_seg.squeeze(1),
                                                           self.segMask_TN_update.squeeze(1))

            else:
                segMask_D_s = interpolate(self.segMask_D, size=rand_size, mode='nearest').long()
                segMask_TN_s = interpolate(self.segMask_TN, size=rand_size, mode='nearest').long()
                real_T_pred_seg = self.netS(real_T_s if self.opt.model.fusion_first else real_TN_s, from_=self.T)
                fake_D_pred_seg = self.netS(fake_D_s, from_=self.D)
                fake_TN_pred_seg = self.netS(fake_TN_s, from_=self.T)
                fake_D_pred_seg_d = self.netS(fake_D_s.detach(), from_=self.D)

                self.segMask_D_update = UpdateVisGT(fake_TN_s.detach(), segMask_D_s, 0.25).long()
                self.criterion_seg = self.update_class_criterion(self.segMask_D_update)
                self.loss_seg[self.D] += self.compute_loss('seg', fake_TN_pred_seg,
                                                           self.segMask_D_update.squeeze(1))
                mask_uncertain = segMask_TN_s == 255
                self.segMask_TN_update = (UpdateIRGTv2(real_T_pred_seg.detach(), fake_D_pred_seg_d, segMask_TN_s,
                                                       real_T_s[:,
                                                       :3]) * mask_uncertain + ~mask_uncertain * segMask_TN_s)
                segMask_TN_update_s = interpolate(self.segMask_TN_update.float(), size=rand_size, mode='nearest').long()
                self.criterion_seg = self.update_class_criterion(segMask_TN_update_s)
                self.loss_seg[self.T] = self.compute_loss('seg', fake_D_pred_seg,
                                                          segMask_TN_update_s.squeeze(1))
            return rand_size

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

    def compute_loss(self, criterion: str, *inputs, criterion_lambda: str = '', loss_name: str = '', **kwargs):
        """Compute a specific loss with given inputs and apply its lambda weight."""
        if not criterion_lambda:
            criterion_lambda = criterion
        if not loss_name:
            loss_name = criterion
        if criterion_lambda in self.lambdas:
            criterion_lambda = self.lambdas[criterion_lambda]
            if criterion_lambda == 0:
                return 0.0
            else:
                criterion = getattr(self, f'criterion_{criterion}', None)
                if criterion is None:
                    raise ValueError(f"Criterion {criterion} not found.")
                self.used_losses = loss_name
                self.sum_lambdas += criterion_lambda
                return criterion(*inputs, **kwargs) * criterion_lambda

        else:
            return 0.0

    def sum_losses(self) -> Tensor:
        sum_losses: Tensor = Tensor(
            sum(sum([v for v in self.losses[used_losses].values()]) for used_losses in self.used_losses))
        sum_lambdas = self.sum_lambdas
        self.used_losses = []
        self.sum_lambdas = -1.0  # reset
        return sum_losses / (sum_lambdas if sum_lambdas > 0 else 1.0)

    def cond(self, *args, mod='G', lambda_loss: float = None) -> bool:
        """
        Condition for training a specific domain
        """
        if (lambda_loss is not None and lambda_loss > 0) or (lambda_loss is None):
            args = list(args)
            if self.opt.training.split_optimizers is False:
                return True
            if mod == 'G':
                if 'D' in args:
                    args.remove('D')
                    args += ['E_D', 'D_D']
                if 'T' in args:
                    args.remove('T')
                    args += ['E_T', 'D_T']
                if 'N' in args:
                    args.remove('N')
                    args += ['E_N', 'D_N']
                corresp = {'E_D': 0, 'E_T': 1, 'E_N': 2, 'D_D': 3, 'D_T': 4, 'D_N': 5, 'Fus': 6}
            else:
                corresp = {'D': 0, 'T': 1, 'N': 2}
            assert all([arg in corresp.keys() for arg in args]), \
                "args: must be of form 'EA', 'EB', 'EC', 'DA', 'DB', 'DC','Fus' "
            return any([corresp[d] in self.partial_train_net[mod] for d in args])

    # endregion

    # region ------------------------ Training Helpers ----------------------- #
    def visualize_current_results(self, save=False):
        visuals = {'real_D': (self.real_D * 0.5 + 0.5 if self.real_D is not None else None),
                   'real_T': (self.real_T * 0.5 + 0.5 if self.real_T is not None else None),
                   'real_N': (self.real_N * 0.5 + 0.5 if self.real_N is not None else None),
                   'fake_D': (self.fake_D * 0.5 + 0.5 if self.fake_D is not None else None),
                   'fake_T': (self.fake_T * 0.5 + 0.5 if self.fake_T is not None else None),
                   'fake_N': (self.fake_N * 0.5 + 0.5 if self.fake_N is not None else self.real_TN * 0.5 + 0.5),
                   'rec_D': (self.rec_D * 0.5 + 0.5 if self.rec_D is not None else None),
                   'rec_T': (self.rec_T * 0.5 + 0.5 if self.rec_T is not None else None),
                   'rec_N': (self.rec_N * 0.5 + 0.5 if self.rec_N is not None else None)}
        out = {lab: ImageTensor(im[0]) for lab, im in visuals.items() if im is not None}
        self.visualizer.display_current_results(out)
        if save:
            self.visualizer.save_current_results(out, self.epoch)

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
