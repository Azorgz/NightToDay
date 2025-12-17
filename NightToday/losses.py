"""
losses.py
---------
Collection of loss functions used in Generative Adversarial Transformers.
Includes GAN variants (LSGAN, Hinge, WGAN-GP, RaGAN) and auxiliary losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ImagesCameras import ImageTensor
from kornia.color import rgb_to_lab
from kornia.contrib import connected_components
from kornia.filters import sobel, laplacian, get_gaussian_kernel2d
from kornia.morphology import erosion, closing, opening, dilation
from torch import Tensor
from torch.nn import LeakyReLU, ReLU
from torchmetrics.functional.image import image_gradients
from torchvision.transforms.v2 import Normalize
from NightToday.ssim import SSIM
from NightToday.utilities import RefineLightMask, ClsMeanPixelValue, GetFeaMatrixCenter, bhw_to_onehot, \
    detect_hotpoint_blob, center_of_mass

ROAD = 0
PAVEMENT = 1
BUILDING = 2
TRAFFICLIGHT = 6
SIGN = 7
VEG = 8
SKY = 10
PERSON = 11
STREETLIGHT = 12
CAR = 13
TRUCK = 14
BUS = 15
TRAIN = 16
MOTORCYCLE = 17
BICYCLE = 18
VEHICLES = [CAR, TRUCK, BUS, TRAIN, MOTORCYCLE, BICYCLE]


class Loss(nn.Module):
    def __init__(self, criterion, name: str):
        super(Loss, self).__init__()
        self.storage = {}
        self.criterion = criterion
        self.name = name

    def initialize_storage(self, keys: list):
        for key in keys:
            self.storage[key] = 0.0

    def update_storage(self, key, value):
        if key in self.storage:
            self.storage[key] += value
        else:
            raise KeyError(f"Key {key} not found in storage.")

    def reset_storage(self):
        for key in self.storage.keys():
            self.storage[key] = 0.0

    def forward(self, *args, key=None, **kwargs):
        assert key in self.storage
        self.update_storage(key, self.criterion(*args, **kwargs))

    def __repr__(self):
        ret = f"Loss_{self.name} ="
        for key, value in self.storage.items():
            ret += f" {key}: {value:.4f};"
        return ret


# =========================
# --- GAN LOSSES ---------
# =========================


class GANLoss(nn.Module):
    """
    Configurable GAN loss wrapper.
    Supported types: ['lsgan', 'hinge', 'wgan-gp', 'ralsgan']
    """

    def __init__(self, gan_type="wgan-gp", gp_weight=1.0, device: torch.device = "cuda"):
        super().__init__()
        self.gan_type = gan_type.lower()
        match self.gan_type:
            case "lsgan":
                self.loss = lambda g_p, p_real, p_fake, f_d: self.lsgan_loss(p_real, p_fake, f_d)
            case "hinge":
                self.loss = lambda g_p, p_real, p_fake, f_d: self.hinge_loss(p_fake, f_d)
            case "wgan-gp":
                self.loss = lambda g_p, p_real, p_fake, f_d: self.wgangp_loss(g_p, p_real, p_fake, f_d)
            case "ralsgan":
                self.loss = lambda g_p, p_real, p_fake, f_d: self.ralsgan_loss(p_real, p_fake, f_d)
        self.gp_weight = gp_weight
        self.device = device
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, D, real, pred_real, fake, for_discriminator=True):
        """
        Computes loss for discriminator or generator.
        """
        if for_discriminator:
            fake = fake.detach()
        else:
            pred_real = tuple([p.detach() for p in p_real] for p_real in pred_real)
        pred_fake = D(fake)
        if self.gan_type == "wgan-gp" and for_discriminator:
            gradient_penalty = self.gradient_penalty(D, real, fake)
        else:
            gradient_penalty = [None] * len(pred_real)
        res = 0.
        for i in range(3):
            losses = self.loss(gradient_penalty, pred_real[i], pred_fake[i], for_discriminator)
            multipliers = list(range(1, len(pred_fake[i]) + 1))
            multipliers[-1] += 1
            res += sum([m * l for m, l in zip(multipliers, losses)]) / sum(multipliers)
        return res / 3

    def lsgan_loss(self, pred_real, pred_fake, for_discriminator) -> list:
        res = []
        for p_fake, p_real in zip(pred_fake, pred_real):
            if for_discriminator:
                real_loss = self.mse(p_real, torch.ones_like(p_real))
                fake_loss = self.mse(p_fake, torch.zeros_like(p_fake))
                res.append(0.5 * (real_loss + fake_loss))
            else:
                res.append(0.5 * self.mse(p_fake, torch.ones_like(p_fake)))
        return res

    def hinge_loss(self, pred_fake, for_discriminator) -> list:
        res = []
        for p_fake in pred_fake:
            if for_discriminator:
                real_loss = torch.mean(F.relu(1.0 - p_fake))
                fake_loss = torch.mean(F.relu(1.0 + p_fake))
                res.append(real_loss + fake_loss)
            else:
                res.append(-torch.mean(p_fake))
        return res

    def ralsgan_loss(self, pred_real, pred_fake, for_discriminator) -> list:
        res = []
        for p_fake, p_real in zip(pred_fake, pred_real):
            if for_discriminator:
                loss = self.mse(p_real - p_fake.mean(), torch.ones_like(p_real)) + \
                       self.mse(p_fake - p_real.mean(), torch.zeros_like(p_fake)) / 2
                res.append(loss)
            else:
                loss = (self.mse(p_real - p_fake.mean(), torch.zeros_like(p_real)) +
                        self.mse(p_fake - p_real.mean(), torch.ones_like(p_fake))) / 2
                res.append(loss)
        return res

    def wgangp_loss(self, gradient_penalty, pred_real, pred_fake, for_discriminator) -> list:
        res = []
        for gp, p_fake, p_real in zip(gradient_penalty, pred_fake, pred_real):
            d_real = torch.mean(p_real)
            d_fake = torch.mean(p_fake)
            if for_discriminator:
                loss_D = -(d_real - d_fake) + self.gp_weight * gp
                res.append(loss_D)
            else:
                res.append(-d_fake)
        return res

    # ----- Gradient Penalty for WGAN-GP -----
    def gradient_penalty(self, D, real, fake):
        batch_size = real.size(0)
        eps = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d_interpolated = D(interpolated)
        gp = []
        for d_i in d_interpolated:
            sub = 0.
            for d in d_i:
                grad_outputs = torch.ones_like(d)
                gradients = torch.autograd.grad(
                    outputs=d, inputs=interpolated,
                    grad_outputs=grad_outputs, create_graph=True,
                    retain_graph=True, only_inputs=True
                )[0]
                gradients = gradients.view(batch_size, -1)
                sub += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            gp.append(sub)
        return gp


# class MILO_Loss(nn.Module):
#     def __init__(self, device: torch.device = "cuda"):
#         super(MILO_Loss, self).__init__()
#         self.model = MILO().to(device)
#         self.model.eval()
#
#     def train(self, mode: bool = False):
#         """Override the default train() to make sure the model is always in eval mode."""
#         return super().train(False)
#
#     def forward(self, fake, real, mask=None):
#         """
#         Computes MILO loss between real visible and fake infrared images.
#         Args:
#             real: (B,3,H,W) real images
#             fake:  (B,3,H,W) generated images
#         """
#         real = (real + 1.0) * 0.5
#         fake = (fake + 1.0) * 0.5
#
#         if mask is not None:
#             mask_resized = F.interpolate(mask.float(), size=real.shape[-2:], mode='bilinear', align_corners=False)
#         else:
#             mask_resized = torch.ones_like(real[:, :1, :, :]).to(real.device)
#         return self.score(real, fake, mask_resized).squeeze()
#
#     def score(self, real, fake, mask):
#         """
#         Computes MILO map between real visible and fake infrared images.
#         Args:
#             real: (B,3,H,W) real images
#             fake:  (B,3,H,W) generated images
#             mask: (B,1,H,W) optional mask
#         """
#         score = self.model.MOS_score(real, fake, mask)
#         return score
#
#     def map(self, real, fake):
#         MILO_err, MILO_mask = self.model.MILO_map(fake, real)
#         MILO_err = ImageTensor(map_visualization(MILO_err))
#         return MILO_err, MILO_mask


def ColorLoss(image_fake, image_target, GT_seg=None, th_high=0.95, th_low=0.15, weights=None):
    """

    """
    # Normalize back to [0,1]
    assert image_fake.shape == image_target.shape, "Input images must have the same dimensions."
    im_fake = ImageTensor(image_fake * 0.5 + 0.5)
    im_target = ImageTensor(image_target * 0.5 + 0.5)
    B = im_fake.shape[0]
    # color distance function
    color_dist = im_fake.color_distance(im_target)  # shape (B,1,H,W) or (B,H,W)
    color_magn = im_target.LAB()[:, 1:3, :, :].norm(p=2, dim=1, keepdim=True)  # AB channels magnitude
    target_l, target_ab = rgb_to_lab(im_target).split([1, 2], 1)  # L channel
    low_lum = target_l < th_low
    high_lum = target_l > th_high
    valid = 1 - (low_lum * 1. + high_lum * 1.)
    color_dist = color_dist * valid
    ssim_loss = SSIM_Loss(channel=3)(im_fake, im_target, full=True).mean(dim=1, keepdim=True) * (1 - low_lum * 1.)
    weights = weights[[CAR, TRUCK, BUILDING, MOTORCYCLE, SIGN, ROAD]].to(im_target.device) if weights is not None else \
        torch.ones(6, device=image_fake.device)
    H, W = im_fake.shape[-2:]
    loss = torch.zeros([B, ], device=im_target.device)

    if GT_seg is not None:
        if color_dist.shape[-2:] != GT_seg.shape[-2:]:
            GT_seg = F.interpolate(GT_seg.float(), size=(H, W), mode='nearest').long()

        # build masks in a single vectorized call
        # ------------------------------------------------------------
        # Class masks (exact value matches)
        classes_eq = torch.tensor([CAR, TRUCK, BUILDING, MOTORCYCLE, SIGN], device=GT_seg.device).view(1, -1, 1,
                                                                                                       1)  # shape (1,5,1,1)

        # Expand GT_seg to compare against all 5 classes at once
        mask_eq = (GT_seg == classes_eq).float()  # (B,5,H,W)

        # road/pavement/building mask (GT_seg <= 2)
        mask_build = (GT_seg <= BUILDING).float()  # (B,1,H,W)

        # concatenate → shape (B,6,H,W)
        masks = torch.cat([mask_eq, mask_build], dim=1)

        # Compute sum of each mask
        sum_masks = masks.sum(dim=(2, 3))  # (B,6)

        # Compute masked losses for each class in parallel:
        # (color_dist: B,1,H,W → broadcast to B,6,H,W)
        masked_loss = (color_dist * masks * color_magn/(color_magn.max()/2)).sum(dim=(2, 3))  # (B,6)
        masked_ssim_loss = (ssim_loss * masks[:, [2, 4, 5]]).sum(dim=(2, 3))  # (B,3)

        # Avoid division by zero: zero out empty masks
        per_class_loss = masked_loss / (sum_masks + 1e-6)  # (B,6)
        per_class_ssim_loss = masked_ssim_loss / (sum_masks[:, [2, 4, 5]] + 1e-6)  # (B,3)

        # Original code uses: losses = [0., L1, L2, ...]
        # Equivalent to summing only valid classes:
        valid_mask = (sum_masks > 0).float()
        sum_losses = (((per_class_loss * valid_mask * weights).sum(dim=1) / weights.mean() +
                       (per_class_ssim_loss * valid_mask[:, [2, 4, 5]] * weights[[2, 4, 5]]).sum(dim=1))
                      / weights[[2, 4, 5]].mean())  # (B,)

        # average batch
        loss += sum_losses
    else:
        color_mask = target_l.clamp(0, 25) * ((target_ab / 128) ** 2).sum(1, keepdim=True)
        high_color_mask = (color_mask > color_mask.mean() + 2 * color_mask.std()) * valid * (GT_seg != SKY)
        loss = (color_dist * high_color_mask).sum(dim=[1, 2, 3]) / (high_color_mask.sum(dim=[1, 2, 3]) + 1e-6)
    return loss.mean() / 20


def ThermalLoss(image_fused, image_target, night_color, GT_seg, weights=None):
    """
    Thermal Loss to enhance the thermal characteristics of the fused image.
    The texture gradient will be maximized in vegetation and vehicles regions
    :param image_fused:
    :param image_target:
    :param GT_seg:
    :param weights:
    :return: loss value
    """
    # night_color = (1.0 - night_color)
    B = image_target.shape[0]
    GT_resized = F.interpolate(GT_seg.float(), size=image_fused.shape[-2:], mode='nearest').long().detach()
    weights = weights[[SKY, VEG, PERSON, CAR]] if weights is not None else (
        torch.tensor([1., 1., 1., 1.], device=image_fused.device))
    sky_mask = (GT_resized == SKY).float()
    area_sky = sky_mask.sum(dim=[1, 2, 3])
    valid_sky = area_sky > 400
    veg_mask = (GT_resized == VEG).float()
    area_veg = veg_mask.sum(dim=[1, 2, 3])
    valid_veg = area_veg > 400
    person_mask = erosion((GT_resized == PERSON).float(), torch.ones(3, 3, device=image_fused.device))
    area_person = person_mask.sum(dim=[1, 2, 3])
    valid_person = area_person > 30
    car_mask = sum([GT_resized == V for V in VEHICLES]).float()
    area_car = car_mask.sum(dim=[1, 2, 3])
    valid_car = area_car > 50
    thermal_diff_low = ReLU()(image_fused - image_target + 0.1)  # only penalize higher values
    thermal_diff_high = ReLU()(image_target - image_fused)  # only penalize lower values

    # losses init
    sky_loss = torch.zeros([B, ], device=image_target.device)
    veg_loss = torch.zeros([B, ], device=image_target.device)
    person_loss = torch.zeros([B, ], device=image_target.device)
    car_loss = torch.zeros([B, ], device=image_target.device)

    #  Thermal correction losses
    sky_loss[valid_sky] += (thermal_diff_low[valid_sky] * sky_mask[valid_sky]).sum(dim=[1, 2, 3]) / area_sky[valid_sky]
    veg_loss[valid_veg] += (thermal_diff_high[valid_veg] * veg_mask[valid_veg]).sum(dim=[1, 2, 3]) / area_veg[valid_veg]
    veg_loss[valid_sky*valid_veg] += ReLU()(veg_mask[valid_sky*valid_veg].sum(dim=[1, 2, 3]) / area_veg[valid_sky*valid_veg] *0.8 -
                                sky_mask[valid_sky*valid_veg].sum(dim=[1, 2, 3]) / area_sky[valid_sky*valid_veg])

    person_loss[valid_person] += (thermal_diff_high[valid_person] * person_mask[valid_person]).sum(dim=[1, 2, 3]) / \
                                 area_person[valid_person]
    # trafficlight_loss = TL_loss(image_target, night_color, image_fused, GT_resized).sum()

    #  Sharpness enhancement losses
    dx, dy = image_gradients(
        torch.stack([image_fused.mean(1), image_target.mean(1), night_color.mean(1)], dim=0))  # 3, B, H, W
    dx2, dy2 = image_gradients(torch.sqrt(dx ** 2 + dy ** 2 + 1e-6))
    grad_imgs = torch.sqrt(dx2 ** 2 + dy2 ** 2 + 1e-6)
    grad_fused, grad_target, grad_night = grad_imgs.permute(1, 0, 2, 3).split(1, dim=1)
    grad_loss = LeakyReLU(1e-4)(grad_target - grad_fused) + LeakyReLU(1e-4)(grad_fused - grad_night)
    veg_loss[valid_veg] += (grad_loss[valid_veg] * veg_mask[valid_veg]).sum(dim=[1, 2, 3]) / area_veg[valid_veg]
    car_loss[valid_car] += (grad_loss[valid_car] * car_mask[valid_car]).sum(dim=[1, 2, 3]) / area_car[valid_car]

    total_classes_loss = ((sky_loss * weights[0] + veg_loss * weights[1] + person_loss * weights[2] + car_loss * weights[3]) /
            (weights * torch.stack([valid_sky, valid_veg, valid_person, valid_car], dim=-1).float()).sum(1)).mean()

    thermal_noise_loss = ThermalNoiseLoss()(image_fused).mean() * 2

    return total_classes_loss + thermal_noise_loss #  + trafficlight_loss


def TL_loss(I_ir, I_vi, I_fused, GT_mask):
    loss_tl = torch.zeros([I_ir.shape[0], ], device=I_ir.device)
    mask_trafficlight = (GT_mask == TRAFFICLIGHT).float()
    area_trafficlight = mask_trafficlight.sum(dim=[1, 2, 3])
    valid_trafficlight = area_trafficlight > 50
    if not valid_trafficlight.any():
        return loss_tl
    labels = connected_components(mask_trafficlight)
    fake_TL = torch.zeros_like(GT_mask, device=GT_mask.device).float()
    for i, val in enumerate(valid_trafficlight):
        if val:
            ir = (I_ir[i]*0.5+0.5).mean(0)
            unique = labels[i].unique(return_counts=True)
            for label, count in zip(unique[0], unique[1]):
                if count > 10000 or count < 50:
                    continue
                else:
                    size_scaler = (count // 50)
                    mask_tl = (labels[i][None] == label).float()
                    mask_dilated = dilation(mask_tl, torch.ones(5*size_scaler, 3*size_scaler, device=GT_mask.device))
                    contours = (mask_dilated - mask_tl).squeeze()
                    fake_TL[i] += create_fake_TL(I_ir[i][None], I_vi[i][None], mask_dilated) * mask_dilated[0]
                    mask_tl = mask_tl.squeeze()
                    if (contours * I_ir[i]).mean() > (mask_tl * I_ir[i]).mean():
                        fake_TL[i] += 1.05 * contours * ir + mask_tl*ir*0.95
                    else:
                        fake_TL[i] += 0.95 * contours * ir + mask_tl*ir*1.05

    # --- Traffic Light reconstitution ---
    return ReLU()(((fake_TL > 0) * (I_fused*0.5+0.5).mean(dim=1, keepdim=True) - fake_TL.clamp(0, 1)) ** 2).mean(dim=[1, 2, 3]).sum()


def create_fake_TL(I_ir, I_vis, mask):
    # extract shape as mean sum(dim[mean +-std())
    I_ir = (I_ir * 0.5 + 0.5).mean(dim=1).squeeze()
    mask = mask.squeeze()
    real_IR_TLight = I_ir * mask
    cx, cy = center_of_mass(mask[None, None])
    h0, w0 = mask.sum(-2), mask.sum(-1)
    h_mean, h_std, w_mean, w_std = h0[h0 > 0].mean(), h0[h0 > 0].std(), w0[w0 > 0].mean(), w0[w0 > 0].std()
    h = int(h0[(h0 <= (h_mean + h_std)) * (h0 >= (h_mean - h_std))].mean())
    w = int(w0[(w0 <= (w_mean + w_std)) * (w0 >= (w_mean - w_std))].mean())
    if w0[0] > 0:
        cut = True
        pad = [0, 0, max(w * 3 - h, 0), 0]  # left, right, top, bottom
        h = w * 3
    else:
        cut = False
        pad = [0, 0, 0, 0]
    #  vertical borders
    dx = (torch.abs(I_ir[:, 1:] - I_ir[:, :-1])*mask[:, 1:]).sum(-2)
    left_border = torch.argmax(dx[: max(cx.long()-w//4, 0)], dim=-1) or 0
    right_border = torch.argmax(dx[max(cx.long()+w//4, dx.shape[-1]-1):], dim=-1) or dx.shape[-1]-1





    # real IR values in the TL area
    TL_ir_target = I_ir[mask > 0].mean() + I_ir[mask > 0].std()

    # switch different case shape of TLights:
    if (w / h) < 0.7:
        # 3 vertical light
        base = torch.ones((1, 1, h, h // 3), device=I_ir.device) * TL_ir_target
        gaussian_kernel = get_gaussian_kernel2d(((h//6)*2 + 1, (h//6)*2 + 1), (h / 6, h / 6)).to(I_ir.device)
        color = determine_color(I_vis * mask)
        positions = [h // 6, h // 2, h * 5 // 6], [h // 6, h // 6, h // 6]
        w = h // 3
        lights = torch.zeros((h, w), device=I_ir.device)
        for i, (pos_y, pos_x) in enumerate(zip(*positions)):
            if color == 'red' and i == 0:
                lights[pos_y, pos_x] = -1.0
            elif color == 'green' and i == 2:
                lights[pos_y, pos_x] = -1.0
            elif color == 'yellow' and i == 1:
                lights[pos_y, pos_x] = -1.0
            else:
                lights[pos_y, pos_x] = 1.0

    elif (w / h) < 0.9:
        # 2 vertical light
        base = torch.ones((1, 1, h, h // 2), device=I_ir.device) * TL_ir_target
        gaussian_kernel = get_gaussian_kernel2d(((h//6)*2+1, (h // 6)*2+1), (h / 6, h / 6)).to(I_ir.device)
        color = determine_color(I_vis * mask)
        positions = [h // 3, h * 2 // 3], [h // 4, h // 4, h // 4]
        w = h // 2
        lights = torch.zeros((h, w), device=I_ir.device)
        for i, (pos_y, pos_x) in enumerate(zip(*positions)):
            if color == 'red' and i == 0:
                lights[pos_y, pos_x] = -1.0
            elif color == 'green' and i == 2:
                lights[pos_y, pos_x] = -1.0
            else:
                lights[pos_y, pos_x] = 1.0
    else:
        # square light
        return torch.zeros_like(I_ir, device=I_ir.device).float()

    lights = F.conv2d(lights[None, None], gaussian_kernel[None], padding=gaussian_kernel.shape[-1] // 2)
    lights_normed = (lights - lights.min()) / (lights.max() - lights.min()) * torch.abs(TL_ir_target) - torch.abs(TL_ir_target / 2)
    fake_TL = (base + lights_normed)
    # place back to original image size
    fake_TL_full = torch.zeros_like(I_ir[None, None], device=I_ir.device).float()
    if cut:
        fake_TL_full = F.pad(fake_TL_full, pad, "constant", 0)
    x_c, y_c = center_of_mass(F.pad(mask[None, None], pad))
    y1, y2 = int(max(0, y_c - h // 2)), int(min(I_ir.shape[0], y_c + (h+1) // 2))
    x1, x2 = int(max(0, x_c - w // 2)), int(min(I_ir.shape[1], x_c + (w+1) // 2))
    fake_TL_full[:, :, y1:y2, x1:x2] = fake_TL
    # unpad if cut
    if cut:
        fake_TL_full = fake_TL_full[:, :, pad[2]:, :]
    return fake_TL_full[0]


def determine_color(TL_vis):
    R = TL_vis[:, 0].mean()
    G = TL_vis[:, 1].mean()
    B = TL_vis[:, 2].mean()
    if R > G and R > B:
        return 'red'
    elif G > R and G >= B:
        return 'green'
    else:
        return 'yellow'


class SharpFusionLoss(torch.nn.Module):
    def __init__(self, lam_grad=1.0, lam_lap=0.7, lam_contrast=0.5, lam_freq=0.3):
        super().__init__()
        self.lam_grad = lam_grad
        self.lam_lap = lam_lap
        self.lam_contrast = lam_contrast
        self.lam_freq = lam_freq

        self.gauss = get_gaussian_kernel2d((7, 7), (3, 3)).to('cuda')

    def local_std(self, x):
        # x: Bx1xHxW grayscale
        x = x.mean(1, keepdim=True)
        mean = F.conv2d(x, self.gauss[None], padding=3)
        mean2 = F.conv2d(x * x, self.gauss[None], padding=3)
        std = torch.sqrt((mean2 - mean * mean)**2 + 1e-6)
        return std

    def to(self, device):
        self.gauss = self.gauss.to(device)

    def fft_high(self, x, cutoff=0.25):
        # x: Bx1xHxW
        B, _, H, W = x.shape
        X = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))

        # Create high-pass frequency mask
        u = torch.linspace(-1, 1, H, device=x.device)
        v = torch.linspace(-1, 1, W, device=x.device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        R = torch.sqrt(U * U + V * V)

        mask = (R > cutoff).float()  # High frequencies only
        mask = mask[None, None]  # Bx1xHxW

        return X * mask

    def forward(self, I_f, I_vi, I_ir):
        # Convert to grayscale if needed
        if I_f.shape[1] == 3:
            I_f = 0.299 * I_f[:, 0:1] + 0.587 * I_f[:, 1:2] + 0.114 * I_f[:, 2:3]
            I_vi = 0.299 * I_vi[:, 0:1] + 0.587 * I_vi[:, 1:2] + 0.114 * I_vi[:, 2:3]

        # -------- Gradient Loss --------
        G_f = sobel(I_f).abs()
        G_vi = sobel(I_vi).abs()
        G_ir = sobel(I_ir).abs()
        G_ref = torch.max(G_vi, G_ir)
        L_grad = F.l1_loss(G_f, G_ref)

        # -------- Laplacian Loss --------
        L_f = laplacian(I_f, 3).abs()
        L_vi = laplacian(I_vi, 3).abs()
        L_ir = laplacian(I_ir, 3).abs()
        L_ref = torch.max(L_vi, L_ir)
        L_lap = F.l1_loss(L_f, L_ref)

        # -------- Frequency Loss --------
        F_f = self.fft_high(I_f)
        F_vi = self.fft_high(I_vi)
        F_ir = self.fft_high(I_ir)
        F_ref = torch.max(torch.abs(F_vi), torch.abs(F_ir))
        L_freq = F.l1_loss(torch.abs(F_f), F_ref)

        # -------- Local Contrast Loss --------
        # I_vi = erosion(I_vi, torch.ones(3, 3, device=I_vi.device))
        C_f = self.local_std(I_f)
        # C_vi = self.local_std(I_vi) * (I_ir > I_ir.mean(dim=[1, 2, 3], keepdim=True))
        # C_ir = self.local_std(I_ir)
        # C_ref = torch.max(C_vi, C_ir)
        C_ref = self.local_std(I_ir)
        L_contrast = F.l1_loss(C_f, C_ref)
        # ---- Total ----
        L = (self.lam_grad * L_grad +
             self.lam_lap * L_lap +
             self.lam_contrast * L_contrast +
             self.lam_freq * L_freq)

        return L.mean()


class ThermalNoiseLoss(nn.Module):

    def __init__(self):
        super(ThermalNoiseLoss, self).__init__()
        self.scales = (2, 4)
        self.alpha = 1.8
        self.eps = 1e-3
        self.w_ms = 1.0
        self.w_tensor = 0.7
        self.w_tv = 0.4
        self.w_freq = 0.6

    # -------- Multiscale coherence --------
    def multiscale_loss(self, x):
        loss = 0
        for s in self.scales:
            ds = F.avg_pool2d(x, s)
            us = F.interpolate(ds, size=x.shape[-2:], mode='bilinear', align_corners=False)
            loss += torch.abs(x - us).mean()
        return loss / len(self.scales)

    # -------- Structure tensor loss --------
    def structure_tensor_loss(self, x):
        gx, gy = sobel(x).chunk(2, dim=1)
        Jxx = gx * gx
        Jyy = gy * gy
        Jxy = gx * gy

        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        lambda2 = trace / 2 - torch.sqrt((trace / 2) ** 2 - det + 1e-6)
        lambda1 = trace - lambda2

        return (lambda2 / (lambda1 + 1e-6)).mean()

    # -------- Charbonnier TV --------
    def charbonnier_tv(self, x):
        dx, dy = image_gradients(x)
        return torch.mean(torch.sqrt(dx**2 + dy**2 + self.eps**2))

    # -------- Frequency decay --------
    def frequency_decay_loss(self, x):
        B, C, H, W = x.shape
        X = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
        mag = torch.abs(X)

        u = torch.linspace(-1, 1, H, device=x.device)
        v = torch.linspace(-1, 1, W, device=x.device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        R = torch.sqrt(U * U + V * V)

        return (mag * (R ** self.alpha)).mean()

    # -------- Final denoising loss --------
    def forward(self, I_fused):
        L = (self.w_ms * self.multiscale_loss(I_fused) +
             self.w_tensor * self.structure_tensor_loss(I_fused) +
             self.w_tv * self.charbonnier_tv(I_fused) +
             self.w_freq * self.frequency_decay_loss(I_fused))
        return L


class SSIM_Loss(SSIM):

    def __init__(self, channel=3):
        super(SSIM_Loss, self).__init__(data_range=1.0, size_average=True, channel=channel)

    def forward(self, img1, img2, full=False):
        assert img1.size() == img2.size(), "Input images must have the same dimensions."
        if not full:
            ssim = lambda x, y: super(SSIM_Loss, self).forward(x, y)
        else:
            ssim = lambda x, y: super(SSIM_Loss, self).forward(x, y, return_map=True)
        if img1.shape[1] == 6:
            img1_1 = img1[:, :3, :, :]
            img2_1 = img2[:, :3, :, :]
            img1_2 = img1[:, 3:, :, :]
            img2_2 = img2[:, 3:, :, :]
            return 1 - (ssim(img1_1, img2_1) + ssim(img1_2, img2_2)) * 0.5
        return 1 - ssim(img1, img2)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size, _, h_x, w_x = x.shape
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.shape[1] * t.shape[2] * t.shape[3]


def SemEdgeLoss(seg_tensor, GT_mask, num_classes):
    """
    Batched semantic edge loss.
    Encourages semantic edge prediction to match GT edges.
    Args:
        seg_tensor: (B,C,H,W) logits
        GT_mask: (B,H,W) ground truth
        num_classes: including uncertain class
    Returns:
        scalar loss
    """
    device = seg_tensor.device
    B = seg_tensor.size(0)

    # Softmax prediction and one-hot GT
    pred_sm = F.softmax(seg_tensor, dim=1)  # (B,C,H,W)
    GT_onehot = bhw_to_onehot(GT_mask, num_classes)  # (B,C,H,W)

    # Edge detection via local averaging
    avg_pool = lambda x: F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    pred_semedge = torch.abs(pred_sm - avg_pool(pred_sm))
    GT_semedge = torch.abs(GT_onehot - avg_pool(GT_onehot))

    # Compute per-sample normalized edge loss
    edge_sum_per_sample = GT_semedge.contiguous().view(B, -1).sum(dim=1)  # (B,)
    # avoid division by zero
    nonzero_mask = edge_sum_per_sample > 0
    loss_per_sample = torch.zeros(B, device=device)

    if nonzero_mask.any():
        loss_per_sample[nonzero_mask] = (
                torch.sum(torch.abs(GT_semedge.detach()[nonzero_mask] - pred_semedge[nonzero_mask]),
                          dim=(1, 2, 3)) / edge_sum_per_sample[nonzero_mask])
    # Average across batch
    loss = loss_per_sample.mean()
    return loss


class StructuralGradientLoss(nn.Module):
    def __init__(self, sqrt_patch_num=4, gradient_th=0.2):
        super(StructuralGradientLoss, self).__init__()
        self.sqrt_patch_num = sqrt_patch_num
        self.gradient_th = gradient_th
        self.AAP_module = nn.AdaptiveAvgPool2d(self.sqrt_patch_num)

    def forward(self, real_edgemap, fake_gradmap):
        """SGA Loss. The ratio of the gradient at the edge location to the maximum gradient in
        its neighborhood is encouraged to be greater than a given threshold."""
        b, c, h, w = fake_gradmap.shape
        pooled_image = self.AAP_module(real_edgemap).squeeze(1)  # shape: (b, sqrt_patch_num, sqrt_patch_num)
        patch_size = h // self.sqrt_patch_num
        losses = torch.tensor([0.], device=real_edgemap.device)
        for i in range(b):
            if pooled_image[i].sum() != 0:
                nonzeros = torch.nonzero(pooled_image[i])
                patch_idx_x, patch_idx_y = nonzeros[torch.randint(low=0, high=nonzeros.shape[0], size=(1,))].split(1, 1)
                pos_list = list(range(0, h, patch_size))
                pos_h = pos_list[patch_idx_x]
                pos_w = pos_list[patch_idx_y]
                rand_edgemap_patch = real_edgemap[:, :, pos_h:(pos_h + patch_size), pos_w:(pos_w + patch_size)]
                rand_gradmap_patch = fake_gradmap[:, :, pos_h:(pos_h + patch_size), pos_w:(pos_w + patch_size)]
                sum_edge_pixels = torch.sum(rand_edgemap_patch) + 1
                fake_grad_norm = rand_gradmap_patch / torch.max(rand_gradmap_patch)
                losses += (torch.sum(F.relu(self.gradient_th * rand_edgemap_patch - fake_grad_norm))) / sum_edge_pixels
            else:
                losses += 0.
        return losses


def FakeIRPersonLoss(Seg_mask: torch.Tensor, fake_IR: torch.Tensor) -> torch.Tensor:
    """
    Temperature regularization for pedestrian region:
    Encourages the min value over the person region in fake_IR to be larger than
    the mean value of the road region.

    Inputs:
      Seg_mask: Long/Float tensor of shape (B, 1, seg_h, seg_w).
      fake_IR:  Tensor of shape (B, C, H, W) (assumes values in [-1, 1])

    Output:
      per-sample loss tensor of shape (B,)
    """
    B, C, H, W = fake_IR.shape
    GT_mask_resize = F.interpolate(Seg_mask.float(), size=[H, W], mode='nearest').long()
    # person mask (B,1,H,W)
    person_mask = erosion((GT_mask_resize == 11).float(), torch.ones(3, 3, device=GT_mask_resize.device))

    fake_img_norm = (fake_IR + 1.0) * 0.5
    fake_mean_fea, fake_cls_tensor, _ = ClsMeanPixelValue(fake_img_norm, Seg_mask.detach(), 19)
    presence_mask = (fake_cls_tensor[:, 11, 0] *
                     fake_cls_tensor[:, 0, 0]) > 0
    presence_mask = presence_mask.float()  # (B,)

    # Expand person mask to match channels
    person_mask_c = person_mask.expand(-1, C, -1, -1)  # (B,C,H,W)
    non_person_mask_c = 1 - person_mask_c

    # Padded person region
    person_region = person_mask_c * fake_img_norm
    padded_person = person_region + non_person_mask_c  # non-person → 1

    # Per-sample minimum
    person_min = padded_person.view(B, C, -1).min(dim=2).values.min(dim=1).values  # (B,)

    # Road mean feature (B,C) → reduce over C
    road_mean = fake_mean_fea[:, 0, :].mean(dim=1)  # (B,)

    loss_raw = F.relu(road_mean - person_min) / (road_mean + 1e-6)
    loss = loss_raw * presence_mask

    return loss.sum()


def BiasCorrLoss(Seg_mask, fake, real_vis, rec_vis, real_edges, fake_gradmap):
    """
    Bias correction loss including artifact and color bias correction.

    Args:
        Seg_mask:   (B,1,H_seg,W_seg) segmentation GT mask
        fake_IR:    (B,3,H,W) fake IR image
        real_vis:   (B,3,H,W) real visible image
        rec_vis:    (B,3,H,W) reconstructed visible image
        real_edges: (B,1,H,W) edge map from real visible
        fake_gradmap: (B,1,H,W) gradient map from fake IR
    Returns:
        total_loss: scalar
    """
    fake_IR, = fake.split(3, dim=1)
    # fake_IR, fake_N = fake.split(3, dim=1)
    device = fake_IR.device
    B, _, H, W = fake_IR.shape
    GT_mask = F.interpolate(Seg_mask.float(), size=(H, W), mode='nearest').detach()

    # Masks
    light_mask_ori = (GT_mask == TRAFFICLIGHT).float()
    veg_mask = (GT_mask == VEG).float()
    sign_mask = (GT_mask == SIGN).float()
    road_mask = (GT_mask == ROAD).float()
    SLight_mask_ori = (GT_mask == STREETLIGHT).float()
    motorcycle_mask = (GT_mask == MOTORCYCLE).float()
    TLight_mask = (GT_mask == TRAFFICLIGHT).float()

    # Normalize images
    fake_ir_norm = (fake_IR + 1.0) * 0.5
    # fake_ni_norm = (fake_N + 1.0) * 0.5
    real_vis_norm = (real_vis + 1.0) * 0.5

    # Grayscale
    real_gray = 0.299 * real_vis_norm[:, 0:1, :, :] + 0.587 * real_vis_norm[:, 1:2, :, :] + 0.114 * real_vis_norm[:,
                                                                                                    2:3, :, :]
    fake_ir_gray = 0.299 * fake_ir_norm[:, 0:1, :, :] + 0.587 * fake_ir_norm[:, 1:2, :, :] + 0.114 * fake_ir_norm[:,
                                                                                                     2:3, :, :]
    real_lab = rgb_to_lab(real_vis_norm)
    real_lab[:, :1, :, :] = real_lab[:, :1, :, :] / 100 + 1e-6  # [1e-6 1]
    real_lab[:, 1:, :, :] = real_lab[:, 1:, :, :] / 128  # [-1 1]

    ########### Artifact Bias Correction
    max_pool = nn.MaxPool2d(3, stride=1, padding=1)
    SLight_mask = -max_pool(-SLight_mask_ori)  # morphological dilation
    SLight_area = SLight_mask_ori.sum(dim=[1, 2, 3])  # per batch

    SLight_loss = torch.zeros(B, device=device)
    for b in range(B):
        if SLight_area[b] > 25:
            SLight_region = SLight_mask[b] * real_gray[b]
            mean_val = SLight_region.sum() / SLight_area[b]
            high_mask = (SLight_region > mean_val).float()
            fake_region_high = high_mask * fake_ir_gray[b] + (1 - high_mask)

            if veg_mask[b].sum() > 0:
                veg_region = veg_mask[b] * fake_ir_gray[b]
                veg_mean = veg_region.sum() / veg_mask[b].sum()
                SLight_loss[b] = F.relu(veg_mean.detach() + 0.25 - fake_region_high.min())
            else:
                SLight_loss[b] = F.relu(0.7 - fake_region_high.min())

    ########### Light region SGA loss
    light_mask_all = light_mask_ori + SLight_mask_ori
    EM_masked = light_mask_all * real_edges
    GM_masked = light_mask_all * fake_gradmap
    edge_sum = EM_masked.sum(dim=[1, 2, 3])
    loss_sga_light = torch.zeros(B, device=device)
    valid_idx = edge_sum > 0
    if valid_idx.any():
        fake_grad_norm = GM_masked[valid_idx] / (
                GM_masked[valid_idx].amax(dim=[1, 2, 3], keepdim=True) + 1e-4)
        loss_sga_light[valid_idx] = 0.5 * (
                F.relu(0.8 * EM_masked[valid_idx] - fake_grad_norm).sum(dim=[1, 2, 3]) / edge_sum[valid_idx])

    ########### Traffic Light luminance adjustment
    TLight_mask = closing(TLight_mask, torch.ones(5, 5, device=device))
    TLight_mask = erosion(TLight_mask, torch.ones(3, 3, device=device))
    TLight_area = TLight_mask.sum(dim=[1, 2, 3])
    TLight_loss = torch.zeros(B, device=device)
    valid_idx = TLight_area > 100
    if valid_idx.any():
        real_TLight = real_lab[valid_idx, :1]
        real_TLight_norm = (real_TLight - real_TLight.min()) / (real_TLight.max() - real_TLight.min())
        fake_real_TLight = (1 - real_TLight_norm)*2-1  # [-1 1]
        fake_region = fake_ir_norm.mean(1, keepdim=True)[valid_idx]
        TLight_loss[valid_idx] += (F.relu(TLight_mask[valid_idx] * ((fake_real_TLight - fake_region)**2 - 0.1))
                                   .sum(dim=[1, 2, 3]) / torch.sqrt(TLight_area[valid_idx] + 1e-6))
        # real_light_lab = TLight_mask[valid_idx] * real_lab[valid_idx]
        # color_magn = real_light_lab[:, :1] + (real_light_lab[:, 1:] ** 2).sum(1, keepdim=True)
        # color_mean = color_magn.sum(dim=[1, 2, 3]) / TLight_area[valid_idx]
        # light_high_mask = (color_magn > 1.2 * color_mean.view(-1, 1, 1, 1)).float()
        # light_high_mask = opening(light_high_mask, torch.ones(3, 3, device=device))
        # high_ratio = light_high_mask.sum(dim=[1, 2, 3]) / TLight_area[valid_idx]
        # mask_idx = high_ratio > 0.1
        # if mask_idx.any():
        #     real_luminance_region = TLight_mask[valid_idx][mask_idx] * real_light_lab[:, :1][mask_idx]
        #     fake_luminance_region = TLight_mask[valid_idx][mask_idx] * fake_ir_gray[valid_idx, :1][mask_idx]
        #     TLight_loss[valid_idx][mask_idx] = F.relu(0.8 - (fake_luminance_region * light_high_mask[mask_idx] *
        #                                                      real_luminance_region * light_high_mask[
        #                                                          mask_idx].detach()).sum(dim=[1, 2, 3]) /
        #                                               light_high_mask[mask_idx].sum())

    ABC_losses = SLight_loss.sum() + loss_sga_light.sum() + TLight_loss.sum()  #+ sign_temp_loss.sum()

    # ########## Color Bias Correction
    # Masks
    rec_losses = torch.zeros(B, device=device)
    for mask, threshold in zip([sign_mask, TLight_mask, motorcycle_mask, road_mask], [10, 10, 10, 100]):
        valid_idx = mask.sum(dim=[1, 2, 3]) > threshold
        if valid_idx.any():
            rec_losses[valid_idx] += PixelConsistencyLoss(rec_vis[valid_idx], real_vis[valid_idx], mask[valid_idx], 3)

    CBC_losses = rec_losses.sum()

    ############ Thermal Channel equality loss
    thermal_eq_loss = torch.max(torch.max(fake_IR, 1)[0] - torch.min(fake_IR, 1)[0])

    total_loss = ABC_losses + CBC_losses + thermal_eq_loss
    return total_loss


def CondGradRepaLoss(fake_img, fake_mask, fake_grad, real_grad):
    """
    Conditional Gradient Repair loss for background categories.
    fake_img: fake vis image. B x 3 x H x W
    fake_mask: IR seg mask (tensor). B x 1 x H x W
    fake_grad: gradient map of fake vis image. B x 2 x H x W
    real_grad: gradient map of real IR image. B x 2 x H x W
    """
    if fake_mask is None:
        fake_mask = torch.ones((fake_img.size(0), 1, fake_img.size(2), fake_img.size(3)),
                               device=fake_img.device) * 255

    # 1. Build fused background mask (boolean)
    IR_bkg_fuse_mask = (fake_mask < 11) | (fake_mask == 255)
    # Broadcast mask to match grad shapes if needed
    mask = IR_bkg_fuse_mask.float()
    if mask.shape[-2:] != real_grad.shape[-2:]:
        mask = F.interpolate(mask, size=real_grad.shape[-2:], mode='nearest')
    # 2. Extract background gradients
    IR_grad_bkg = real_grad * mask
    vis_grad_bkg = fake_grad * mask
    # 3. Sum of IR background gradients
    IR_grad_bkg_sum = IR_grad_bkg.sum()
    if IR_grad_bkg_sum <= 0:
        return torch.tensor(0.0, device=fake_img.device)
    # 4. Mean of IR background gradients
    IR_grad_bkg_mean = IR_grad_bkg_sum / mask.sum()
    # 5. Mask of high IR gradients
    IR_grad_bkg_high_mask = (IR_grad_bkg > IR_grad_bkg_mean).float()
    IR_grad_bkg_high = IR_grad_bkg_high_mask * IR_grad_bkg
    vis_grad_bkg_high = IR_grad_bkg_high_mask * vis_grad_bkg
    IR_grad_bkg_high_sum = IR_grad_bkg_high.sum()
    if IR_grad_bkg_high_sum <= 0:
        return torch.tensor(0.0, device=fake_img.device)
    # 6. Actual loss (EC loss)
    bkg_EC_loss = (F.relu(IR_grad_bkg_high.detach() - vis_grad_bkg_high).sum() /
                   IR_grad_bkg_high_sum.detach())
    return bkg_EC_loss


def AdaptativeColAttentionLoss(real_vis_mask, real_vis_fea,
                               fake_vis_mask, fake_vis_fea,
                               cluster_num, max_iter):
    """Adaptive Collaborative Attention Loss with class-presence masks."""

    B, C, H, W = real_vis_fea.size()
    if fake_vis_mask is None:
        fake_vis_mask = torch.ones_like(real_vis_mask, device=real_vis_mask.device) * 255

    # Resize masks to feature map resolution (no expand needed)
    real_mask = F.interpolate(real_vis_mask.float(), size=[H, W], mode='nearest')
    fake_mask = F.interpolate(fake_vis_mask.float(), size=[H, W], mode='nearest')

    # --- CLASS DEFINITIONS ---
    # Light: 6
    # Sign: 7
    # Person: 11
    # Vehicle: 13–16 (exclusive of 12, inclusive of 13 to 16)
    # Motor: 17

    Light_mask_real = (real_mask == 6).float()
    Sign_mask_real = (real_mask == 7).float()
    Person_mask_real = (real_mask == 11).float()
    Vehicle_mask_real = ((real_mask > 12) & (real_mask < 17)).float()
    Motor_mask_real = (real_mask == 17).float()

    Light_mask_fake = (fake_mask == 6).float()
    Sign_mask_fake = (fake_mask == 7).float()
    Person_mask_fake = (fake_mask == 11).float()
    Vehicle_mask_fake = ((fake_mask > 12) & (fake_mask < 17)).float()
    Motor_mask_fake = (fake_mask == 17).float()

    # --- Helper: compute one class loss if both masks have enough pixels ---
    def maybe_loss(real_m, fake_m):
        if (real_m.sum() > cluster_num) and (fake_m.sum() > cluster_num):
            return (ClsACALoss(real_vis_fea, real_m, fake_vis_fea, fake_m,
                               C, cluster_num, max_iter), 1.0)
        return 0.0, 0.0

    # Compute for each class
    loss_light, idx_light = maybe_loss(Light_mask_real, Light_mask_fake)
    loss_sign, idx_sign = maybe_loss(Sign_mask_real, Sign_mask_fake)
    loss_person, idx_person = maybe_loss(Person_mask_real, Person_mask_fake)
    loss_vehicle, idx_vehicle = maybe_loss(Vehicle_mask_real, Vehicle_mask_fake)
    loss_motor, idx_motor = maybe_loss(Motor_mask_real, Motor_mask_fake)

    # Number of valid classes
    obj_cls_num = idx_light + idx_sign + idx_person + idx_vehicle + idx_motor

    # Average over valid classes
    if obj_cls_num > 0:
        total_loss = (loss_light + loss_sign + loss_person + loss_vehicle + loss_motor) / obj_cls_num
    else:
        total_loss = torch.tensor(0.0, device=real_vis_fea.device)

    return total_loss


def ClsACALoss(real_vis_fea, cls_mask_real, fake_vis_fea, cls_mask_fake,
               fea_dim, cluster_num, max_iter):
    """
    Adaptive Collaborative Attention Loss for one class.
    real_vis_fea, fake_vis_fea : (B, C, H, W)
    cls_mask_* : (B, 1, H, W) masks with 0/1 values
    """

    # Mask features (broadcast automatically)
    real_fea_masked = real_vis_fea * cls_mask_real
    fake_fea_masked = fake_vis_fea * cls_mask_fake

    # Flatten spatial dimensions -> shape: (N, C)
    # N = number of pixels
    real_flat = real_fea_masked.permute(0, 2, 3, 1).reshape(-1, fea_dim)
    fake_flat = fake_fea_masked.permute(0, 2, 3, 1).reshape(-1, fea_dim)

    # Keep only non-zero rows (valid class pixels)
    real_nonzero = real_flat.abs().sum(dim=1) > 0
    real_flat = real_flat[real_nonzero]

    # Normalize pixel features
    real_norm = F.normalize(real_flat, p=2, dim=1)

    # ---- Cluster centers from real features ----
    centers = GetFeaMatrixCenter(real_norm, cluster_num, max_iter)  # (K, C)
    centers_norm = F.normalize(centers, p=2, dim=1)

    # ---- Real similarity ----
    sim_real = real_norm @ centers_norm.T  # (N_real, K)
    sim_real_max = sim_real.max(dim=1).values.mean()  # mean over pixels
    sim_real_cluster = sim_real.max(dim=0).values.mean()  # mean over clusters

    # ---- Fake similarity ----
    # Only normalize rows with non-zero values
    fake_nonzero = fake_flat.abs().sum(dim=1) > 0
    fake_flat = fake_flat[fake_nonzero]
    fake_norm = F.normalize(fake_flat, p=2, dim=1)

    sim_fake = fake_norm @ centers_norm.T  # (N_fake, K)
    sim_fake_max = sim_fake.max(dim=1).values.mean()  # mean over pixels
    sim_fake_cluster = sim_fake.max(dim=0).values.mean()  # mean over clusters
    # ---- Loss terms ----
    loss_sim = F.relu(0.9 * sim_real_max.detach() - sim_fake_max)
    loss_div = F.relu(0.9 * sim_real_cluster.detach() - sim_fake_cluster)

    return loss_sim + loss_div


def PixelConsistencyLoss(inputs_img, GT_img, ROI_mask, ssim_winsize):
    "Pixel-wise Consistency Loss. inputs_img and GT_img are 4D tensors range [-1, 1], while ROI_mask is a 2D tensor."
    input_masked = inputs_img.mul(ROI_mask.expand_as(inputs_img))
    GT_masked = GT_img.mul(ROI_mask.expand_as(GT_img))
    if len(ROI_mask.size()) == 4:
        _, _, h, w = ROI_mask.shape
        area_ROI = torch.sum(ROI_mask[0, 0, :, :])
    elif len(ROI_mask.size()) == 3:
        _, h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask[0, :, :])
    else:
        h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask)

    criterionSSIM = SSIM_Loss()
    criterionL1 = torch.nn.SmoothL1Loss()
    lambda_L1 = 10.0
    if area_ROI > 0:
        losses = ((h * w) / area_ROI) * (lambda_L1 * criterionL1(input_masked, GT_masked.detach()) + \
                                         criterionSSIM((input_masked + 1) / 2, (GT_masked.detach() + 1) / 2))
    else:
        losses = 0.0

    return losses
