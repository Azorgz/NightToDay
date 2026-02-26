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
from kornia import create_meshgrid
from kornia.color import rgb_to_lab, rgb_to_hsv
from kornia.contrib import connected_components
from kornia.filters import sobel, laplacian, get_gaussian_kernel2d, median_blur, bilateral_blur
from kornia.morphology import erosion, closing, dilation
from torch.nn import LeakyReLU, ReLU
from torchmetrics.functional.image import image_gradients
from torchvision.transforms.v2 import GaussianBlur

from .ssim import SSIM
from .utilities import ClsMeanPixelValue, GetFeaMatrixCenter, bhw_to_onehot, determine_color_N, \
    center_of_mass, detect_TL_blobs_mask_free, getLightDarkRegionMean, RefineLightMask, detect_TL_colorblobs_mask_free, \
    get_disk_kernel

ROAD = 0
PAVEMENT = 1
BUILDING = 2
POLE = 5
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


class ColorConsistencyLoss(nn.Module):
    """
    Color Consistency Loss to ensure color fidelity between generated and target images.
    """

    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        self.lambda_saturation = 1.
        self.lambda_lab_edge = 0.1
        self.lambda_l1 = 0.1

    def forward(self, fake_color: torch.Tensor, real_color: torch.Tensor, mask_high_color) -> torch.Tensor:
        """
        Computes the color consistency loss.
        :param fake_color: Generated color image tensor of shape (B, 3, H, W).
        :param real_color: Target color image tensor of shape (B, 3, H, W).
        :return: Color consistency loss value.
        """
        self_lab_edge_loss = self.lab_edge_sharpness_loss(fake_color)
        self_saturation_loss = self.saturation_loss_color(fake_color * mask_high_color,
                                                          (real_color * mask_high_color))
        l1_loss = self.L1_loss_color(fake_color, real_color)
        return self_saturation_loss + l1_loss + self_lab_edge_loss

    def luminance_loss_color(self, fake_rgb, real_rgb):
        if not self.lambda_l1:
            return 0.0
        fake_lum = 0.299 * fake_rgb[:, 0:1, :, :] + 0.587 * fake_rgb[:, 1:2, :, :] + 0.114 * fake_rgb[:, 2:3, :, :]
        real_lum = 0.299 * real_rgb[:, 0:1, :, :] + 0.587 * real_rgb[:, 1:2, :, :] + 0.114 * real_rgb[:, 2:3, :, :]
        real_lum = real_lum * (real_lum > 0.3).float() * (real_lum < 0.90).float()
        return torch.relu(real_lum + 0.05 - fake_lum) * self.lambda_l1

    def L1_loss_color(self, fake_rgb, real_rgb):
        if not self.lambda_l1:
            return 0.0
        mask = ((0.1 < real_rgb) * (real_rgb < 0.90)).float()
        real_rgb = real_rgb * mask
        fake_rgb = fake_rgb * mask
        weight_lum = 1 - (real_rgb.mean(1, keepdim=True) - 0.5).abs() * 2
        weight_sat = ((real_rgb.max(1, keepdim=True)[0] - real_rgb.min(1, keepdim=True)[0]) * 2).clamp(0, 1)
        weights = torch.max(torch.cat([weight_lum, weight_sat], dim=1), dim=1, keepdim=True)[0]
        loss = F.l1_loss(fake_rgb, real_rgb, reduction='none') + SSIM_Loss()(fake_rgb, real_rgb) * weights
        return (loss ** 2 + loss).mean() * self.lambda_l1

    # def saturation_loss_color(self, fake_rgb, real_n, tau=0.1):
    #     """
    #     thermal: (B,1,H,W) normalized to [0,1]
    #     """
    #     if not self.lambda_saturation:
    #         return 0.0
    #     hsv_fake = rgb_to_hsv(fake_rgb)
    #     hsv_real = rgb_to_hsv(real_n)
    #     V_mask = ((hsv_real[:, 2] > 0.33).float() * (hsv_real[:, 2] < 0.95)).squeeze(1)
    #     weight = ((hsv_real[:, 1]*1.1 > hsv_fake[:, 1]) * V_mask).float().squeeze(1)
    #     Sat_diff = torch.relu(hsv_real[:, 1] - hsv_fake[:, 1] + tau).squeeze(1)
    #     S = Sat_diff
    #     H = torch.sqrt((hsv_real[:, 0] - hsv_fake[:, 0]) ** 2 + 1e-6).squeeze(1)
    #     loss = (S * H * weight).sum() / (weight.sum() + 1e-6)
    #     return loss * self.lambda_saturation

    def saturation_loss_color(self, fake_rgb, real_n):
        """
        thermal: (B,1,H,W) normalized to [0,1]
        """
        if not self.lambda_saturation:
            return 0.0
        real_maxc, _ = real_n.max(dim=1, keepdim=True)
        real_minc, _ = real_n.min(dim=1, keepdim=True)
        real_sat = ((real_maxc - real_minc) / (real_maxc + 1e-6))

        fake_maxc, _ = fake_rgb.max(dim=1, keepdim=True)
        fake_minc, _ = fake_rgb.min(dim=1, keepdim=True)
        fake_sat = ((fake_maxc - fake_minc) / (fake_maxc + 1e-6))

        hue_fake = rgb_to_hsv(fake_rgb)[:, 0:1, :, :]
        hue_real = rgb_to_hsv(real_n)[:, 0:1, :, :]
        cos_dist = torch.cos(hue_fake - hue_real) ** 3
        # coeff = (real_n_ds > 0.25).float() * (real_n_ds < 0.90).float()
        return (torch.relu(real_sat - fake_sat * cos_dist)).mean() * self.lambda_saturation

    def lab_edge_sharpness_loss(self, fake):
        if not self.lambda_lab_edge:
            return 0.0
        lab_fake = rgb_to_lab(fake)
        L_channel, ab_channel = lab_fake[:, 0:1, :, :], lab_fake[:, 1:3, :, :]
        color_edges = laplacian(ab_channel, kernel_size=3)
        return torch.relu(0.7 - (abs(color_edges)).mean()) * self.lambda_lab_edge


def cycle_loss(real, fake):
    """
    Cycle consistency loss between real and fake images.
    """
    h, s, v = rgb_to_hsv(real).split(3, 1)
    s = s ** 1.2
    hsv_real = torch.cat([h, s, v], dim=1)
    hsv_fake = rgb_to_hsv(fake)
    loss = nn.SmoothL1Loss(beta=0.5)(hsv_fake, hsv_real.detach()) + SSIM()(real, fake) * 5
    return loss


def ColorLoss(fake_color, real_color, GT_seg=None, th_high=0.95, th_low=0.15, weights=None):
    """
    Color loss that focuses on color differences in high saturation and medium luminance areas.
    """
    # Normalize back to [0,1]
    assert fake_color.shape == real_color.shape, "Input images must have the same dimensions."
    im_fake = ImageTensor(fake_color * 0.5 + 0.5)
    im_target = ImageTensor(real_color * 0.5 + 0.5)
    B = im_fake.shape[0]
    # color distance function
    color_dist = im_fake.color_distance(im_target)  # shape (B,1,H,W) or (B,H,W)
    hsv_target = im_target.HSV()
    color_magn = hsv_target[:, 1:2, :, :] * hsv_target[:, 2:3, :, :]
    target_l, target_ab = rgb_to_lab(im_target.to_tensor()).split([1, 2], 1)  # L channel
    low_lum = target_l < th_low * 100
    high_lum = target_l > th_high * 100
    valid = ~low_lum * ~high_lum * 1.
    color_dist = F.relu((color_dist - 0.01) * valid)
    ssim_loss = F.relu(
        SSIM_Loss(channel=3)(im_fake, im_target, full=True).mean(dim=1, keepdim=True) * (1 - low_lum * 1.) - 0.05)
    color_mask = target_l.clamp(0, 25) * ((target_ab / 128) ** 2).sum(1, keepdim=True)
    high_color_mask = (color_mask > color_mask.mean()) * valid
    weights = weights[[CAR, TRUCK, BUILDING, MOTORCYCLE, SIGN, ROAD]].to(im_target.device) if weights is not None else \
        torch.ones(6, device=fake_color.device)
    H, W = im_fake.shape[-2:]
    loss = torch.zeros([B, ], device=im_target.device)

    if GT_seg is not None:
        if color_dist.shape[-2:] != GT_seg.shape[-2:]:
            GT_seg = F.interpolate(GT_seg.float(), size=(H, W), mode='nearest').long()
        valid = valid * (GT_seg != SKY)

        # build masks in a single vectorized call
        # ------------------------------------------------------------
        # Class masks (exact value matches)
        classes_eq = torch.tensor([CAR, TRUCK, BUILDING, MOTORCYCLE, SIGN], device=GT_seg.device).view(1, -1, 1, 1)

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
        masked_loss = (color_dist * masks * color_magn / (color_magn.max() / 2)).sum(dim=(2, 3))  # (B,6)
        masked_ssim_loss = (ssim_loss * masks[:, [4, 5]]).sum(dim=(2, 3))  # (B,2)

        # Avoid division by zero: zero out empty masks
        per_class_loss = masked_loss / (sum_masks + 1e-6)  # (B,6)
        per_class_ssim_loss = masked_ssim_loss / (sum_masks[:, [4, 5]] + 1e-6)  # (B,2)

        # Equivalent to summing only valid classes:
        valid_mask = (sum_masks > 0).float()
        sum_losses = (((per_class_loss * valid_mask * weights).sum(dim=1) / weights.mean() +
                       (per_class_ssim_loss * valid_mask[:, [4, 5]] * weights[[4, 5]]).sum(dim=1))
                      / weights[[2, 4, 5]].mean())  # (B,)
        # average batch
        loss += sum_losses
    else:
        loss += (color_dist * high_color_mask).sum(dim=[1, 2, 3]) / (high_color_mask.sum(dim=[1, 2, 3]) + 1e-6)
    loss += ColorConsistencyLoss()(im_fake.to_tensor(), im_target.to_tensor(), valid) * 0.2

    return loss.mean()


def ThermalLoss(TN, T, N, GT_seg, weights=None):
    """
    Thermal Loss to enhance the thermal characteristics of the fused image.
    The texture gradient will be maximized in vegetation and vehicles regions
    :param TN: fused thermal image
    :param T: real thermal image
    :param N: real night color image
    :param GT_seg:
    :param weights:
    :return: loss value
    """
    B = TN.shape[0]
    device = T.device
    GT_resized = F.interpolate(GT_seg.float(), size=TN.shape[-2:], mode='nearest').long().detach()
    weights = weights[[SKY, VEG, PERSON, CAR]] if weights is not None else (
        torch.tensor([1., 1., 1., 1.], device=TN.device))
    weights[0] *= 0.2  # sky
    sky_mask = erosion((GT_resized == SKY).float(), torch.ones(3, 3, device=TN.device))
    area_sky = sky_mask.sum(dim=[1, 2, 3])
    valid_sky = area_sky > 200
    veg_mask = (GT_resized == VEG).float()
    area_veg = veg_mask.sum(dim=[1, 2, 3])
    valid_veg = area_veg > 200
    person_mask = erosion((GT_resized == PERSON).float(), torch.ones(5, 5, device=TN.device))
    area_person = person_mask.sum(dim=[1, 2, 3])
    valid_person = area_person > 30
    # car_mask = sum([GT_resized == V for V in VEHICLES]).float()
    # area_car = car_mask.sum(dim=[1, 2, 3])
    # valid_car = area_car > 50
    # building_mask = (GT_resized <= 2).float()
    # area_building = building_mask.sum(dim=[1, 2, 3])
    # valid_building = area_building > 200
    thermal_diff_low = ReLU()(TN - T + 0.1)  # only penalize higher values
    thermal_diff_high = ReLU()(T - TN)  # only penalize lower values
    T_filtered = median_blur(T.mean(1, keepdim=True), kernel_size=3)
    N_filtered = median_blur(-N.mean(1, keepdim=True), kernel_size=3)
    min_values = torch.min(torch.cat([T_filtered, N_filtered], dim=1), dim=1, keepdim=True)[0]

    # losses init
    sky_loss = torch.zeros([B, ], device=device)
    veg_loss = torch.zeros([B, ], device=device)
    person_loss = torch.zeros([B, ], device=device)
    # blobs_loss = torch.zeros([B, ], device=image_target.device)

    #  Thermal correction losses per classes
    mask_up_low = (T[:, :, ::2] < -0.85).float()
    sky_loss[valid_sky] += ((thermal_diff_low[:, :, ::2] * mask_up_low).sum(dim=[1, 2, 3])
                            / (mask_up_low.sum(dim=[1, 2, 3]) + 1e-6) * 2)
    # sky_loss[valid_sky] += (thermal_diff_low[valid_sky] * sky_mask[valid_sky]).sum(dim=[1, 2, 3]) / area_sky[
    #     valid_sky] * 5
    # sky_loss[valid_sky] += (thermal_diff_low * sky_mask).sum(dim=[1, 2, 3]) / (sky_mask.sum(dim=[1, 2, 3]) + 1e-6)
    # veg_loss[valid_veg] += torch.relu((min_values[valid_veg] + 0.1 - TN[valid_veg]) * veg_mask[valid_veg].detach()).sum(
    #     dim=[1, 2, 3]) / area_veg[valid_veg] * 5
    # veg_loss[valid_veg] += (thermal_diff_high[valid_veg] * veg_mask[valid_veg]).sum(dim=[1, 2, 3]) / area_veg[valid_veg]
    # veg_loss[valid_sky*valid_veg] += ReLU()((TN*sky_mask)[valid_sky*valid_veg].sum(dim=[1, 2, 3]) / area_sky[valid_sky*valid_veg] -
    #                                         (TN*veg_mask)[valid_sky*valid_veg].sum(dim=[1, 2, 3]) / area_veg[valid_sky*valid_veg] * 0.8)
    # sky_loss[valid_sky] += ReLU()((TN * sky_mask)[valid_sky * valid_veg].sum(dim=[1, 2, 3]) / (area_sky[valid_sky].sum(dim=[1, 2, 3]) + 1e-6))

    person_loss[valid_person] += (thermal_diff_high[valid_person] * person_mask[valid_person]).sum(dim=[1, 2, 3]) / \
                                 area_person[valid_person]
    total_classes_loss = ((sky_loss * weights[0] + veg_loss * weights[1] + person_loss * weights[2]) /
                          (weights[:3] * torch.stack([valid_sky, valid_veg, valid_person], dim=-1).float()).sum(
                              1)).mean()
    grad_TN_y, grad_TN_x = image_gradients(TN)
    grad_T_y, grad_T_x = image_gradients(T)
    grad_N_y, grad_N_x = image_gradients(T)
    gradient_loss = (torch.relu(torch.abs(grad_T_x) - torch.abs(grad_TN_x)) +
                     torch.relu(torch.abs(grad_N_x) - torch.abs(grad_TN_x)) +
                     torch.relu(torch.abs(grad_N_y * grad_T_y) - grad_TN_y ** 2))

    thermal_noise_loss = ThermalNoiseLoss()(TN, T, N).mean() * 2

    return total_classes_loss + thermal_noise_loss + gradient_loss.mean() * 0.5


def TL_fake_loss(D, f_T, GT_mask_D):
    """
    Function that optimize the traffic light reconstruction in the fake thermal image
    :param D:
    :param f_T:
    :param GT_mask_D:
    :return:
    """
    loss_tl = torch.zeros([D.shape[0], ], device=D.device)
    if not (GT_mask_D == TRAFFICLIGHT).any():
        return loss_tl
    GT_mask = RefineLightMask(GT_mask_D, D)
    area_trafficlight = GT_mask.sum(dim=[1, 2, 3])
    valid_trafficlight = area_trafficlight > 50
    if not valid_trafficlight.any():
        return loss_tl
    traffic_light_gray = (D[valid_trafficlight].mean(1, keepdim=True) * 0.5 + 0.5) * GT_mask[valid_trafficlight]
    traffic_light_color_mean = traffic_light_gray.sum(dim=[2, 3], keepdim=True) / (
            area_trafficlight[valid_trafficlight] + 1e-6)
    traffic_light_color_std = ((traffic_light_color_mean * GT_mask[valid_trafficlight] - traffic_light_gray) ** 2).sum(
        dim=[1, 2, 3]) / (area_trafficlight[valid_trafficlight] + 1e-6)
    traffic_light_color_normalized = (traffic_light_gray - traffic_light_color_mean * GT_mask[
        valid_trafficlight]) / torch.sqrt(traffic_light_color_std + 1e-6)

    fake_thermal_TL = f_T[valid_trafficlight] * GT_mask[valid_trafficlight]
    fake_thermal_TL_mean = fake_thermal_TL.sum(dim=[2, 3], keepdim=True) / (
            area_trafficlight[valid_trafficlight] + 1e-6)
    fake_thermal_TL_std = ((fake_thermal_TL_mean * GT_mask[valid_trafficlight] - fake_thermal_TL) ** 2).sum(
        dim=[1, 2, 3]) / (area_trafficlight[valid_trafficlight] + 1e-6)
    fake_thermal_TL_normalized = (fake_thermal_TL - fake_thermal_TL_mean * GT_mask[valid_trafficlight]) / torch.sqrt(
        fake_thermal_TL_std + 1e-6)
    loss_tl[valid_trafficlight] = (ReLU()(
        torch.sqrt((traffic_light_color_normalized - fake_thermal_TL_normalized) ** 2 + 1e-6)) - 0.1).sum(dim=[1, 2, 3])

    # --- Traffic Light reconstitution ---
    return loss_tl


def TL_color_loss(real_T, real_N, fused_TN, fake_D, GT_seg):
    mask_tl = (GT_seg == TRAFFICLIGHT).float()
    mask_tl = F.interpolate(mask_tl, size=real_T.shape[-2:], mode='nearest')
    area_tl = mask_tl.sum(dim=[1, 2, 3])
    valid_tl = area_tl > 50
    if not valid_tl.any():
        return torch.zeros([real_T.shape[0], ], device=real_T.device)
    blobs = detect_TL_colorblobs_mask_free(real_N[valid_tl] * 0.5 + 0.5, real_T[valid_tl] * 0.5 + 0.5)
    mask = blobs.mean(1, keepdim=True) > 0
    mask_dilated = mask.clone().float()
    blobs_labels = connected_components(mask.float())
    uniques = blobs_labels.unique(return_counts=True)
    for label, count in zip(uniques[0], uniques[1]):
        if label == 0:
            continue
        if count < 500 or count > 10000:
            radius = int(torch.sqrt(count / torch.pi)) * 2 + 1
            mask_dilated += dilation((blobs_labels == label).float(),
                                     torch.ones(3 * radius, radius, device=real_T.device))
    mask_dilated = ((mask_dilated > 0) ^ mask).float()
    loss_tl = ReLU()(mask * torch.sqrt((blobs - fake_D[valid_tl]) ** 2 + 1e-6) - 1e-2).mean(dim=[1, 2, 3])
    loss_tl += ReLU()(mask * torch.sqrt(
        (mask_dilated * real_T[valid_tl] - mask_dilated * fused_TN[valid_tl]) ** 2 + 1e-6) - 1e-2).mean(dim=[1, 2, 3])
    return loss_tl


# def TL_color_loss(I_ir, I_vi, I_fused, GT_mask):
#     loss_tl = torch.zeros([I_ir.shape[0], ], device=I_ir.device)
#     mask_trafficlight = (GT_mask == TRAFFICLIGHT).float()
#     area_trafficlight = mask_trafficlight.sum(dim=[1, 2, 3])
#     valid_trafficlight = area_trafficlight > 50
#     if not valid_trafficlight.any():
#         return loss_tl
#     blobs = detect_TL_blobs_mask_free(I_vi*0.5+0.5, I_ir*0.5+0.5)
#     labels = connected_components(mask_trafficlight)
#     fake_TL = torch.zeros_like(GT_mask, device=GT_mask.device).float()
#     for i, val in enumerate(valid_trafficlight):
#         if val:
#             ir = (I_ir[i]*0.5+0.5).mean(0)
#             unique = labels[i].unique(return_counts=True)
#             for label, count in zip(unique[0], unique[1]):
#                 if count > 10000 or count < 50:
#                     continue
#                 else:
#                     size_scaler = (count // 50)
#                     mask_tl = (labels[i][None] == label).float()
#                     mask_dilated = dilation(mask_tl, torch.ones(5*size_scaler, 3*size_scaler, device=GT_mask.device)).squeeze()
#                     mask_tl = mask_tl.squeeze()
#                     contours = mask_dilated - mask_tl
#                     if (contours * I_ir[i]).mean() > (mask_tl * I_ir[i]).mean():
#                         fake_TL[i] += 1.02 * contours * ir + mask_tl*ir*0.98 - mask_dilated * blobs[i] * 0.9 * ir
#                     else:
#                         fake_TL[i] += 0.98 * contours * ir + mask_tl*ir*1.02 - mask_dilated * blobs[i] * 0.9 * ir
#
#     # --- Traffic Light reconstitution ---
#     return ReLU()(((fake_TL > 0) * (I_fused*0.5+0.5).mean(dim=1, keepdim=True) - fake_TL.clamp(0, 1)) ** 2).mean(dim=[1, 2, 3]).sum()

# def create_fake_TL(I_ir, I_vis, mask):
#     # extract shape as mean sum(dim[mean +-std())
#     I_ir = (I_ir * 0.5 + 0.5).mean(dim=1).squeeze()
#     mask = mask.squeeze()
#     real_IR_TLight = I_ir * mask
#     cx, cy = center_of_mass(mask[None, None])
#     h0, w0 = mask.sum(-2), mask.sum(-1)
#     h_mean, h_std, w_mean, w_std = h0[h0 > 0].mean(), h0[h0 > 0].std(), w0[w0 > 0].mean(), w0[w0 > 0].std()
#     h = int(h0[(h0 <= (h_mean + h_std)) * (h0 >= (h_mean - h_std))].mean())
#     w = int(w0[(w0 <= (w_mean + w_std)) * (w0 >= (w_mean - w_std))].mean())
#     if w0[0] > 0:
#         cut = True
#         pad = [0, 0, max(w * 3 - h, 0), 0]  # left, right, top, bottom
#         h = w * 3
#     else:
#         cut = False
#         pad = [0, 0, 0, 0]
#     #  vertical borders
#     dx = (torch.abs(I_ir[:, 1:] - I_ir[:, :-1])*mask[:, 1:]).sum(-2)
#     left_border = torch.argmax(dx[: max(cx.long()-w//4, 0)], dim=-1) or 0
#     right_border = torch.argmax(dx[max(cx.long()+w//4, dx.shape[-1]-1):], dim=-1) or dx.shape[-1]-1
#
#     # real IR values in the TL area
#     TL_ir_target = I_ir[mask > 0].mean() + I_ir[mask > 0].std()
#
#     # switch different case shape of TLights:
#     if (w / h) < 0.7:
#         # 3 vertical light
#         base = torch.ones((1, 1, h, h // 3), device=I_ir.device) * TL_ir_target
#         gaussian_kernel = get_gaussian_kernel2d(((h//6)*2 + 1, (h//6)*2 + 1), (h / 6, h / 6)).to(I_ir.device)
#         color = determine_color(I_vis * mask)
#         positions = [h // 6, h // 2, h * 5 // 6], [h // 6, h // 6, h // 6]
#         w = h // 3
#         lights = torch.zeros((h, w), device=I_ir.device)
#         for i, (pos_y, pos_x) in enumerate(zip(*positions)):
#             if color == 'red' and i == 0:
#                 lights[pos_y, pos_x] = -1.0
#             elif color == 'green' and i == 2:
#                 lights[pos_y, pos_x] = -1.0
#             elif color == 'yellow' and i == 1:
#                 lights[pos_y, pos_x] = -1.0
#             else:
#                 lights[pos_y, pos_x] = 1.0
#
#     elif (w / h) < 0.9:
#         # 2 vertical light
#         base = torch.ones((1, 1, h, h // 2), device=I_ir.device) * TL_ir_target
#         gaussian_kernel = get_gaussian_kernel2d(((h//6)*2+1, (h // 6)*2+1), (h / 6, h / 6)).to(I_ir.device)
#         color = determine_color(I_vis * mask)
#         positions = [h // 3, h * 2 // 3], [h // 4, h // 4, h // 4]
#         w = h // 2
#         lights = torch.zeros((h, w), device=I_ir.device)
#         for i, (pos_y, pos_x) in enumerate(zip(*positions)):
#             if color == 'red' and i == 0:
#                 lights[pos_y, pos_x] = -1.0
#             elif color == 'green' and i == 2:
#                 lights[pos_y, pos_x] = -1.0
#             else:
#                 lights[pos_y, pos_x] = 1.0
#     else:
#         # square light
#         return torch.zeros_like(I_ir, device=I_ir.device).float()
#
#     lights = F.conv2d(lights[None, None], gaussian_kernel[None], padding=gaussian_kernel.shape[-1] // 2)
#     lights_normed = (lights - lights.min()) / (lights.max() - lights.min()) * torch.abs(TL_ir_target) - torch.abs(TL_ir_target / 2)
#     fake_TL = (base + lights_normed)
#     # place back to original image size
#     fake_TL_full = torch.zeros_like(I_ir[None, None], device=I_ir.device).float()
#     if cut:
#         fake_TL_full = F.pad(fake_TL_full, pad, "constant", 0)
#     x_c, y_c = center_of_mass(F.pad(mask[None, None], pad))
#     y1, y2 = int(max(0, y_c - h // 2)), int(min(I_ir.shape[0], y_c + (h+1) // 2))
#     x1, x2 = int(max(0, x_c - w // 2)), int(min(I_ir.shape[1], x_c + (w+1) // 2))
#     fake_TL_full[:, :, y1:y2, x1:x2] = fake_TL
#     # unpad if cut
#     if cut:
#         fake_TL_full = fake_TL_full[:, :, pad[2]:, :]
#     return fake_TL_full[0]
def ForegroundContourLoss(fake, GT_seg):
    GT = F.interpolate(GT_seg.float(), size=fake.shape[-2:], mode='nearest')
    sky_mask = (GT == SKY).float()
    Foreground_mask = (GT == SIGN).float() + (GT == POLE).float() + (GT == TRAFFICLIGHT).float() + (
            GT == STREETLIGHT).float()
    Foreground_contour = dilation(Foreground_mask, torch.ones(5, 5, device=GT.device)) - Foreground_mask
    sky_contour = Foreground_contour * sky_mask
    valid_mask = sky_contour.sum(dim=[1, 2, 3]) > 0
    if valid_mask.any():
        sky_mean_fake_D = (sky_mask[valid_mask] * fake[valid_mask]).sum(dim=[1, 2, 3]) / (
                3 * sky_mask[valid_mask].sum(dim=[1, 2, 3]) + 1e-6)
        sky_contour_min_fake_D = (sky_contour[valid_mask] * fake[valid_mask] + 1 - sky_contour[valid_mask]).min()
        sky_loss = torch.relu(sky_mean_fake_D.detach() - sky_contour_min_fake_D) * 0.5
        return sky_loss.mean()
    else:
        return 0.0


class SharpFusionLoss(torch.nn.Module):
    def __init__(self, lam_grad=7.0, lam_lap=4.5, lam_contrast=3.5, lam_freq=1.5):
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
        std = torch.sqrt((mean2 - mean * mean) ** 2 + 1e-6)
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
        size = I_f.shape[-1] // 256 * 2 + 1
        # Convert to grayscale if needed
        I_f = (0.299 * I_f[:, 0:1] + 0.587 * I_f[:, 1:2] + 0.114 * I_f[:, 2:3]) * 0.5 + 0.5 if I_f.shape[
                                                                                                   1] == 3 else I_f * 0.5 + 0.5
        I_vi = (0.299 * I_vi[:, 0:1] + 0.587 * I_vi[:, 1:2] + 0.114 * I_vi[:, 2:3]) * 0.5 + 0.5 if I_vi.shape[
                                                                                                       1] == 3 else I_vi * 0.5 + 0.5
        I_ir = (0.299 * I_ir[:, 0:1] + 0.587 * I_ir[:, 1:2] + 0.114 * I_ir[:, 2:3]) * 0.5 + 0.5 if I_ir.shape[
                                                                                                       1] == 3 else I_ir * 0.5 + 0.5
        # -------- Gradient Loss --------
        G_f = sobel(I_f).abs()
        G_vi = sobel(I_vi).abs()
        G_ir = sobel(I_ir).abs()
        G_ref = torch.max(G_vi, G_ir)
        L_grad = F.l1_loss(G_f, G_ref)

        # -------- Laplacian Loss --------
        L_f = laplacian(I_f, size).abs()
        L_vi = laplacian(I_vi, size).abs()
        L_ir = laplacian(I_ir, size).abs()
        L_ref = torch.max(L_vi, L_ir)
        L_lap = F.l1_loss(L_f, L_ref)

        # -------- Frequency Loss --------
        F_f = self.fft_high(I_f)
        F_vi = self.fft_high(I_vi)
        F_ir = self.fft_high(I_ir)
        F_ref = torch.max(torch.abs(F_vi), torch.abs(F_ir))
        L_freq = F.l1_loss(torch.abs(F_f), F_ref)

        # -------- Local Contrast Loss --------
        C_f = self.local_std(I_f)
        C_vi = self.local_std(I_vi)
        C_ir = self.local_std(I_ir)
        C_ref = torch.max(C_vi, C_ir)
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
        self.w_edges = 0.2

    # -------- Multiscale coherence --------
    def multiscale_loss(self, x):
        loss = 0
        for s in self.scales:
            ds = F.avg_pool2d(x, s)
            us = F.interpolate(ds, size=x.shape[-2:], mode='bilinear', align_corners=False)
            loss += torch.abs(x - us).mean()
        return loss / len(self.scales)

    # -------- Edge consistency loss --------
    def edge_consistency_loss(self, Fus, IR, N):
        edges_x = sobel(Fus).abs()
        edges_ref = sobel(IR).abs()
        edges_ref2 = sobel(N).abs() * dilation((N < 0.9).float(),
                                               kernel=torch.ones(3, 3, device=N.device))  # focus on dark areas
        return F.relu(edges_x - edges_ref * 0.9) / 2 + F.relu(edges_x - edges_ref2 * 0.9) / 2

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
    def charbonnier_tv(self, x, y, z):
        dx_x, dy_x = image_gradients(x)
        dx_y, dy_y = image_gradients(y)
        dx_z, dy_z = image_gradients(z)
        dx_x = dx_x * (dx_x > dx_y) * (dx_x > dx_z)
        dy_x = dy_x * (dy_x > dy_y) * (dy_x > dy_z)
        return torch.mean(torch.sqrt(dx_x ** 2 + dy_x ** 2 + self.eps ** 2))

    # -------- Frequency decay --------
    def frequency_decay_loss(self, x):
        B, C, H, W = x.shape
        X = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
        mag = torch.abs(X)

        R = torch.sqrt(create_meshgrid(H, W, device=x.device).pow(2).sum(dim=-1))
        # u = torch.linspace(-1, 1, H, device=x.device)
        # v = torch.linspace(-1, 1, W, device=x.device)
        # U, V = torch.meshgrid(u, v, indexing='ij')
        # R = torch.sqrt(U * U + V * V)

        return (mag * (R ** self.alpha)).mean()

    # -------- Final denoising loss --------
    def forward(self, I_fused, I_remapped, N):
        L = (self.w_ms * self.multiscale_loss(I_fused) +
             self.w_tensor * self.structure_tensor_loss(I_fused) +
             self.w_tv * self.charbonnier_tv(I_fused, I_remapped, N) +
             self.w_freq * self.frequency_decay_loss(I_fused) +
             self.w_edges * self.edge_consistency_loss(I_fused, I_remapped, N).mean())
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


def BiasCorrLoss(Seg_D, Seg_TN, fake_IR, real_vis, real_IR, rec_vis, real_edges, fake_gradmap):
    """
    Bias correction loss including artifact and color bias correction.

    Args:
        Seg_D:   (B,1,H_seg,W_seg) segmentation GT mask
        Seg_TN:  (B,1,H_seg,W_seg) segmentation mask from TN input
        fake_IR:    (B,3,H,W) fake IR image
        real_vis:   (B,3,H,W) real visible image
        real_IR:    (B,3,H,W) real IR image
        rec_vis:    (B,3,H,W) reconstructed visible image
        real_edges: (B,1,H,W) edge map from real visible
        fake_gradmap: (B,1,H,W) gradient map from fake IR
    Returns:
        total_loss: scalar
    """
    # fake_IR, fake_N = fake.split(3, dim=1)
    device = fake_IR.device
    B, _, H, W = fake_IR.shape
    GT_mask = F.interpolate(Seg_D.float(), size=(H, W), mode='nearest').detach()
    TN_mask = F.interpolate(Seg_TN.float(), size=(H, W), mode='nearest').detach()

    # Masks
    light_mask_ori = (GT_mask == TRAFFICLIGHT).float()
    veg_mask = (GT_mask == VEG).float()
    sign_mask = (GT_mask == SIGN).float()
    road_mask = (GT_mask == ROAD).float()
    SLight_mask_ori = (GT_mask == STREETLIGHT).float()
    vehicles_mask = ((GT_mask == MOTORCYCLE) | (GT_mask == BUS) | (GT_mask == CAR) | (GT_mask == TRAIN) | (
            GT_mask == TRUCK)).float()
    sky_mask = (GT_mask == SKY).float()

    # Normalize images
    fake_ir_norm = (fake_IR + 1.0) * 0.5
    # fake_ni_norm = (fake_N + 1.0) * 0.5
    real_vis_norm = (real_vis + 1.0) * 0.5

    # Grayscale
    real_gray = 0.299 * real_vis_norm[:, 0:1, :, :] + 0.587 * real_vis_norm[:, 1:2, :, :] + 0.114 * real_vis_norm[:,
                                                                                                    2:3, :, :]
    # fake_ir_gray = 0.299 * fake_ir_norm[:, 0:1, :, :] + 0.587 * fake_ir_norm[:, 1:2, :, :] + 0.114 * fake_ir_norm[:,
    #                                                                                                  2:3, :, :]
    fake_ir_gray = fake_ir_norm[:, 0:1]
    real_lab = rgb_to_lab(real_vis_norm)
    real_lab[:, :1, :, :] = real_lab[:, :1, :, :] / 100 + 1e-6  # [1e-6 1]
    real_lab[:, 1:, :, :] = real_lab[:, 1:, :, :] / 128  # [-1 1]

    # region Artifact Bias Correction
    max_pool = nn.MaxPool2d(3, stride=1, padding=1)
    SLight_mask = -max_pool(-SLight_mask_ori)  # morphological dilation
    SLight_area = SLight_mask_ori.sum(dim=[1, 2, 3])  # per batch

    SLight_loss = torch.zeros(B, device=device)
    SLight_valid = SLight_area > 25
    if SLight_valid.any():
        SLight_region = SLight_mask[SLight_valid] * real_gray[SLight_valid]
        mean_val = (SLight_region.sum(dim=[1, 2, 3]) / SLight_area[SLight_valid]).view(-1, 1, 1, 1)
        high_mask = (SLight_region > mean_val).float()
        fake_region_high = high_mask * fake_ir_gray[SLight_valid] + (1 - high_mask)

        veg_valid = (veg_mask.sum(dim=[1, 2, 3]) > 0) * SLight_valid
        if veg_valid.any():
            veg_region = veg_mask[veg_valid] * fake_ir_gray[veg_valid]
            veg_mean = (veg_region.sum(dim=[1, 2, 3]) / veg_mask[veg_valid].sum())
            SLight_loss[veg_valid] += F.relu(
                veg_mean.detach() + 0.25 - fake_region_high[veg_valid[SLight_valid]].flatten(1).min(1).values)
        else:
            SLight_loss[SLight_valid] += F.relu(0.7 - fake_region_high.flatten(1).min(1).values)
    # endregion

    # region Cloud Artifact Correction, force the temperature of the sky to be consistent with the input infrared image at the same height
    valid_veg = veg_mask.sum(dim=[1, 2, 3]) > 0
    sky_loss = torch.zeros(B, device=device)
    if valid_veg.any():
        veg_min = (veg_mask * fake_ir_gray)[valid_veg].sum(dim=[1, 2, 3]) / veg_mask.sum(dim=[1, 2, 3]).detach()  # (B,)
        sky_region = (sky_mask * fake_ir_gray)[valid_veg]
        # sky_day = (sky_mask * real_gray)[valid_veg]
        # sky_region_HL = sky_day > sky_day.sum(dim=[1, 2, 3])/(sky_mask.sum())*1.1 # (B,1,H,W)
        # valid_veg = valid_veg * (sky_region_HL.sum(dim=[1, 2, 3]) > 0)
        # if valid_veg.any():
        #     sky_loss[valid_veg] += (F.relu(veg_min * 1.1 - sky_region) * sky_region_HL).flatten(1).max(1).values
        # sky_mean_height_real_ir = (infrared_sky_region[valid_sky].sum(dim=[1, 3]) / (common_sky_mask[valid_sky].sum(dim=[1, 3]) + 1e-6))  # (B, H)
        # sky_mean_height_fake_ir = sky_region[valid_sky].sum(dim=[1, 3]) / (common_sky_mask[valid_sky].sum(dim=[1, 3]) + 1e-6)  # (B, H)
        # sky_loss[valid_sky] += F.relu(sky_mean_height_real_ir - sky_mean_height_fake_ir).sum(dim=1) / (common_sky_mask[valid_sky].sum(dim=[1, 3]) > 0).sum(1) * 0.2
        upper_sky_mask = sky_region[:, :, :H // 5]
        up_area = sky_mask[:, :, :H // 5].sum(dim=[1, 2, 3])
        mid_sky_mask = sky_region[:, :, H // 5:H // 4]
        mid_area = sky_mask[:, :, H // 5:H // 4].sum(dim=[1, 2, 3])
        lower_sky_mask = sky_region[:, :, H // 4:]
        lower_area = sky_mask[:, :, H // 4:].sum(dim=[1, 2, 3])
        gradient_horizon = torch.arange(H // 4 - H // 5, device=device).view(1, 1, H // 4 - H // 5, 1).repeat(1, 1, 1,
                                                                                                              W) / (
                                       H // 4 - H // 5) * veg_min
        gradient_horizon_region = gradient_horizon * mid_sky_mask
        if up_area:
            sky_loss[valid_veg] += F.relu(upper_sky_mask).sum(dim=[1, 2, 3]) / up_area * 0.1
        if mid_area:
            sky_loss[valid_veg] += F.relu((gradient_horizon_region * sky_mask[:, :, H // 5:H // 4] - mid_sky_mask)).sum(
                dim=[1, 2, 3]) / mid_area * 0.1
        if lower_area:
            sky_loss[valid_veg] += F.relu(veg_min * sky_mask[:, :, H // 4:] - lower_sky_mask).sum(
                dim=[1, 2, 3]) / lower_area * 0.1
    # endregion

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

    ABC_losses = SLight_loss.sum() + loss_sga_light.sum()  #+ TLight_loss.sum() * 5

    # ########## Color Bias Correction
    # Masks
    rec_losses = torch.zeros(B, device=device)
    for mask, threshold, weight in zip([sign_mask, SLight_mask_ori, vehicles_mask, road_mask], [50, 50, 50, 100],
                                       [2, 1, 0.1, 0.5]):
        valid_idx = mask.sum(dim=[1, 2, 3]) > threshold
        if valid_idx.any():
            rec_losses[valid_idx] += PixelConsistencyLoss(rec_vis[valid_idx], real_vis[valid_idx],
                                                          mask[valid_idx]) * weight

    CBC_losses = rec_losses.sum()

    ############ Thermal Channel equality loss
    # thermal_eq_loss = torch.max(torch.max(fake_IR, 1)[0] - torch.min(fake_IR, 1)[0])

    total_loss = ABC_losses + CBC_losses + sky_loss.sum() * 0.2 # + thermal_eq_loss
    return total_loss


def TrafLighLumiLoss_TN(N, T, TN, rec_T, real_D, fake_D, fake_T, mask, contour, weights, seg_mask):
    "Traffic Light Luminance Loss. fake_img: fake vis image. fake_mask: IR seg mask. real_mask: Vis seg mask."
    B, _, h, w = N.shape
    _, _, seg_h, seg_w = mask.shape
    if (h != seg_h) or (w != seg_w):
        mask = F.interpolate(mask.float(), size=[h, w], mode='nearest').long()
    N_norm = (N + 1.0) * 0.5
    T_norm = (T + 1.0) * 0.5
    N_gray = N_norm.max(dim=1)[0]
    T_gray = T_norm.mean(dim=1)
    losses = torch.zeros([B, ], device=N.device)
    labels = connected_components((mask + contour).float())
    for b in range(B):
        unique_labels, counts = labels.unique(return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == 0:
                continue
            mask_ = mask * (labels == label).float()
            contour_ = contour * (labels == label).float()
            total_ = mask_ + contour_
            weight_ = ((weights * mask_)[b:b + 1]).max()
            ys, xs = (mask_[0, 0]).nonzero().permute(1, 0)
            if len(ys) == 0 or len(xs) == 0:
                continue
            y0_mask, y1_mask, h_mask = ys.min(), ys.max(), ys.max() - ys.min() + 1
            x0_mask, x1_mask, w_mask = xs.min(), xs.max(), xs.max() - xs.min() + 1
            mask_lighted_area = torch.zeros_like(mask, device=mask.device)
            color = determine_color_N(N[b:b + 1, :, y0_mask:y1_mask, x0_mask:x1_mask] * 0.5 + 0.5)

            if color == 'green':
                # green light is usually at the bottom
                y0, y1 = y1_mask - h_mask // 2, y1_mask
                mask_lighted_area[b:b + 1, :, y0:y1, x0_mask:x1_mask] = 1.
            elif color == 'red':
                # red light is usually at the top
                y0, y1 = y0_mask, y0_mask + h_mask // 2
                mask_lighted_area[b:b + 1, :, y0:y1, x0_mask:x1_mask] = 1.
            else:
                # yellow light is usually in the middle
                y0, y1 = y0_mask + h_mask // 4, y0_mask + 2 * h_mask // 4
                mask_lighted_area[b:b + 1, :, y0:y1, x0_mask:x1_mask] = 1.
            mean_N_light_region = (N_gray * total_).sum() / (total_.sum() + 1e-6)
            mean_T_light_region = (T_gray * mask_lighted_area).sum() / (mask_lighted_area.sum() + 1e-6)
            HL_region_T = dilation((T_gray * mask_lighted_area > mean_T_light_region).float(),
                                   torch.ones(3, 3, device=mask.device))
            HL_region_N = (N_gray * total_ > mean_N_light_region).float()
            HL_region = HL_region_T * HL_region_N
            if HL_region.sum() == 0: # and HL_region_N.sum() > 0:
            #     HL_region = HL_region_N
            # elif HL_region_N.sum() == 0:
                continue
            else:
                pass
            radius_HL = torch.sqrt(HL_region.sum() / torch.pi).cpu().numpy()
            w_mask = w_mask.cpu().numpy()
            if 1 / 8 > radius_HL / w_mask:
                continue
            elif 1 / 8 <= radius_HL / w_mask < 1 / 4:
                HL_region = erosion(HL_region, get_disk_kernel(radius_HL // 4, device=mask.device))
                HL_region = dilation(HL_region, get_disk_kernel(radius_HL // 8, device=mask.device))
            elif 1 / 2 > radius_HL / w_mask >= 1 / 4:
                HL_region = erosion(HL_region, get_disk_kernel(radius_HL // 8, device=mask.device))
            else:
                HL_region = erosion(HL_region, get_disk_kernel(radius_HL // 4, device=mask.device))
            # losses fake TN composition
            sky_mask = F.interpolate((seg_mask[b:b + 1] == SKY).float(), size=(h, w), mode='nearest')
            T_adjusted = (T * 0.5 + 0.5) ** (mean_T_light_region / 0.5) * 2 - 1
            traffic_light_final = T_adjusted * total_ * (1 - sky_mask) * (1 - HL_region) - HL_region * N_gray
            TN_region = TN * mask_
            compo_loss = (PixelConsistencyLoss(TN_region[b:b + 1], traffic_light_final[b:b + 1],
                                               total_ * (1 - sky_mask)) * 0.5) * weight_
            # losses color consistency
            if color == 'red':
                target_color = torch.tensor([1.0, 0.0, 0.0], device=N.device).view(1, 3, 1, 1)
                color_fake_D = fake_D[:, 0:1] - fake_D[:, 2:3] / 2 - fake_D[:, 1:2] / 2  # [-2:2]

            elif color == 'green':
                target_color = torch.tensor([0.0, 1.0, 0.1], device=N.device).view(1, 3, 1, 1)
                color_fake_D = fake_D[:, 1:2] - fake_D[:, 0:1] * 9 / 10 - fake_D[:, 2:3] / 10  # [-2:2]
            else:
                target_color = torch.tensor([1.0, 1.0, 0.0], device=N.device).view(1, 3, 1, 1)
                color_fake_D = fake_D[:, :2].mean(1, keepdim=True) - fake_D[:, 2:3]  # [-2:2]

            luminosity_loss = torch.relu(
                2. - (color_fake_D * HL_region[b:b + 1] + 2 - 2 * HL_region[b:b + 1])).sum() / (
                                      HL_region[b:b + 1].sum() + 1e-6)
            color_dist = ImageTensor(fake_D[b:b + 1] * 0.5 + 0.5).color_distance(
                ImageTensor(target_color * torch.ones_like(fake_D, device=fake_D.device)))
            color_loss = ((color_dist * HL_region[b:b + 1]).sum() / (HL_region[b:b + 1].sum() + 1e-6) +
                          torch.relu((fake_D[b:b + 1] * 0.5 + 0.5 - target_color) * HL_region[b:b + 1]).sum() / (
                                  HL_region[b:b + 1].sum() + 1e-6))

            # losses rec D consistency
            rec_consistency_loss = PixelConsistencyLoss(rec_T[b:b + 1], T[b:b + 1],
                                                        total_[b:b + 1] * sky_mask[b:b + 1]) + \
                                   PixelConsistencyLoss(rec_T[b:b + 1], TN[b:b + 1], mask_[b:b + 1])
            HL_fake_T = (fake_T < fake_T[b:b + 1, :, y0_mask:y1_mask, x0_mask:x1_mask].mean()) * mask_
            HL_common = HL_region * HL_fake_T
            if HL_common.sum() > 0:
                rec_consistency_loss += PixelConsistencyLoss(fake_D[b:b + 1], real_D[b:b + 1], HL_common[b:b + 1])
            std_loss = (fake_D[b:b + 1] * (mask_ - HL_region)).std(1).mean()
            # grad_loss to enhance the gradient of traffic light region
            grad_TN = torch.abs(sobel(TN[b:b + 1].mean(1, keepdim=True))) * total_[b:b + 1]
            grad_fake_D = torch.abs(sobel(fake_D[b:b + 1].mean(1, keepdim=True))) * total_[b:b + 1]
            grad_loss = (nn.L1Loss()(grad_fake_D, grad_TN) * total_[b:b + 1]).sum() / (
                        total_[b:b + 1].sum() + 1e-6) * 2
            sky_contour = sky_mask * contour[b:b + 1]
            if sky_contour.sum() > 0:
                sky_mean_fake_D = (sky_mask * fake_D[b:b + 1]).sum() / (3 * sky_mask.sum() + 1e-6)
                sky_contour_min_fake_D = (sky_contour * fake_D[b:b + 1] + 1 - sky_contour).min()
                sky_loss = torch.relu(sky_mean_fake_D.detach() - sky_contour_min_fake_D) * 0.8
            else:
                sky_loss = 0.

            losses[b] += (compo_loss + color_loss + luminosity_loss + grad_loss +
                          rec_consistency_loss + std_loss + sky_loss) * weight_
    return losses


class IlluminationAwareFusionLoss(nn.Module):
    """
    Combined loss for unpaired illumination-aware fusion.

    Inputs:
        I     : Illumination map        (B,1,H,W) in [0,1]
        R     : Reflectance map         (B,1,H,W) in [-1,1]
        T    : Original IR image       (B,3,H,W) or (B,1,H,W) in [-1,1]
        mask  : Highlight mask          (B,1,H,W) in [0,1]

    Returns:
        total_loss, dict_of_components
    """

    def __init__(
        self,
        lambda_structure=1.0,
        lambda_smooth=1.0,
        lambda_highlight=5.,
        lambda_gamma=0.2
    ):
        super().__init__()
        self.lambda_structure = lambda_structure
        self.lambda_smooth = lambda_smooth
        self.lambda_highlight = lambda_highlight
        self.lambda_gamma = lambda_gamma

    # ---------------------------------------------------------
    # Gradient utility
    # ---------------------------------------------------------

    @staticmethod
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    # ---------------------------------------------------------
    # Loss components
    # ---------------------------------------------------------

    def illumination_smoothness(self, I):
        dx, dy = self.gradient(I)
        return (dx ** 2).mean() + (dy ** 2).mean()

    def highlight_suppression(self, I, T, TN, R, mask):
        # penalize illumination in highlight regions
        loss_high_light = (torch.relu(mask * (R - I))).mean()
        loss_struct = (torch.relu(T * I - TN**2) * mask).mean()
        return loss_high_light + loss_struct

    def correlation_I_N_gamma(self, I, N, mask):
        C = (N.max(1, keepdim=True)[0] - N.min(1, keepdim=True)[0])
        corr = ((I * C) * (1 - mask)).sum(dim=[1, 2, 3]) / torch.sqrt((I**2 * (1-mask)).sum(dim=[1, 2, 3]) * (C**2 * (1-mask)).sum(dim=[1, 2, 3]) + 1e-6)
        return 1 - corr.mean()

    def structure_consistency(self, R, T, mask):
        # ensure T is single channel
        if T.shape[1] == 3:
            T = T.mean(dim=1, keepdim=True)
        dx_r, dy_r = self.gradient(R)
        dx_T, dy_T = self.gradient(T)
        loss_struct = ((torch.relu(dx_T - dx_r))*(mask[..., 1:]+1)).mean() + ((torch.relu(dy_T - dy_r))*(mask[..., 1:, :]+1)).mean() * 0.9
        return loss_struct

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, I, R, mask, T, N, TN):

        L_smooth = self.illumination_smoothness(I)
        L_highlight = self.highlight_suppression(I, T, TN, R, mask)
        L_structure = self.structure_consistency(R, T, mask)
        L_gamma = self.correlation_I_N_gamma(I, N, mask)

        return (
                self.lambda_structure * L_structure +
                self.lambda_smooth * L_smooth +
                self.lambda_highlight * L_highlight +
                self.lambda_gamma * L_gamma
               )


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

    Light_mask_real = (real_mask == STREETLIGHT).float()
    Sign_mask_real = (real_mask == SIGN).float()
    Person_mask_real = (real_mask == PERSON).float()
    Vehicle_mask_real = ((real_mask > STREETLIGHT) & (real_mask < MOTORCYCLE)).float()
    Motor_mask_real = (real_mask == MOTORCYCLE).float()

    Light_mask_fake = (fake_mask == STREETLIGHT).float()
    Sign_mask_fake = (fake_mask == SIGN).float()
    Person_mask_fake = (fake_mask == PERSON).float()
    Vehicle_mask_fake = ((fake_mask > STREETLIGHT) & (fake_mask < MOTORCYCLE)).float()
    Motor_mask_fake = (fake_mask == MOTORCYCLE).float()

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
    fake_nonzero = fake_flat.abs().sum(dim=1) > 0
    fake_flat = fake_flat[fake_nonzero]

    if real_flat.size(0) < cluster_num or fake_flat.size(0) < cluster_num:
        return 0.0

    # Normalize pixel features
    real_norm = F.normalize(real_flat, p=2, dim=1)
    fake_norm = F.normalize(fake_flat, p=2, dim=1)

    # ---- Cluster centers from real features ----
    centers = GetFeaMatrixCenter(real_norm, cluster_num, max_iter)  # (K, C)
    centers_norm = F.normalize(centers, p=2, dim=1)

    # ---- Real similarity ----
    sim_real = real_norm @ centers_norm.T  # (N_real, K)
    sim_real_max = sim_real.max(dim=1).values.mean()  # mean over pixels
    sim_real_cluster = sim_real.max(dim=0).values.mean()  # mean over clusters

    sim_fake = fake_norm @ centers_norm.T  # (N_fake, K)
    sim_fake_max = sim_fake.max(dim=1).values.mean()  # mean over pixels
    sim_fake_cluster = sim_fake.max(dim=0).values.mean()  # mean over clusters
    # ---- Loss terms ----
    loss_sim = F.relu(0.9 * sim_real_max.detach() - sim_fake_max)
    loss_div = F.relu(0.9 * sim_real_cluster.detach() - sim_fake_cluster)

    return loss_sim + loss_div


def PixelConsistencyLoss(inputs_img, GT_img, ROI_mask):
    "Pixel-wise Consistency Loss. inputs_img and GT_img are 4D tensors range [-1, 1], while ROI_mask is a 2D tensor."
    input_masked = inputs_img * ROI_mask
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
