import functools
from dataclasses import dataclass
from typing import Literal, List

import numpy as np
from kmeans_pytorch import kmeans
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.contrib import connected_components
from kornia.morphology import closing, dilation, opening, erosion
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.morphology import disk
from skimage.util import random_noise
from torch import nn, Tensor
from torch.nn.functional import conv2d
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms.v2 import GaussianBlur

ROAD = 0
PAVEMENT = 1
BUILDING = 2
CLOUD = 3
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


# region ------------------------ Utilities ---------------------------

def rgb_to_ycbcr(x):
    """Convert a batch of RGB images [0,1] to YCbCr approx. Returns tensor same shape.
    Formula (BT.601):
        Y  =  0.299 R + 0.587 G + 0.114 B
        Cb = -0.168736 R -0.331264 G +0.5 B
        Cr =  0.5 R -0.418688 G -0.081312 B
    Input: x (B,3,H,W) in range [-1,1] or [0,1]. We assume [-1,1] and map to [0,1].
    """
    if x.min() < -0.5:
        x = (x + 1.0) / 2.0
    R = x[:, 0:1]
    G = x[:, 1:2]
    B = x[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
    return torch.cat([Y, Cb, Cr], dim=1)


def sobel_gradients(x):
    """Compute image gradients using Sobel filters. x in shape (B,C,H,W).
    Returns gradient magnitude per channel aggregated as (B,1,H,W)"""
    b, c, h, w = x.shape
    device = x.device
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    grads = []
    for ch in range(c):
        xi = x[:, ch:ch + 1]
        grad_x = conv2d(xi, gx, padding=1)
        grad_y = conv2d(xi, gy, padding=1)
        grads.append(torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6))
    grads = torch.cat(grads, dim=1)
    # optionally aggregate channels by mean
    return grads.mean(dim=1, keepdim=True)


def gkern_2d(size=5, sigma=3, nchannels=3):
    # Create 2D gaussian kernel
    dirac = np.zeros((size, size))
    dirac[size // 2, size // 2] = 1
    mask = gaussian_filter(dirac, sigma)
    # Adjust dimensions for torch conv2d
    return np.stack([np.expand_dims(mask, axis=0)] * nchannels)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        return functools.partial(nn.GroupNorm, num_groups=32, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def GetFeaMatrixCenter(fea_array, cluster_num, max_iter):
    """
    Compute K cluster centers from normalized feature vectors.
    fea_array: (N, C)
    """
    _, centers = kmeans(
        X=fea_array,
        num_clusters=cluster_num,
        distance='cosine',
        device=fea_array.device,
        iter_limit=max_iter,
        tqdm_flag=False
    )
    return centers.to(fea_array.device)


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


def RefineIRMask(ori_mask, input_IR):
    """
    Refine segmentation mask using IR image for specific categories:
    Sky, Vegetation, Pole, Person.
    Args:
        ori_mask: (B,H,W), integer labels
        input_IR: (B,3,H,W), float IR image
    Returns:
        mask_refine: (B,H,W), refined mask with uncertain areas marked as 255
    """
    device = input_IR.device
    B, _, H, W = input_IR.shape

    # Normalize IR to [0,1]
    x_min = input_IR.view(B, -1).min(dim=1).values.view(B, 1, 1, 1)
    x_max = input_IR.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
    x_norm = (input_IR - x_min) / (x_max - x_min + 1e-6)

    # Grayscale conversion
    IR_gray = 0.299 * x_norm[:, 0:1, :, :] + 0.587 * x_norm[:, 1:2, :, :] + 0.114 * x_norm[:, 2:3, :, :]
    IR_gray = IR_gray.squeeze(1)  # (B,H,W)

    # Category masks
    categories = {"Pole": 5, "Veg": 8, "Sky": 10, "Person": 11}
    masks = {k: (ori_mask == v).float() for k, v in categories.items()}

    # Region mean and intradis per category
    region_mean = {}
    intradis = {}
    for k in categories:
        cnt = masks[k].sum(dim=(1, 2))  # (B,)
        region = masks[k] * IR_gray
        region_mean[k] = torch.where(cnt > 0, region.view(B, -1).sum(dim=1) / cnt, 0.0)  # (B,)
        intradis[k] = masks[k] * (region - region_mean[k].view(B, 1, 1)) ** 2

    # Sky denoising
    cnt_Sky = masks["Sky"].sum(dim=(1, 2))
    cnt_Veg = masks["Veg"].sum(dim=(1, 2))

    valid = (cnt_Sky * cnt_Veg) > 0
    if valid.any():
        Sky_Veg_dis_err = intradis["Sky"] - masks["Sky"] * (IR_gray - region_mean["Veg"].view(B, 1, 1)) ** 2
        Sky2Veg_mask = (Sky_Veg_dis_err > 0).float()
        mask_Sky_refine = Sky2Veg_mask * 255.0 + (masks["Sky"] - Sky2Veg_mask) * 10.0
        # Update Sky mean after refinement
        new_Sky_mask = masks["Sky"] - Sky2Veg_mask
        cnt_Sky_new = new_Sky_mask.sum(dim=(1, 2))
        region_Sky_new = new_Sky_mask * IR_gray
        Sky_region_mean_new = torch.where(cnt_Sky_new > 0, region_Sky_new.view(B, -1).sum(dim=1) / cnt_Sky_new,
                                          region_mean["Sky"])
    else:
        mask_Sky_refine = masks["Sky"] * 10.0
        Sky_region_mean_new = region_mean["Sky"]

    # Pole denoising
    cnt_Pole = masks["Pole"].sum(dim=(1, 2))
    valid = (cnt_Pole * cnt_Sky) > 0
    if valid.any():
        Pole_Sky_dis_err = intradis["Pole"] - masks["Pole"] * (IR_gray - Sky_region_mean_new.view(B, 1, 1)) ** 2
        Pole2Sky_mask = (Pole_Sky_dis_err > 0).float()
        mask_Pole_refine = Pole2Sky_mask * 255.0 + (masks["Pole"] - Pole2Sky_mask) * 5.0
    else:
        mask_Pole_refine = masks["Pole"] * 5.0

    # Person denoising
    cnt_Person = masks["Person"].sum(dim=(1, 2))
    valid = (cnt_Person * cnt_Veg) > 0
    if valid.any():
        Person_Veg_dis_err = intradis["Person"] - masks["Person"] * (IR_gray - region_mean["Veg"].view(B, 1, 1)) ** 2
        Person2Veg_mask = (Person_Veg_dis_err > 0).float()
        mask_Person_refine = Person2Veg_mask * 255.0 + (masks["Person"] - Person2Veg_mask) * 11.0
    else:
        mask_Person_refine = masks["Person"] * 11.0

    # Vegetation denoising
    fuse_uncer = torch.zeros_like(ori_mask, dtype=torch.float32, device=device)
    if (cnt_Veg * cnt_Sky * cnt_Person).any():
        Veg2Sky_mask = (intradis["Veg"] - masks["Veg"] * (IR_gray - Sky_region_mean_new.view(B, 1, 1)) ** 2 > 0).float()
        Veg2Person_mask = (
                intradis["Veg"] - masks["Veg"] * (IR_gray - region_mean["Person"].view(B, 1, 1)) ** 2 > 0).float()
        fuse_uncer = Veg2Sky_mask + Veg2Person_mask
        uncertain_mask_veg = (fuse_uncer > 0).float()
        mask_Veg_refine = uncertain_mask_veg * 255.0 + (masks["Veg"] - uncertain_mask_veg) * 8.0
    else:
        mask_Veg_refine = masks["Veg"] * 8.0

    # Combine all masks, keep other labels
    all_mask = masks["Pole"] + masks["Veg"] + masks["Sky"] + masks["Person"]
    mask_refine = mask_Sky_refine + mask_Pole_refine + mask_Person_refine + mask_Veg_refine + (1 - all_mask) * ori_mask

    return mask_refine.detach()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias'):
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        if hasattr(m, 'conv'):
            weights_init(m.conv)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Spectral normalization base class
# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# endregion -----------------------------


# region --------------------Losses Utilities -------------------------
def ClsMeanPixelValue(input_tensor, SegMask, num_class, exclude_classes=None):
    """ Compute mean feature vector for each category in the segmentation mask.
    Args: input_tensor: (B, C, H, W)
    feature tensor SegMask: (B, 1, H_seg, W_seg)
    segmentation mask num_class: int,
    number of classes exclude_classes: list[int],
    classes to ignore (optional)
    Returns: out_tensor: (1, num_class, C)
    mean feature per class out_cls_tensor: (1, num_class, 1), 1 if class exists in batch
    out_cls_ratio_tensor: (1, num_class, 1), ratio of pixels for that class """
    device = input_tensor.device
    B, C, H, W = input_tensor.shape
    _, _, H_seg, W_seg = SegMask.shape  # Resize mask to match feature size
    mask = F.interpolate(SegMask.float(), size=(H, W), mode='nearest')  # (B,1,H,W) # Flatten spatial dimensions
    mask_flat = mask.view(B, 1, H * W)  # (B,1,N)
    feat_flat = input_tensor.view(B, C, H * W)  # (B,C,N)
    out_tensor = torch.zeros(B, num_class, C, device=device)
    out_cls_tensor = torch.zeros(B, num_class, 1, device=device)
    out_cls_ratio_tensor = torch.zeros(B, num_class, 1, device=device)
    exclude_classes = exclude_classes or []
    for i in range(num_class):
        if i in exclude_classes:
            continue  # Binary mask for class i
        class_mask = (mask_flat == i).float()  # (B,1,N)
        class_count = class_mask.sum(dim=2).squeeze(1)  # (B,)
        total_count = H * W  # Only compute if class exists in at least one batch element
        if (class_count > 0).any():
            out_cls_tensor[class_count > 0, i] = 1.0
            out_cls_ratio_tensor[class_count > 0, i] = class_count.sum() / (B * total_count)  # Compute mean feature
            masked_feat = feat_flat * class_mask  # broadcast multiply
            class_sum = masked_feat.sum(dim=2)  # sum over pixels (B,C)
            class_mean = class_sum.sum(dim=0) / class_count.sum()  # sum over batch then divide by total pixels
            out_tensor[class_count > 0, i] = class_mean
    return out_tensor, out_cls_tensor, out_cls_ratio_tensor


def RefineLightMask(Seg_mask, real_vis):
    """Denoising of the traffic light mask region."""
    Seg_mask = Seg_mask.clone()
    if ((Seg_mask == TRAFFICLIGHT).sum(dim=[1, 2, 3]) > 50).any():
        Seg_mask = LightMaskDenoised(Seg_mask, real_vis, 5)
        Seg_mask = LightMaskDenoised(Seg_mask, real_vis, 3)
    return Seg_mask == TRAFFICLIGHT


def LightMaskDenoised(Seg_mask, real_vis, Avg_KernelSize):
    """
    Fully batched denoising of traffic light masks.

    Args:
        Seg_mask: (B,1,H,W) segmentation mask
        real_vis: (B,3,H,W) real visible images
        Avg_KernelSize: int, kernel size for local averaging
        min_area_ratio: minimum area of small holes to fill relative to mask area

    Returns:
        out_mask: (B, H,W) denoised mask
    """
    B, _, H, W = real_vis.shape

    # Original masks
    Seg_mask = Seg_mask.squeeze(1)  # (B,H,W)
    light_mask_ori = (Seg_mask == TRAFFICLIGHT).float()
    sky_mask = (Seg_mask == SKY).float()

    # Grayscale normalized
    real_gray = ((real_vis + 1.0) * 0.5).mean(dim=1)  # (B,1,H,W)

    # Local average pooling
    padsize = Avg_KernelSize // 2
    local_mean = F.avg_pool2d(light_mask_ori * real_gray, Avg_KernelSize,
                              stride=1, padding=padsize)

    # Sky mean per batch
    sky_sum = sky_mask.sum(dim=[1, 2], keepdim=True)
    sky_mean = (real_gray * sky_mask).sum(dim=[1, 2], keepdim=True) / (sky_sum + 1e-6)
    sky_mean = sky_mean.view(B, 1, 1)

    # Distances
    light_gray = light_mask_ori * real_gray  # (B,H,W)
    dist_sky = light_mask_ori * (light_gray - sky_mean) ** 2
    dist_local = light_mask_ori * (light_gray - local_mean) ** 2
    sky_diff = dist_local - dist_sky
    sky_noise = (sky_diff > 0).float() * light_mask_ori

    # Denoised mask
    light_mask_denoised = F.relu(light_mask_ori - sky_noise)

    # Small-hole filling (vectorized)
    light_mask_denoised = fill_holes(light_mask_denoised.unsqueeze(1))  # (B,1,H,W)
    # area_th = light_mask_ori.sum(dim=[1, 2]) - light_mask_denoised.sum(dim=[1, 2])  # (B,)
    # th = max(1, area_th.cpu()//2+1).numpy()
    # # # Invert mask
    # kernel = torch.tensor(disk(th), device=light_mask_ori.device).float()
    # hole = closing(light_mask_denoised.unsqueeze(1), kernel) - light_mask_denoised.unsqueeze(1)  # (B,1,H,W)
    # hole = opening(hole, torch.ones(3, 3, device=light_mask_ori.device))  # Remove noise
    # #
    # light_mask_denoised = closing(light_mask_denoised.unsqueeze(1) + hole, torch.ones(3, 3, device=light_mask_ori.device))
    # Construct final mask
    out_mask = ((1 - light_mask_ori) * Seg_mask + 6.0 * light_mask_denoised +
                255.0 * (light_mask_ori - light_mask_denoised))

    return out_mask  # (B,1,H,W)


def fill_holes(mask: Tensor, max_iters=200):
    """
    mask: (B, 1, H, W), binary {0,1}, on GPU
    """
    # Invert mask
    inv = 1.0 - mask

    # Marker = background connected to borders
    marker = torch.zeros_like(inv)
    marker[..., 0, :]  = inv[..., 0, :]
    marker[..., -1, :] = inv[..., -1, :]
    marker[..., :, 0]  = inv[..., :, 0]
    marker[..., :, -1] = inv[..., :, -1]

    kernel = torch.ones((3, 3), device=mask.device)

    # Morphological reconstruction by dilation
    for _ in range(max_iters):
        new_marker = dilation(marker, kernel)
        new_marker = torch.minimum(new_marker, inv)
        if torch.equal(new_marker, marker):
            break
        marker = new_marker

    # Holes are what's not connected to border
    filled = 1.0 - marker
    return filled


def create_fake_TLight(img, img_fake, mask_p):
    TLight_region = mask_p.mul(img)
    fake_TLight_region = mask_p.mul(img_fake)
    img_processed = TLight_region ** 7
    m = TLight_region.std(dim=1, keepdim=True) > (
            (TLight_region > 0) * TLight_region.std(dim=1, keepdim=True)).sum() / (
                (TLight_region > 0).sum() + 1e-6)
    img_processed = img_processed * m.expand_as(img_processed)
    padsize = 5 // 2
    MaxPool_k5 = nn.MaxPool2d(5, stride=1, padding=padsize)
    for i in range(1):
        img_processed = MaxPool_k5(img_processed)
        img_processed = gaussian_blur(img_processed / (img_processed.max() + 1e-14), (5, 5), (1.6, 1.6))
    img_processed = (img_processed / (img_processed.max() + 1e-14) + TLight_region * 0.1).clamp(0, 1)
    fake = torch.zeros_like(img_processed).to(img.device)
    label_connect, num = measure.label((img_processed.mean(dim=1) > img_processed.mean() + img_processed.std()).cpu(),
                                       connectivity=2, background=0, return_num=True)
    for j in range(1, num + 1):
        "Since background index is 0, the num is num+1."
        temp_connect_mask = torch.where(torch.from_numpy(label_connect) == j, 1.0, 0.0).to(img.device)
        light_i_ = temp_connect_mask.expand_as(img_processed) * img_processed
        fake_TLight_region_i = temp_connect_mask.expand_as(img_processed) * fake_TLight_region
        patch_mean = light_i_[0].flatten(1)[:, light_i_[0].flatten(1).mean(dim=0) > 0].mean(dim=1)
        patch_overlap = gaussian_blur(temp_connect_mask.expand_as(img_processed), (11, 11), (7., 7.))
        patch_overlap /= patch_overlap.max()
        if patch_mean[0] - 1.5 * patch_mean[2] > 0:  # if red
            light_i = patch_overlap * light_i_ * 3
            light_i = light_i.clamp(int(fake_TLight_region_i.mean(dim=1).min().cpu()), 1)
        elif patch_mean[2] - 1.5 * patch_mean[0] > 0:  # if green
            light_i = patch_overlap * light_i_ * 3
            light_i = light_i.clamp(int(fake_TLight_region_i.mean(dim=1).min().cpu()), 1)
        else:
            light_i = 0
        fake += light_i
    fake = fake / (fake.max() + 1e-6)
    return fake


def center_of_mass(img):
    # img: B×1×H×W
    B, _, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=img.device),
        torch.linspace(0, W - 1, W, device=img.device),
        indexing='ij'
    )
    img_sum = img.sum(dim=[2, 3]) + 1e-6
    cx = (img * grid_x).sum(dim=[2, 3]) / img_sum
    cy = (img * grid_y).sum(dim=[2, 3]) / img_sum
    return cx, cy


def detect_TL_colorblobs_mask_free(I_vi, I_ir):
    # ---- Luminance ----
    colors = {0: torch.tensor([1, 0.1, 0], device=I_vi.device),
              1: torch.tensor([0., 1, 0.6], device=I_vi.device),
              2: torch.tensor([1, 0.5, 0], device=I_vi.device)}
    scale = I_ir.shape[-2] / 256
    vi_squared = I_vi ** 2
    R = 1.25 * vi_squared[:, 0:1] - 1. * vi_squared[:, 1:2] - 0.5 * vi_squared[:, 2:3]
    G = 0.75 * vi_squared[:, 1:2] - 1.75 * vi_squared[:, 0:1] + 0.75 * vi_squared[:, 2:3]
    O = 1.1 * vi_squared[:, 0:1] - 0.1 * vi_squared[:, 1:2] - 2. * vi_squared[:, 2:3]
    C_intensity, color_idx = torch.max(torch.cat([R, G, O], dim=1), dim=1, keepdim=True)
    C_intensity = C_intensity * (C_intensity > 0.1)

    # C_intensity = I_vi * C_intensity
    Y = I_vi.mean(1, keepdim=True) * (C_intensity == 0)
    criterion = C_intensity.mean() + C_intensity.std() * 2 + 0.01
    # ---- Blobs mapping ----
    #  case where the center of the blob is saturated
    M = (fill_holes((Y==0).float()) - (Y == 0).float()) * (Y > Y[Y>0].mean())
    # M = (Y * I_ir.mean(1, keepdim=True) > min((Y.mean(), 0.80))).float()
    M_color = I_vi * 0
    labels = connected_components(M)
    # ---- Saturation enclosure ----
    for B, label in enumerate(labels):
        label = label.unsqueeze(0)
        uniques = label.unique(return_counts=True)
        for i, (uni, count) in enumerate(zip(*uniques)):
            mask = (label == uni).float()
            if uni == 0:
                continue
            elif count < 10 or count > 500 * scale:
                M = M - mask
                continue
            size = min(int(torch.sqrt(count / np.pi / scale).cpu().numpy()), 11) * 2 + 1
            kernel_ring = get_disk_kernel(size, I_vi.device)
            kernel_ring_small = get_disk_kernel(max(size//4, 1), I_vi.device)
            mask_ = dilation(mask, kernel=kernel_ring_small)
            surrounding = dilation(mask_, kernel=kernel_ring) - mask_
            mean_sat = ((C_intensity[B][None] * surrounding).sum()) / (surrounding.sum() + 1e-6)
            if mean_sat < criterion:
                M = M - mask
            else:
                # only keep the disk shaped blobs
                cx, cy = center_of_mass(mask)
                radius = torch.sqrt(count / np.pi)
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(0, M.shape[-2] - 1, M.shape[-2], device=I_vi.device),
                    torch.linspace(0, M.shape[-1] - 1, M.shape[-1], device=I_vi.device),
                    indexing='ij'
                )
                dist_map = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                disk_mask = dist_map <= radius * 1.5
                if (mask - mask * disk_mask).sum() != 0:
                    continue  # not disk enough
                surroundings_mask = (C_intensity[B][None] * surrounding) > 0
                color_idx_blob = torch.bincount((surroundings_mask[surroundings_mask] * color_idx[B][None][surroundings_mask]).to(torch.int)).argmax()
                color = colors[int(color_idx_blob)]
                M_color[B, :, int(cy), int(cx)] = count * color * 1.4142

    color_blur = GaussianBlur(kernel_size=25, sigma=1.6)(M_color).clamp(0, 1)
    return M * color_blur


def detect_TL_blobs_mask_free(I_vi):
    # ---- Luminance ----
    scale = I_vi.shape[-2] / 256
    vi_squared = I_vi ** 2
    R = 1.25 * vi_squared[:, 0:1] - 1. * vi_squared[:, 1:2] - 0.5 * vi_squared[:, 2:3]
    G = 0.75 * vi_squared[:, 1:2] - 1.75 * vi_squared[:, 0:1] + 0.75 * vi_squared[:, 2:3]
    O = 1.1 * vi_squared[:, 0:1] - 0.1 * vi_squared[:, 1:2] - 2. * vi_squared[:, 2:3]
    C_intensity, color_idx = torch.max(torch.cat([R, G, O], dim=1), dim=1, keepdim=True)
    Y = I_vi.mean(1, keepdim=True) * (C_intensity <= 0.0)
    # ---- Blobs mapping ----
    M = (Y > Y[Y > 0].mean() + Y[Y > 0].std()).float()
    M = opening(M, get_disk_kernel(1, I_vi.device))
    M = dilation(M, get_disk_kernel(1, I_vi.device))
    labels = connected_components(M)
    # ---- Saturation enclosure ----
    for B, label in enumerate(labels):
        label = label.unsqueeze(0)
        uniques = label.unique(return_counts=True)
        for i, (uni, count) in enumerate(zip(*uniques)):
            mask = (label == uni).float()
            if uni == 0:
                continue
            elif count < 10 or count > 500 * scale:
                M = M - mask
                continue
            # only keep the disk shaped blobs
            cx, cy = center_of_mass(mask)
            radius = torch.sqrt(count / np.pi)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, M.shape[-2] - 1, M.shape[-2], device=I_vi.device),
                torch.linspace(0, M.shape[-1] - 1, M.shape[-1], device=I_vi.device),
                indexing='ij'
            )
            dist_map = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            disk_mask = dist_map <= radius * 1.5
            if (mask - mask * disk_mask).sum() != 0:
                M = M - mask
    return M


def create_fake_Light(img, mask_p):
    fake = torch.zeros_like(img).to(img.device)
    b, c, h_, w_ = fake.shape
    img_processed = []
    for i in range(b):
        mas_p_i = mask_p[i].squeeze(0).cpu()
        label_connect, num = measure.label(mas_p_i, connectivity=2, background=0, return_num=True)
        for j in range(1, num + 1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = torch.where(torch.from_numpy(label_connect) == j, 1.0, 0.0).to(img.device)
            h, w = temp_connect_mask.sum(dim=-2).max() + 1e-14, temp_connect_mask.sum(dim=-1).max()
            kernel_size = max(int(h * 2 + 1), 5), max(int(w * 2 + 1), 5)
            sigma = torch.tensor([min(h / 2, kernel_size[0] / 3)]).to(img.device), torch.tensor(
                [min(w / 2, kernel_size[1] / 3)]).to(img.device)
            if w / h > 1.75:
                # Horizontal white streetlight from the top
                Light_region = mask_p.mul(
                    torch.Tensor([1., 0.9, 0.85])[None, :, None, None].expand_as(img).to(mask_p.device))
                #drawn a bit lower
                temp = torch.zeros([1, c, h_ + 3, w_]).to(img.device)
                temp[:, :, 3:] = Light_region
                Light_region = temp[:, :, :-3]
                fake += gaussian_blur(Light_region, kernel_size, (1.6, 2))
            else:
                color = [1., 0.7, 0.05] if torch.rand(1) > 0.5 else [1., 0.95, 0.95]
                Light_region = mask_p.mul(
                    torch.Tensor(color)[None, :, None, None].expand_as(img).to(mask_p.device))
                fake += gaussian_blur(Light_region, kernel_size, sigma)
        img_processed.append((fake / fake.max() + img * mask_p).clamp(0, 1))
    return torch.cat(img_processed)


def get_disk_kernel(radius, device):
    return torch.from_numpy(disk(radius)).to(device=device).float()


# endregion ----------------------------

# region -------------------- SegMask Update Utilities ----------------


def UpdateIRGTv1(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th=0.9):
    """
    Online semantic distillation module (batched, GPU-optimized)
    Args:
        seg_tensor1: (B,C,H,W) logits for real IR
        seg_tensor2: (B,C,H,W) logits for fake RGB
        ori_seg_GT:  (B,1,H,W) original GT
        input_IR:    (B,3,H,W)
        prob_th: threshold for high confidence
    Returns:
        mask_CurtVeg: (B,1,H,W), refined GT
    """
    # Softmax and max predictions
    ori_seg_GT = ori_seg_GT.squeeze(1)
    pred_max_val1, pred_max_cat1 = F.softmax(seg_tensor1.detach(), dim=1).max(dim=1)
    pred_max_val2, pred_max_cat2 = F.softmax(seg_tensor2.detach(), dim=1).max(dim=1)

    mask_inter = (pred_max_cat1 == pred_max_cat2).float()
    mask_inter_HP = mask_inter * (pred_max_val1 > prob_th) * (pred_max_val2 > prob_th)
    seg_inter_mask_UC = mask_inter_HP * (ori_seg_GT == 255).float()

    mask_new_GT = seg_inter_mask_UC * pred_max_cat1.float() + (1 - seg_inter_mask_UC) * ori_seg_GT.float()

    # Refine with IR
    mask_final = RefineIRMask(mask_new_GT, input_IR)
    # Veg/Road LP mask
    mask_Bkg_all = (mask_final < 11).float()
    mask_Build_new = (mask_final == 2).float()
    mask_Sign_new = (mask_final == 6).float()
    mask_Light_new = (mask_final == 7).float()
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new
    # Adaptive threshold
    High_th = prob_th + 0.04 if (pred_max_cat1 - pred_max_cat2).float().mean() == 0 else prob_th
    LHP_mask = (pred_max_val1 < High_th).float()
    VegRoad_LP_mask = LHP_mask * mask_Bkg_stuff
    mask_CurtVeg = (1 - VegRoad_LP_mask) * mask_final + VegRoad_LP_mask * 255.0
    return mask_CurtVeg.unsqueeze(1).detach()


def UpdateIRGTv2(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th=0.8):
    """
    Update NTIR segmentation pseudo-labels using online semantic distillation
    and IR image refinement.

    Args:
        seg_tensor1: (B,C,H,W) logits for real IR
        seg_tensor2: (B,C,H,W) logits for fake RGB
        ori_seg_GT:  (B,1,H,W) original GT, 255 = uncertain
        input_IR:    (B,3,H,W) IR image
        prob_th: high-confidence threshold
    Returns:
        out_mask: (B,1,H,W), updated pseudo-labels
    """
    ori_seg_GT = ori_seg_GT.squeeze(1)
    # Softmax and max predictions
    pred_sm1 = F.softmax(seg_tensor1.detach(), dim=1)
    pred_sm2 = F.softmax(seg_tensor2.detach(), dim=1)
    pred_max_val1, pred_max_cat1 = pred_sm1.max(dim=1)
    pred_max_val2, pred_max_cat2 = pred_sm2.max(dim=1)

    # Mask agreement
    mask_inter = (pred_max_cat1 == pred_max_cat2).float()
    mask_inter_HP = mask_inter * (pred_max_val1 > prob_th).float() * (pred_max_val2 > prob_th).float()

    # Update high-confidence uncertain pixels
    mask_new_GT = mask_inter_HP * pred_max_cat1.float() + (1 - mask_inter_HP) * 255.0
    mask_final = RefineIRMask(mask_new_GT, input_IR)

    # Remove veg/stuff areas from supervision
    mask_Bkg_all = (mask_final < 11).float()
    mask_Build_new = (mask_final == 2).float()
    mask_Sign_new = (mask_final == 6).float()
    mask_Light_new = (mask_final == 7).float()
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new

    # Adaptive threshold
    High_th = prob_th if (pred_max_cat1 - pred_max_cat2).float().mean() == 0 else prob_th + 0.04
    LHP_mask = (pred_max_val1 < High_th).float()
    VegRoad_LP_mask = LHP_mask * mask_Bkg_stuff

    # Confusing categories mask
    mask_CurtVeg = (1 - VegRoad_LP_mask) * mask_final + VegRoad_LP_mask * 255.0

    # Fuse with original GT for thing classes
    seg_GT_float = ori_seg_GT.float()
    segGT_obj_mask = (seg_GT_float < 255).float()
    out_mask = (1 - segGT_obj_mask) * mask_CurtVeg + segGT_obj_mask * seg_GT_float

    return out_mask.unsqueeze(1).detach()


def UpdateVisGT(fake_IR, Seg_mask, dis_th):
    """
    Update GT for bright vegetation regions in fake IR images.
    Args:
        fake_IR: (B,3,H,W), normalized fake IR image [-1,1]
        Seg_mask: (B,1,H,W), integer segmentation mask
        dis_th: float, threshold for veg high-brightness ratio
    Returns:
        out_mask: (B,1,H,W), updated mask with uncertain regions as 255
    """
    B, _, H, W = fake_IR.shape
    # Resize Seg_mask to match IR size if needed
    Seg_mask = Seg_mask.squeeze(1)  # (B,Hs,Ws)
    seg_H, seg_W = Seg_mask.shape[-2:]
    if (seg_H != H) or (seg_W != W):
        Seg_mask = F.interpolate(Seg_mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)

    # Create veg and sky masks
    veg_mask = (Seg_mask == VEG).float()
    sky_mask = (Seg_mask == SKY).float()

    # Convert IR to [0,1] and grayscale
    fake_IR_norm = (fake_IR + 1.0) * 0.5
    fake_IR_gray = 0.299 * fake_IR_norm[:, 0:1, :, :] + 0.587 * fake_IR_norm[:, 1:2, :, :] + 0.114 * fake_IR_norm[:,
                                                                                                     2:3, :, :]
    fake_IR_gray = fake_IR_gray.squeeze(1)  # (B,H,W)

    out_mask = Seg_mask.clone().float()

    veg_exists = veg_mask.sum(dim=(1, 2)) > 0
    if veg_exists.any():
        # Veg region stats
        region_veg = veg_mask * fake_IR_gray
        veg_mean = region_veg.view(B, -1).sum(dim=1) / (veg_mask.view(B, -1).sum(dim=1) + 1e-6)
        veg_max = region_veg.view(B, -1).max(dim=1).values
        veg_range_ratio = (veg_max - veg_mean) / (veg_mean + 1e-6)
        """If the difference between the maximum brightness value and the average brightness value of a vegetation region is "
        "greater than a given threshold, the semantic labeling of the corresponding bright region (i.e., the region with "
        "greater than average brightness) is set to uncertain."""
        high_veg = veg_range_ratio > dis_th
        if high_veg.any():
            # Create high-brightness veg mask
            veg_high_mask = (region_veg > veg_mean.view(B, 1, 1)).float()
            out_mask = veg_high_mask * 255.0 + (1.0 - veg_high_mask) * out_mask
        # Optional sky correction
        sky_exists = sky_mask.sum(dim=(1, 2)) > 0
        if sky_exists.any():
            region_sky = sky_mask * fake_IR_gray
            sky_high_mask = (region_sky > veg_mean.view(B, 1, 1)).float()
            out_mask = sky_high_mask * 3 + (1.0 - sky_high_mask) * out_mask
    return out_mask.unsqueeze(1)


def bhw_to_onehot(GT_mask, num_classes):
    """
    Convert GT_mask (B,1,H,W) to one-hot (B,num_classes,H,W), ignoring uncertain pixels (255).

    Args:
        GT_mask: (B,1,H,W)
        num_classes: including uncertain class
    Returns:
        one_hot: (B,num_classes,H,W), float
    """
    GT_mask = GT_mask.squeeze(1).long()
    uncertain_clsidx = num_classes
    gt = torch.where(GT_mask == 255, uncertain_clsidx, GT_mask)
    one_hot = F.one_hot(gt, num_classes=num_classes + 1).float().permute(0, 3, 1, 2)
    return one_hot[:, :-1, :, :]


class AttackImages(nn.Module):
    """ Add small perturbations to input images for adversarial training. """

    def __init__(self, device='cuda', noise_type: str | List[str] = None):
        super(AttackImages, self).__init__()
        self.device = device
        noise_type = noise_type or ['speckle', 'gaussian', 'salt_pepper']
        self.noise_type = noise_type if isinstance(noise_type, list) else [noise_type]
        self.noise_funcs = {
            'gaussian': self._perturb_gaussian,
            'salt_pepper': self._perturb_salt_pepper,
            'speckle': self._perturb_speckle,
            }

    # def forward(self, *images, balance: float = 0.2, total: bool = False, epsilon=0.1):
    #     image_T, image_N = images
    #     if torch.rand(1) > balance:
    #         perturbed_image_T = self._perturb(image_T, total, epsilon)
    #         perturbed_image_N = image_N
    #     else:
    #         perturbed_image_T = image_T
    #         perturbed_image_N = self._perturb(image_N, total, epsilon)
    #
    #     return perturbed_image_T.detach(), perturbed_image_N.detach()

    def forward(self, *images, epsilon=0.1):
        perturbed_images = []
        for image in images:
            perturbed_image = self._perturb(image.detach(), epsilon=epsilon)
            perturbed_images.append(perturbed_image.to(image.device).detach())
        return perturbed_images if len(perturbed_images) > 1 else perturbed_images[0]

    def _perturb_gaussian(self, image, epsilon):
        return torch.from_numpy(random_noise(image.cpu().numpy(), mode='gaussian', mean=0, var=epsilon/2, clip=True)).float()

    def _perturb_salt_pepper(self, image, epsilon):
        return torch.from_numpy(random_noise(image.cpu().numpy(), mode='s&p', salt_vs_pepper=0.5, clip=True)).float()

    def _perturb_poisson(self, image, epsilon):
        return torch.from_numpy(random_noise(image.cpu().numpy(), mode='poisson', clip=True)).float()

    def _perturb_speckle(self, image, epsilon):
        return torch.from_numpy(random_noise(image.cpu().numpy(), mode='speckle', mean=0, var=epsilon/2, clip=True)).float()

    def _perturb(self, image, epsilon: float):
        idx = torch.randperm(len(self.noise_type))[0]
        image = self.noise_funcs[self.noise_type[idx]](image, epsilon)
        return image


class Perturb_Lightness(nn.Module):
    """
    BaseDataset class for the Lightness Experiment.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/test_lightness/"

    night_levels: int = 16
    temperature: int = 30  # in degrees Celsius, for the dark current noise generation
    exposure_time: float = 0.025  # in seconds, for the dark current noise generation (per default 1/fps)
    black_level_offset: float = 5.0  # in [0, 100], to simulate the black level offset of the sensor in % of the maximum pixel value
    full_well_capacity: float = 20000  # in electrons, for the dark current noise generation
    leaky_pixel_percentage: float = 0.005  # percentage of pixels that are 'leaky' (hot pixels) (%)

    def __init__(self):
        self.noise_sigma_per_channel: tuple[float, float, float] = (0.0380987636744976, 0.0388190858066082, 0.0499677807092667)
        self.noise_mean_per_channel: tuple[float, float, float] = (0, 0, 0)
        self.night_scale = torch.arange(0, self.night_levels) / (self.night_levels - 1)  # from 0 to 1
        self.hot_pixel_map = None  # Initialize hot pixel map as None
        super().__init__()

    def forward(self, img_vis):
        if img_vis.min() < 0:
            img_vis = (img_vis + 1) / 2  # scale to [0,1]
        night_level = self.night_scale[torch.randint(0, self.night_levels, (1,)).item()]
        img_vis_noised = self._process_day(img_vis, night_level)
        return img_vis_noised * 2 - 1  # scale back to [-1,1]

    def _process_day(self, img_vis, night_level):
        # decrease luminance of the visible image according to the night level
        img_vis_night = img_vis * night_level
        shape = img_vis_night.shape[-2:]
        gaussian_noise = self._generate_gaussian_noise(shape)
        dark_noise = self._generate_dark_current_noise(shape)
        offset = self.black_level_offset / 100.0
        noise_image = (dark_noise + gaussian_noise + offset).to(img_vis.device)
        img_vis_night_noisy = (img_vis_night * (1 - offset) + noise_image).clamp(0, 1)
        return img_vis_night_noisy

    def _generate_gaussian_noise(self, shape):
        noise_r = torch.randn(shape) * self.noise_sigma_per_channel[0] + self.noise_mean_per_channel[0]
        noise_g = torch.randn(shape) * self.noise_sigma_per_channel[1] + self.noise_mean_per_channel[1]
        noise_b = torch.randn(shape) * self.noise_sigma_per_channel[2] + self.noise_mean_per_channel[2]
        gaussian_noise = torch.stack((noise_r, noise_g, noise_b), dim=0)
        return gaussian_noise

    def _generate_dark_current_noise(self, shape):
        # 1. Base dark current (Poisson)
        mean_electrons = self._estimate_dark_rate() * self.exposure_time
        dark_shot_noise = torch.poisson(torch.full((1, *shape), mean_electrons))
        # 2. Add Hot Pixels (DSNU)
        # We simulate a few pixels that are 100x leakier
        if self.hot_pixel_map is None and self.leaky_pixel_percentage > 0:
            self.hot_pixel_map = self._generate_hot_pixels(shape)
        else:
            self.hot_pixel_map = torch.zeros(shape)
        hot_pixel_noise = self.hot_pixel_map * (mean_electrons * 50)
        total_thermal = (dark_shot_noise + hot_pixel_noise).repeat(3, 1, 1)
        return total_thermal / self.full_well_capacity

    def _generate_hot_pixels(self, shape):
        # Create a static mask of 'leaky' pixels (% of pixels)
        indices = torch.rand(shape) > 1 - self.leaky_pixel_percentage/100
        hot_pixel_map = torch.zeros(shape)
        # Hot pixels leak significantly more electrons
        hot_pixel_map[indices] = torch.rand(indices.sum()) * 0.5
        return hot_pixel_map

    def _estimate_dark_rate(self):
        return 2.0 * 2**((self.temperature - 20) / 8)


# -----------------------------
# GPU helpers that approximate original numpy helpers
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kornia.color as kcolor  # kornia expects RGB in [0,1]

    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False
    raise RuntimeError("kornia is required for rgb<->hsv conversions. Install 'kornia'.")


def ComIoU(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two masks (H,W) on the same device. Returns scalar tensor."""
    # intersection sum
    inter = (mask1 * mask2).sum()
    # union: 1 where either mask > 0
    union = ((mask1 + mask2) > 0.0).float().sum()
    # safe IoU
    return inter / (union + 1e-10)

def determine_color_N(TL_D):
    if TL_D.ndim == 4:
        if TL_D.shape[0] == 1:
            TL_D = TL_D.squeeze(0)
        else:
            res = []
            for TL_D_i in TL_D:
                res.append(determine_color_N(TL_D_i))
            return res
    elif TL_D.ndim != 3:
        raise ValueError("Input TL_D must be 3D or 4D tensor.")
    top_half = TL_D[:, :TL_D.shape[1]//2, :]
    R = (2 * top_half[0] - top_half[2]).mean() + top_half.mean()
    mid_half = TL_D[:, TL_D.shape[1]//3:2*TL_D.shape[1]//3, :]
    Y = (mid_half[1] + mid_half[0] - mid_half[2]).mean() + mid_half.mean()
    bottom_half = TL_D[:, TL_D.shape[1]//2:, :]
    G = (bottom_half[1] + bottom_half[2] - bottom_half[0]).mean() + bottom_half.mean()
    if R > Y and R > G:
        return 'red'
    elif G > Y and G > R:
        return 'green'
    else:
        return 'orange'

