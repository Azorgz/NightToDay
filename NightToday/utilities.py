import functools

import numpy as np
import skimage
import torch
from kmeans_pytorch import kmeans
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.contrib import connected_components
from kornia.morphology import closing, dilation
from scipy.ndimage import gaussian_filter
from skimage import measure
from torch import nn, tensor
from torch.nn.functional import conv2d
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

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
        tqdm_flag=False,
        iter_limit=max_iter
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
        class_count = class_mask.sum(dim=2)  # (B,1)
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
    Segmask_Light_DN_k5 = LightMaskDenoised(Seg_mask, real_vis, 5)
    Segmask_Light_DN_k3 = LightMaskDenoised(Segmask_Light_DN_k5, real_vis, 3)
    return torch.where(Segmask_Light_DN_k3 == TRAFFICLIGHT, 1., 0.)


def LightMaskDenoised(Seg_mask, real_vis, Avg_KernelSize, min_area_ratio=0.001):
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
    device = real_vis.device

    # Original masks
    Seg_mask = Seg_mask.squeeze(1)  # (B,H,W)
    light_mask_ori = (Seg_mask == TRAFFICLIGHT).float()
    sky_mask = (Seg_mask == SKY).float()

    # Grayscale normalized
    real_gray = ((real_vis + 1.0) * 0.5).mean(dim=1)  # (B,1,H,W)

    # Local average pooling
    padsize = Avg_KernelSize // 2
    local_mean = F.avg_pool2d(light_mask_ori * real_gray, Avg_KernelSize, stride=1,
                              padding=padsize)

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
    # # Invert mask
    # inverted = 1 - light_mask_denoised.unsqueeze(1)  # (B,1,H,W)
    #
    # # Label connected regions using max pooling approximation
    # filled = F.max_pool2d(inverted, 3, stride=1, padding=1)
    # small_holes = (filled > 0) & (inverted > 0)
    #
    # # Estimate area threshold based on batch-wise mask area
    # mask_area = light_mask_ori.sum(dim=[1, 2], keepdim=True).clamp(min=1.0)
    # area_th = (mask_area * min_area_ratio).view(B, 1, 1, 1)
    #
    # # Fill only small holes
    # filled_holes = F.relu(inverted - small_holes.float())
    # filled_holes = torch.where(filled_holes.sum(dim=[2, 3], keepdim=True) < area_th, 1.0, 0.0)
    #
    # # Combine with denoised mask
    # light_mask_denoised = torch.clamp(light_mask_denoised + filled_holes.squeeze(1), 0, 1)

    # Construct final mask
    out_mask = (1 - light_mask_ori) * Seg_mask + 6.0 * light_mask_denoised + 255.0 * (
            light_mask_ori - light_mask_denoised)

    return out_mask.unsqueeze(1)  # (B,1,H,W)


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


def detect_hotpoint_blob(I_vi, M_tl, contrast_k=0.98, sigma_ratio=0.20):
    """
    Detect hot-point inside TL mask based on brightness blob + COM.
    Inputs:
        I_vi: Bx3xHxW (visible night RGB)
        M_tl: Bx1xHxW (traffic light mask)
    Output:
        M_hot: Bx1xHxW (Gaussian hot-point mask)
    """

    # --- 1. Convert to grayscale ---
    Gray = I_vi.mean(1, keepdim=True)
    masked_gray = Gray * M_tl

    # --- 2. Adaptive brightness threshold ---
    max_v = masked_gray.flatten(1).max(dim=-1)[0]
    T = contrast_k * max_v

    # --- 3. Bright region mask ---
    M_blob = ((Gray > T.view(-1, 1, 1, 1)).float()) * M_tl

    # --- 4. Morphological cleaning ---
    kernel = torch.ones(3, 3, device=I_vi.device)
    M_blob = closing(M_blob, kernel)
    M_blob = dilation(M_blob, kernel)

    area_blob = M_blob.sum(dim=[1, 2, 3])
    valid_blob = area_blob > 0
    ret = torch.zeros_like(Gray, device=Gray.device)
    if not valid_blob.any():
        return ret

    # --- 5. Compute center of mass of the blob ---
    B, _, H, W = M_blob.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=I_vi.device),
        torch.arange(W, device=I_vi.device),
        indexing='ij'
    )
    grid_y, grid_x = grid_y[None].repeat(B, 1, 1)[valid_blob], grid_x[None].repeat(B, 1, 1)[valid_blob]
    cx = (M_tl[valid_blob] * grid_x).sum((1, 2, 3)) / M_tl.sum(dim=[1, 2, 3])[valid_blob]
    cy = (M_blob[valid_blob] * grid_y).sum((1, 2, 3)) / area_blob[valid_blob]

    # --- 6. Generate Gaussian hot-point mask ---
    # sigma proportional to TL size
    tl_size = torch.sqrt(M_tl.sum((1, 2, 3)))  # ~diameter of TL region
    sigma = sigma_ratio * tl_size

    # Build gaussian for each batch
    M_hot = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2 + 1e-6))

    M_hot = M_hot.unsqueeze(1)  # Bx1xHxW
    M_hot = M_hot / (M_hot.flatten(2).max(dim=-1, keepdim=True)[0] + 1e-6)  # normalize
    M_hot = M_hot * M_tl[valid_blob]  # mask outside TL region
    ret[valid_blob] = M_hot/2
    return ret


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
    veg_mask = (Seg_mask == 8).float()
    sky_mask = (Seg_mask == 10).float()

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
            out_mask = sky_high_mask * 255.0 + (1.0 - sky_high_mask) * out_mask
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

    def __init__(self, device='cuda'):
        super(AttackImages, self).__init__()
        self.device = device

    def forward(self, *images, balance: float = 0.2, total: bool = False, epsilon=0.1):
        image_T, image_N = images
        if torch.rand(1) > balance:
            perturbed_image_T = self._perturb(image_T, total, epsilon)
            perturbed_image_N = image_N
        else:
            perturbed_image_T = image_T
            perturbed_image_N = self._perturb(image_N, total, epsilon)

        return perturbed_image_T.detach(), perturbed_image_N.detach()

    def _perturb(self, image, total: bool, epsilon):
        l, a, b = rgb_to_lab(image * 0.5 + 0.5).split(1, dim=1)
        noise = torch.randn_like(l, device=self.device)
        if total:
            perturbed_l = (epsilon / 5 * noise) * 100
        else:
            perturbed_l = (l / 100 + epsilon * noise) * 100
        perturbed_l = torch.clamp(perturbed_l, 0, 100.0)
        return lab_to_rgb(torch.cat([perturbed_l, a, b], dim=1)) * 2 - 1


# -----------------------------
# GPU helpers that approximate original numpy helpers
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: these functions expect torch tensors on GPU (or same device).
# Shapes assumed:
#  - mask tensors: (H, W) floats {0.,1.} or bool
#  - fake_IR_masked: (C, H, W)
#  - real_vis_masked: (C, H, W)
# All operations are torch-only, vectorized, and avoid needless temporaries.

# Use kornia color conversions if available; otherwise provide a fallback.
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


def LocalVerticalFlip(input_mask: torch.Tensor,
                      fake_IR: torch.Tensor,
                      real_vis: torch.Tensor,
                      region_min_row: int,
                      region_max_row: int):
    """
    Locally vertically flip the band defined by [region_min_row, region_max_row] in the H dimension.
    - input_mask: (H, W)
    - fake_IR: (C, H, W)
    - real_vis: (C, H, W)
    All tensors on same device. Returns (mask_out, fake_out, vis_out) with same shapes.
    """
    dev = input_mask.device
    H = input_mask.shape[0]

    # flip full images along vertical axis (height dim 0 for mask; dim 1 for C,H,W tensors)
    mask_flip = torch.flip(input_mask, dims=[0])                       # (H, W)
    fake_flip = torch.flip(fake_IR, dims=[1])                          # (C, H, W)
    vis_flip = torch.flip(real_vis, dims=[1])                          # (C, H, W)

    # compute flipped region indices (these are torch scalars but used for slicing, convert to int)
    # Using python ints for slicing is acceptable; they don't move large tensors.
    flip_rmin = int(H - region_max_row - 1)
    flip_rmax = int(H - region_min_row - 1)

    # prepare outputs (zeros_like keeps dtype/device)
    out_mask = torch.zeros_like(input_mask)                             # (H, W)
    out_fake = torch.zeros_like(fake_IR)                                # (C, H, W)
    out_vis = torch.zeros_like(real_vis)                                # (C, H, W)

    # copy flipped region into the original region positions
    out_mask[region_min_row:region_max_row + 1, :] = mask_flip[flip_rmin:flip_rmax + 1, :]
    out_fake[:, region_min_row:region_max_row + 1, :] = fake_flip[:, flip_rmin:flip_rmax + 1, :]
    out_vis[:, region_min_row:region_max_row + 1, :] = vis_flip[:, flip_rmin:flip_rmax + 1, :]

    return out_mask, out_fake, out_vis


def _dilate_mask_torch(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Dilate binary mask using two passes of max-pooling (mimics the original double negative-maxpool trick).
    - mask: (H,W) float tensor {0.,1.}
    returns dilated mask (H,W) float
    """
    # convert to shape (1,1,H,W) for pool
    m = mask.unsqueeze(0).unsqueeze(0)                                  # (1,1,H,W)
    # max pool (same padding)
    pad = (k - 1) // 2
    m1 = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad)        # dilate (first pass)
    m2 = -F.max_pool2d(-m1, kernel_size=k, stride=1, padding=pad)       # dilate (second pass)
    return m2.squeeze(0).squeeze(0)                                     # (H,W)


def Red2Green(input_rgb: torch.Tensor, input_mask: torch.Tensor):
    """
    Torch-only replacement of Red2Green.
    Inputs:
      - input_rgb: (C, H, W), expected in range [-1,1] like original code
      - input_mask: (H, W) binary {0,1}
    Returns:
      - out_rgb: (C, H, W) float in [0,1] (similar to original res_numpy)
      - red_mask_area: scalar float tensor
      - red_mask_fused: (H, W) float mask (0/1)
    Behavior replicates thresholds and transforms from original numpy code.
    """
    dev = input_rgb.device
    # normalize to [0,1] as original did
    rgb_norm = (input_rgb + 1.0) * 0.5                                  # (C,H,W)

    # apply mask (broadcast)
    masked_rgb = rgb_norm * input_mask.unsqueeze(0)                      # (C,H,W)

    # convert to HSV (kornia expects (B, C, H, W) and RGB in [0,1])
    hsv = kcolor.rgb_to_hsv(masked_rgb.unsqueeze(0))                     # (1,3,H,W) with H in [0,1]
    hsv = hsv.squeeze(0)                                                 # (3,H,W)

    # match original scalings:
    Hchan = hsv[0, :, :] * 180.0                                         # hue in [0,180)
    Schan = hsv[1, :, :] * 255.0                                         # saturation in [0,255]
    Vchan = hsv[2, :, :] * 255.0                                         # value in [0,255]

    # threshold masks as original
    s_mask = (Schan > 42.0).float()
    v_mask = (Vchan > 45.0).float()
    h_mask1 = (Hchan < 25.0).float()
    h_mask2 = (Hchan > 155.0).float()

    # two red candidate masks
    red1 = s_mask * v_mask * h_mask1                                     # (H,W)
    red2 = s_mask * v_mask * h_mask2                                     # (H,W)

    # dilate both masks using torch pooling trick (double negative max pool)
    red1_d = _dilate_mask_torch(red1)
    red2_d = _dilate_mask_torch(red2)

    # fused mask and areas
    red_mask_fused = (red1_d + red2_d).clamp(0.0, 1.0)                    # (H,W)
    area1 = red1_d.sum()
    area2 = red2_d.sum()
    red_mask_area = area1 + area2

    # compute modified hue outputs following original linear transforms
    # h_r2g_out1 = red_mask1 * ((input_h * 0.12) + 87.0)
    # h_r2g_out2 = red_mask2 * ((input_h * 0.12) + 64.0)
    h_r2g_out1 = red1_d * (Hchan * 0.12 + 87.0)
    h_r2g_out2 = red2_d * (Hchan * 0.12 + 64.0)

    # compose final hue: keep original where not red, replace where red
    not_red = (1.0 - red1_d - red2_d).clamp(0.0, 1.0)
    h_out = not_red * Hchan + h_r2g_out1 + h_r2g_out2                      # (H,W)

    # assemble HSV output (scale back to kornia expected ranges [0,1])
    hsv_out = torch.empty_like(hsv, device=dev)
    hsv_out[0] = h_out / 180.0                                              # hue in [0,1]
    hsv_out[1] = not_red * hsv[1] + (red1_d + red2_d) * (hsv[1] * 0.5)      # saturation mixing like original
    hsv_out[2] = hsv[2]                                                      # keep V

    # back to RGB in [0,1]
    rgb_out = kcolor.hsv_to_rgb(hsv_out.unsqueeze(0)).squeeze(0)            # (3,H,W) in [0,1]

    return rgb_out, red_mask_area, red_mask_fused


def Green2Red(input_rgb: torch.Tensor, input_mask: torch.Tensor):
    """
    Torch-only replacement of Green2Red.
    Inputs and returns same types as Red2Green.
    """
    dev = input_rgb.device
    rgb_norm = (input_rgb + 1.0) * 0.5
    masked_rgb = rgb_norm * input_mask.unsqueeze(0)

    hsv = kcolor.rgb_to_hsv(masked_rgb.unsqueeze(0)).squeeze(0)
    Hchan = hsv[0] * 180.0
    Schan = hsv[1] * 255.0
    Vchan = hsv[2] * 255.0

    # thresholds per original
    s_mask = (Schan > 25.0).float()
    v_mask = (Vchan > 45.0).float()
    h_mask1 = (Hchan < 90.0).float()
    h_mask2 = (Hchan > 67.0).float()
    h_mask3 = (Hchan > 90.0).float()
    h_mask4 = (Hchan < 110.0).float()

    green1 = s_mask * v_mask * h_mask1 * h_mask2
    green2 = s_mask * v_mask * h_mask3 * h_mask4

    green1_d = _dilate_mask_torch(green1)
    green2_d = _dilate_mask_torch(green2)

    green_mask_fused = (green1_d + green2_d).clamp(0.0, 1.0)
    area1 = green1_d.sum()
    area2 = green2_d.sum()
    green_mask_area = area1 + area2

    # hue transforms as original:
    # h_g2r_out1 = green_mask1 * ((input_h * 0.5) - 33.5)
    # h_g2r_out2 = green_mask2 * ((input_h * (-0.5)) + 55.0)
    h_g2r_out1 = green1_d * (Hchan * 0.5 - 33.5)
    h_g2r_out2 = green2_d * (Hchan * -0.5 + 55.0)

    not_green = (1.0 - green1_d - green2_d).clamp(0.0, 1.0)
    h_out = not_green * Hchan + h_g2r_out1 + h_g2r_out2

    hsv_out = torch.empty_like(hsv, device=dev)
    hsv_out[0] = h_out / 180.0
    hsv_out[1] = not_green * hsv[1] + (green1_d + green2_d) * (hsv[1] * 4.0)
    hsv_out[2] = hsv[2]

    rgb_out = kcolor.hsv_to_rgb(hsv_out.unsqueeze(0)).squeeze(0)

    return rgb_out, green_mask_area, green_mask_fused


def ObtainTLightMixedMask(temp_connect_mask: torch.Tensor,
                          fake_IR_masked: torch.Tensor,
                          real_vis_masked: torch.Tensor,
                          patch_height: int):
    """
    Torch-only, GPU-friendly reimplementation of the original function.
    Inputs:
      - temp_connect_mask: (H, W) {0,1} float tensor
      - fake_IR_masked: (C, H, W) float tensor
      - real_vis_masked: (C, H, W) float tensor
      - patch_height: int (H)
    Returns:
      (output_FG_Mask, output_FG_FakeIR, output_FG_RealVis,
       output_highlight_mask, output_FG_top_mask, output_FG_bottom_mask)
    All outputs are torch tensors on the same device as inputs.
    """
    dev = temp_connect_mask.device
    H, W = temp_connect_mask.shape

    # compute row/col sums for aspect ratio estimate (like original)
    row_sum = temp_connect_mask.sum(dim=1)                                # (H,)
    col_sum = temp_connect_mask.sum(dim=0)                                # (W,)
    # avoid division-by-zero: add tiny eps
    region_AspectRatio = (col_sum.max() + 1e-10) / (row_sum.max() + 1e-10)

    # get red/green original analysis using torch functions
    _, temp_r2g_area_ori, temp_red_mask_ori = Red2Green(real_vis_masked, temp_connect_mask)
    _, temp_g2r_area_ori, temp_green_mask_ori = Green2Red(real_vis_masked, temp_connect_mask)

    # compute row position matrix (H,W) without extra big temporaries when possible
    rows = torch.arange(patch_height, device=dev, dtype=temp_connect_mask.dtype).view(patch_height, 1)  # (H,1)
    row_pos = rows.expand(-1, patch_height)                                 # (H,H) but H==patch_height
    # mask_pos stores row index inside mask, 0 elsewhere
    mask_pos = temp_connect_mask * row_pos                                   # (H,W) when W==patch_height in orig; if not, broadcasting will align
    # mask_pos padding sets non-mask to patch_height
    mask_pos_padding_h = temp_connect_mask * row_pos + (1.0 - temp_connect_mask) * float(patch_height)

    # compute min / max rows covered by mask (tensors)
    mask_pos_row_min = mask_pos_padding_h.min()                              # scalar tensor
    mask_pos_row_max = mask_pos.max()                                        # scalar tensor
    mask_mid_row = ((mask_pos_row_min + mask_pos_row_max) / 2.0).floor()      # scalar tensor

    # top/bottom masks as boolean comparisons (no slicing ints)
    top_mask = ((row_pos >= mask_pos_row_min) & (row_pos <= mask_mid_row)).float()  # (H,H)
    # if W != H we must tile top_mask to (H,W) properly: make a (H,1) comparison then expand to W
    if top_mask.shape[1] != W:
        top_mask = ((rows >= mask_pos_row_min) & (rows <= mask_mid_row)).float().expand(-1, W)  # (H,W)
    bottom_mask = 1.0 - top_mask                                                 # (H,W)

    IoU_th = 0.5

    # initialize outputs (defaults)
    output_FG_Mask = temp_connect_mask
    output_FG_FakeIR = fake_IR_masked
    output_FG_RealVis = real_vis_masked
    output_highlight_mask = temp_red_mask_ori if temp_r2g_area_ori > temp_g2r_area_ori else temp_green_mask_ori

    # branch for large aspect ratio (vertical flip + extra synthesis)
    if (region_AspectRatio > 1.75).item():
        # random decision whether to vertically flip (tensor rand on device)
        ver_flip_idx = torch.rand(1, device=dev)
        temp_VerFlip_mask, temp_VerFlip_fakeIR, temp_VerFlip_realVis = LocalVerticalFlip(
            temp_connect_mask, fake_IR_masked, real_vis_masked,
            int(mask_pos_row_min.item()), int(mask_pos_row_max.item())
        )

        # compute red/green analysis on flipped VIS
        _, temp_r2g_area, temp_red_mask = Red2Green(temp_VerFlip_realVis, temp_VerFlip_mask)
        _, temp_g2r_area, temp_green_mask = Green2Red(temp_VerFlip_realVis, temp_VerFlip_mask)

        DLS_idx = torch.rand(1, device=dev)                                       # double-light synthesis coin flip
        Decay_factor = 1.0

        if ver_flip_idx.item() > 0.5:
            # flipped case
            if (temp_r2g_area > temp_g2r_area).item():
                vis_GT_masked = (temp_r2g_vis := None)  # placeholder for clarity (we don't reuse the variable)
                # produce real-vision output choosing flipped->red2green
                vis_GT_masked, _, _ = Red2Green(temp_VerFlip_realVis, temp_VerFlip_mask)  # get vis transform
                output_FG_RealVis = (vis_GT_masked - 0.5) * 2.0
                output_highlight_mask = temp_red_mask

                if DLS_idx.item() > 0.5:
                    mask_vertical_IoU = ComIoU(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU.item() > IoU_th:
                        # synthesize top/bottom fused fake IR
                        top_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (top_mask * temp_connect_mask * fake_IR_masked)
                        bottom_fake_IR_masked = bottom_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_connect_mask + bottom_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_VerFlip_mask * fused_mask
                    else:
                        output_FG_Mask = temp_VerFlip_mask
                        output_FG_FakeIR = temp_VerFlip_fakeIR
                else:
                    output_FG_Mask = temp_VerFlip_mask
                    output_FG_FakeIR = temp_VerFlip_fakeIR

            else:
                # flipped and green-dominant
                vis_GT_masked, _, _ = Green2Red(temp_VerFlip_realVis, temp_VerFlip_mask)
                output_FG_RealVis = (vis_GT_masked - 0.5) * 2.0
                output_highlight_mask = temp_green_mask

                if DLS_idx.item() > 0.5:
                    mask_vertical_IoU = ComIoU(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU.item() > IoU_th:
                        bottom_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (bottom_mask * temp_connect_mask * fake_IR_masked)
                        top_fake_IR_masked = top_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = bottom_mask * temp_connect_mask + top_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_VerFlip_mask * fused_mask
                    else:
                        output_FG_Mask = temp_VerFlip_mask
                        output_FG_FakeIR = temp_VerFlip_fakeIR
                else:
                    output_FG_Mask = temp_VerFlip_mask
                    output_FG_FakeIR = temp_VerFlip_fakeIR

        else:
            # no flip applied in this branch: use original VIS and possibly synthesize double spot
            output_FG_RealVis = real_vis_masked
            if (temp_r2g_area_ori > temp_g2r_area_ori).item():
                output_highlight_mask = temp_red_mask_ori
                if DLS_idx.item() > 0.5:
                    mask_vertical_IoU = ComIoU(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU.item() > IoU_th:
                        top_fake_IR_masked = top_mask * temp_connect_mask * fake_IR_masked
                        bottom_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (bottom_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR)
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_connect_mask + bottom_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_connect_mask * fused_mask
                    else:
                        output_FG_Mask = temp_connect_mask
                        output_FG_FakeIR = fake_IR_masked
                else:
                    output_FG_Mask = temp_connect_mask
                    output_FG_FakeIR = fake_IR_masked
            else:
                output_highlight_mask = temp_green_mask_ori
                if DLS_idx.item() > 0.5:
                    mask_vertical_IoU = ComIoU(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU.item() > IoU_th:
                        top_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (top_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR)
                        bottom_fake_IR_masked = bottom_mask * temp_connect_mask * fake_IR_masked
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_VerFlip_mask + bottom_mask * temp_connect_mask
                        output_FG_Mask = temp_connect_mask * fused_mask
                    else:
                        output_FG_Mask = temp_connect_mask
                        output_FG_FakeIR = fake_IR_masked
                else:
                    output_FG_Mask = temp_connect_mask
                    output_FG_FakeIR = fake_IR_masked

    else:
        # small aspect ratio: use original masks/patches; highlight mask based on original analysis
        output_FG_Mask = temp_connect_mask
        output_FG_FakeIR = fake_IR_masked
        output_FG_RealVis = real_vis_masked
        output_highlight_mask = temp_red_mask_ori if temp_r2g_area_ori > temp_g2r_area_ori else temp_green_mask_ori

    # compute top/bottom portions of final mask
    output_FG_top_mask = output_FG_Mask * top_mask
    output_FG_bottom_mask = output_FG_Mask * bottom_mask

    return output_FG_Mask, output_FG_FakeIR, output_FG_RealVis, output_highlight_mask, output_FG_top_mask, output_FG_bottom_mask


# -----------------------------
# Updated FakeIRFGMergeMaskv4 (GPU-first, Kornia CC)
# -----------------------------
def get_FG_MergeMask(vis_gt,
                     IR_seg_tensor,
                     real_vis,
                     fake_tn):
    """
    Batched GPU-first implementation that uses the torch-based ObtainTLightMixedMask_torch by default.
    Args:
        vis_gt: (B,1,H,W) integer segmentation mask for visible image
        IR_seg_tensor: (B,C,H,W) logits for IR segmentation
        real_vis: (B,3,H,W) real visible image
        fake_tn: (B,3,H,W) fake IR image
    Returns:
        out_FG_mask: (B,3,H,W) float {0,1}
        out_FG_FakeIR: (B,3,H,W) float
        out_FG_RealVis: (B,3,H,W) float
        out_FG_mask_ori: (B,3,H,W) float {0,1}
        out_HL_mask: (B,3,H,W) float {0,1}
        out_Light_TopMask: (B,1,H,W) float {0,1}
        out_Light_BottomMask: (B,1,H,W) float {0,1}
    """

    device = IR_seg_tensor.device
    B, C, H, W = IR_seg_tensor.shape
    fake_ir, fake_ni = fake_tn.split(3, dim=1)

    # segmentation argmax on GPU
    ir_seg = torch.argmax(IR_seg_tensor.detach(), dim=1).float().unsqueeze(1)  # (B,H,W)

    vis_FG_idx_list = [6, 7, 17]
    traffic_sign_list = [6, 7]

    IR_road_mask = (ir_seg < 2.0).float()
    IR_FG1_mask = (ir_seg > 10.0).float()
    IR_light_mask = (ir_seg == 6.0).float()
    IR_sign_mask = (ir_seg == 7.0).float()
    IR_FG_mask = IR_FG1_mask + IR_light_mask + IR_sign_mask

    # per-image accumulators
    FG_Mask = torch.zeros((B, 1, H, W), device=device)
    HL_Mask = torch.zeros((B, 1, H, W), device=device)
    Light_Top = torch.zeros((B, 1, H, W), device=device)
    Light_Bottom = torch.zeros((B, 1, H, W), device=device)
    FG_FakeTN = torch.zeros((B, 6, H, W), device=device)
    FG_RealVis = torch.zeros((B, 3, H, W), device=device)
    valid = False

    for i, idx in enumerate(vis_FG_idx_list):
        temp_mask_ori = (vis_gt == idx).float()
        if temp_mask_ori.sum().item() == 0:
            continue

        temp_mask_erode = -nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(-temp_mask_ori)
        binary = (temp_mask_erode > 0.5).float()
        labels = connected_components(binary)  # (B,1,H,W)

        for b in range(B):
            unique_labels, areas = torch.unique(labels[b], return_counts=True)
            unique_labels, areas = unique_labels[unique_labels != 0], areas[unique_labels != 0]
            if unique_labels.numel() == 0:
                continue

            for area, comp_label in zip(areas, unique_labels):
                if area <= 50:
                    continue
                valid = True
                comp_mask = (labels[b] == comp_label).float()  # (1,H,W) Component mask from VIS GT
                if (comp_mask * IR_FG_mask[b]).sum().item() == 0:  # Overlap FG check
                    fake_TN_masked = comp_mask * fake_tn[b]
                    real_vis_masked = comp_mask * real_vis[b]
                    if idx in traffic_sign_list:  # traffic sign or light
                        if idx == 6:  # traffic light
                            (temp_FG_Mask,
                             temp_FG_FakeIR,
                             temp_FG_RealVis,
                             temp_highlight_mask,
                             temp_TopMask,
                             temp_BottomMask) = (
                                ObtainTLightMixedMask(comp_mask, fake_TN_masked, real_vis_masked, H))
                            FG_Mask[b] += temp_FG_Mask
                            FG_FakeTN[b] += temp_FG_FakeIR
                            FG_RealVis[b] += temp_FG_RealVis
                            HL_Mask[b] += temp_highlight_mask
                            Light_Top[b] += temp_TopMask
                            Light_Bottom[b] += temp_BottomMask
                        else:  # sign
                            FG_Mask[b] += comp_mask
                            FG_FakeTN[b] += fake_TN_masked
                            FG_RealVis[b] += real_vis_masked
                    else:
                        road_mask_prod = comp_mask * IR_road_mask[b]
                        IoU_th = 0.1 * area
                        if road_mask_prod.sum().item() > IoU_th:
                            FG_Mask[b] += comp_mask
                            FG_FakeTN[b] += fake_TN_masked
                            FG_RealVis[b] += real_vis_masked

    return (valid, FG_Mask.repeat(1, 3, 1, 1), FG_FakeTN, FG_RealVis, HL_Mask.repeat(1, 3, 1, 1),
            torch.cat([Light_Top, Light_Bottom], dim=1))
# endregion -----------------------------------------------------
