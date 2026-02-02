import numpy as np
import torch
from ImagesCameras import ImageTensor
from kornia.contrib import connected_components
from torchvision.transforms.functional import crop
from tqdm import tqdm
import cv2 as cv
from NightToday import build_train_data_from_config
from NightToday.CrossRAFT import get_wrapper

dataset, _, _ = build_train_data_from_config()
min_size = 300
raft_model = get_wrapper('vis2ir').to('cuda')


def find_RoI(mask, *imgs):
    labels = connected_components(mask.float())
    unique_labels = torch.unique(labels, return_counts=True)
    ret = [[] for _ in imgs]
    for u_label, count in zip(unique_labels[0], unique_labels[1]):
        if u_label == 0:
            continue
        if count < min_size:
            mask[labels == u_label] = 0
            continue
        elif count < 5000:
            ys, xs = torch.where(labels.squeeze() == u_label)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min = torch.min(xs).item()
            x_max = torch.max(xs).item()
            y_min = torch.min(ys).item()
            y_max = torch.max(ys).item()
            for idx, img in enumerate(imgs):
                cropped_img = crop(img, y_min, x_min, y_max - y_min, x_max - x_min)
                if cropped_img.shape[-2] * cropped_img.shape[-1] > min_size:
                    ret[idx].append(ImageTensor(cropped_img * 0.5 + 0.5))
    return ret


def determine_color(FG):
    FG = FG ** 2
    R = 1.25 * FG[:, 0:1] - 1. * FG[:, 1:2] - 0.5 * FG[:, 2:3]
    G = 0.75 * FG[:, 1:2] - 1.75 * FG[:, 0:1] + 0.75 * FG[:, 2:3]
    C_intensity, color_idx = torch.max(torch.cat([R, G], dim=1), dim=1, keepdim=True)
    C_intensity = C_intensity * (C_intensity > 0.1)
    if C_intensity.sum() == 0:
        return 'none'
    color_ratio = torch.sum(C_intensity * (color_idx == 0)) / C_intensity.sum()
    if color_ratio > 0.5:
        return 'red'
    elif color_ratio <= 0.5:
        return 'green'


def manual_keypoints_selection(N: ImageTensor, T: ImageTensor = None) -> tuple[list, list] | list:
    global im_temp, pts_temp, rightclick

    def mouseHandler(event, x, y, flags, param):
        global pts_temp, rightclick
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(im_temp, (x, y), 2, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(im_temp, str(len(pts_temp) + 1), (x + 3, y + 3),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 200, 250), 1, cv.LINE_AA)
            if len(pts_temp) < 2:
                pts_temp = np.append(pts_temp, [(x, y)], axis=0)
            elif len(pts_temp) == 2:
                rightclick = True
            if not rightclick:
                cv.imshow('Select the area to crop TL -> BR, rightclick to end', im_temp)
        if event == cv.EVENT_RBUTTONDOWN or event == cv.EVENT_MOUSEWHEEL:
            rightclick = True
        # Image REF

    if T is not None:
        N_ = np.ascontiguousarray((T*4/5 + N/5).pyrUp().to_opencv(), dtype=np.uint8)
    else:
        N_ = np.ascontiguousarray(N.to_opencv(), dtype=np.uint8)

    # Vector temp
    pts_temp = np.empty((0, 2), dtype=np.int32)
    nb_patch = 0
    patches = ([], []) if T is not None else []  # N and T
    # Create a window
    cv.namedWindow('Select the area to crop TL -> BR, rightclick to end')

    rightclick = False

    while not rightclick:
        im_temp = N_.copy()
        cv.imshow('Select the area to crop TL -> BR, rightclick to end', im_temp)
        cv.setMouseCallback('Select the area to crop TL -> BR, rightclick to end', mouseHandler)
        while True:
            if cv.waitKey(10) == 27 or rightclick or (len(pts_temp) == 2):
                break
        if pts_temp.shape[0] < 2:
            continue
        x0, x1, y0, y1 = pts_temp[0][0], pts_temp[1][0], pts_temp[0][1], pts_temp[1][1]
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if T is not None:
            x0, x1, y0, y1 = x0//2, x1//2, y0//2, y1//2
            w, h = x1 - x0, y1 - y0
            crop_N = N[:, :, y0-h:y1+h, x0-w:x1+w]
            crop_T = T[:, :, y0:y1, x0:x1]
            patches[0].append(ImageTensor(crop_N))
            patches[1].append(ImageTensor(crop_T))
        else:
            patches.append(ImageTensor(N[:, :, y0:y1, x0:x1]))

        N_[pts_temp[0][1]:pts_temp[1][1], pts_temp[0][0]:pts_temp[1][0]] = 0
        pts_temp = np.empty((0, 2), dtype=np.int32)

    cv.destroyAllWindows()
    del im_temp, pts_temp, rightclick
    return patches if T is not None else patches


count_red_D = 0
count_green_D = 0
count_red_N = 0
count_green_N = 0
dataset_name = "LYNRED"
dataset_path = f"/home/godeta/PycharmProjects/TIR2VIS/datasets/{dataset_name}/{dataset_name}_datasets/"
for data in tqdm(dataset):
    TF_D = data['seg_D'] == 6
    if TF_D.sum() > 0:
        FG = manual_keypoints_selection(ImageTensor(data['D']))
        if len(FG) > 0:
            for fg in FG:
                surface = fg.shape[-2] * fg.shape[-1]
                color = determine_color(fg)
                if color == 'red':
                    count_red_D += 1
                elif color == 'green':
                    count_green_D += 1
                else:
                    continue
                fg.save(dataset_path + "FG_sample_D/",
                        name=f"{color}_{surface}_{count_red_D if color == 'red' else count_green_D}")
    TF_N = data['seg_TN'] == 6
    if TF_N.sum() > 0:
        data['N'], data['T'] = data['N'].to('cuda')*0.5+0.5, data['T'].to('cuda')*0.5+0.5
        data['N'] = raft_model(data['N'].to('cuda'), data['T'].to('cuda'))
        FG_N, FG_T = manual_keypoints_selection(ImageTensor(data['N']), ImageTensor(data['T']))
        if len(FG_N) > 0:
            for fg_N, fg_T in zip(FG_N, FG_T):
                surface = fg_N.shape[-2] * fg_N.shape[-1]
                color = determine_color(fg_N)
                if color == 'red':
                    count_red_N += 1
                elif color == 'green':
                    count_green_N += 1
                else:
                    continue
                ImageTensor(fg_N).save(dataset_path + "FG_sample_N/",
                                       name=f"{color}_{surface}_{count_red_N if color == 'red' else count_green_N}")
                ImageTensor(fg_T).save(dataset_path + "FG_sample_T/",
                                       name=f"{color}_{surface}_{count_red_N if color == 'red' else count_green_N}")
