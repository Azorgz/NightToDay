import numpy as np
import torch
import cv2
from tqdm import tqdm

from Dataset_ICCV import test_loader
from ImagesCameras import ImageTensor
from posenet.third_party.CrossModalFlow.utils import InputPadder
from flow_utils import backwarp_tensor
from get_model import get_model

model = get_model()

dataset = ['LLVIP']  #'Lynred_day', 'Lynred_night', 'FLIR_day', 'FLIR_night', 'CATS_in', 'CATS_out'


def img2tensor(img):
    #print(img.shape)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img.cuda()


def tensor2img(img):
    img = img[0].clip(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    img = np.round(img * 255).astype(np.uint8)
    return img


def run_tensor(img1, img2):
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    flow = model(img1, img2)['flow']
    return padder.unpad(flow)


def run_numpy(img1, img2):
    img1 = img2tensor(img1)
    img2 = img2tensor(img2)
    flow = run_tensor(img1, img2)
    flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    return flow


def align_img(img1, img2):
    flow = run_tensor(img1, img2)
    warped = backwarp_tensor(img2, flow)
    warped = tensor2img(warped)
    return warped


def imread(path):
    img = cv2.imread(path)
    print(img.shape)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    for d in dataset:
        Dataset = test_loader(dataset=d, batch_size=1)
        for ir, vis, name in tqdm(Dataset, f'processing {d}'):
            aligned = align_img(ir, vis)
            ImageTensor(aligned).save(f'/home/godeta/Images/ICCV/Data_publi/{d}/ir_reg_CrossRAFT/', name=name[0],
                                             ext='png')
