import os
import socket
from ImagesCameras import ImageTensor
from torch import Tensor
from .DatasetBase import TrainDataset, TestDataset


class FLIR(TrainDataset):
    """
    Dataset class for the FLIR dataset.
    """
    name = 'FLIR'
    root = '/silenus/PROJECTS/pr-remote-sensing-1a/godeta/datasets/FLIR/' \
        if not 'laptop'in socket.gethostname() else \
        '/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/'


    def __init__(self, opt):
        self.train_D = self.root + "FLIR_datasets/trainA"
        self.train_D_T = self.root + "FLIR_datasets/trainA_T"
        self.train_T = self.root + "FLIR_datasets/trainB"
        self.train_N = self.root + "FLIR_datasets/trainC"
        self.TN_edges = self.root + "FLIR_datasets/FLIR_IR_edge_map"
        self.D_edges = self.root + "FLIR_datasets/FLIR_Vis_edge_map"
        self.D_seg = self.root + "FLIR_datasets/FLIR_Vis_seg_mask"
        self.TN_seg = self.root + "FLIR_datasets/FLIR_IR_seg_mask"
        self.TL_D = self.root + "FLIR_datasets/FG_sample_D_TN/"
        self.TL_D = [self.TL_D + f for f in sorted(os.listdir(self.TL_D))]
        self.TL_T = self.root + "FLIR_datasets/FG_sample_T/"
        self.TL_T = [self.TL_T + f for f in sorted(os.listdir(self.TL_T))]
        self.TL_N = self.root + "FLIR_datasets/FG_sample_N/"
        self.TL_N = [self.TL_N + f for f in sorted(os.listdir(self.TL_N))]
        self.crop_path = self.root + '/FLIR_datasets/crop.yaml'
        super().__init__(opt)

    def load_image(self, path: list[str], idx: int, crop: bool = False, seg=False, fac=1., **kwargs) -> Tensor:
        """
        Load an image from a given path and return it as a Tensor.
        """
        image = ImageTensor(path[idx]) ** fac * 255 if seg else ImageTensor(path[idx])
        if image.depth==16:
            image = image.normalize()
        # if crop and self.crop_xxyy:
            # crop = self.crop_xxyy[idx]
            # crop = crop[0]*500//640, crop[1]*500//640, crop[2]*400//512, crop[3]*400//512
            # crop = (max(crop[0]-120//2, 0), max(crop[1]-120//2, 0), max(crop[2] - 112//2, 0), max(crop[3] - 112//2, 0))
        # else:
        #     crop = (0, 0, 0, 0)
        # image = (image.crop(crop, mode='lrtb') if seg else
        if not seg:
            crop_ratio_w = 1/((500 - 360) / 2 / 500)
            crop_ratio_h = 1/((400 - 288) / 2 / 400)
            x = int(image.shape[3]//crop_ratio_w)
            y = int(image.shape[2]//crop_ratio_h)
            image = image.crop((x, x, y, y), mode='lrtb')
            shape_ratio = image.shape[-2] / image.shape[-1]
            if shape_ratio != 288/360:
                if shape_ratio > 288/360:
                    new_w = int(image.shape[-2] / (288/360))
                    image = image.resize((image.shape[-2], new_w))
                else:
                    new_h = int(image.shape[-1] * (288/360))
                    image = image.resize((new_h, image.shape[-1]))
            # image = image.resize((400, 500)).crop((200, 250, 288, 360), mode='uvhw', center=True)
            if crop:
                image[(image.sum(1, keepdim=True) == 0).repeat(1, 3, 1, 1)] = 0.5
        h, w = image.shape[-2:]
        if self.resize_and_crop:
            min_scale = min(h / self.load_size[0], w / self.load_size[1])
            new_h = int(h / min_scale)
            new_w = int(w / min_scale)
            l = max(0, (new_w - self.load_size[1]) // 2)
            r = new_w - self.load_size[1] - l
            t = max(0, (new_h - self.load_size[0]) // 2)
            b = new_h - self.load_size[0] - t
            if seg:
                return image.resize((new_h, new_w), keep_ratio=True, mode='nearest').crop((l, r, t, b), mode='lrtb')
            return image.resize((new_h, new_w), keep_ratio=True).crop((l, r, t, b), mode='lrtb')
        else:
            if seg:
                return image.resize(self.load_size, mode='nearest')
            return image.resize(self.load_size)


class FLIR_reg_night(TestDataset):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_night/"

    def __init__(self, opt):
        self.train_N = self.root_dir + "vis/"
        self.train_T = self.root_dir + "ir_reg_ours/"

        super().__init__(opt)


class FLIR_DAY_SAMPLES(TestDataset):
    """
    Dataset class for the FLIR day dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/FLIR_DAY/"

    def __init__(self, opt):
        self.train_N = self.root_dir + "vis/"
        self.train_T = self.root_dir + "ir/"

        super().__init__(opt)


class FLIR_NIGHT_SAMPLES(TestDataset):
    """
    Dataset class for the FLIR night dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/FLIR_NIGHT/"

    def __init__(self, opt):
        self.train_N = self.root_dir + "vis/"
        self.train_T = self.root_dir + "ir/"

        super().__init__(opt)