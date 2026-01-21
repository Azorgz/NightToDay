import os
from _socket import gethostname

import oyaml
import torch
from kornia.augmentation import Normalize
from kornia.geometry import hflip
from torch import Tensor
from torch.utils.data import Dataset
from ImagesCameras import ImageTensor


class TestDataset(Dataset):
    """
    Base class for test datasets.
    """
    name: str = 'BaseTest'
    root: str = ''
    test_D = ''
    test_T = ''
    test_N = ''

    def __init__(self, opt=None):
        # self.load_size = opt.load_size
        self.test_T = [os.path.join(self.test_T, f) for f in sorted(os.listdir(self.test_T))]
        self.test_N = [os.path.join(self.test_N, f) for f in sorted(os.listdir(self.test_N))]
        assert len(self.test_T) == len(self.test_N), "Number of thermal and night images must be equal."

    def __len__(self):
        return max(len(self.test_D), len(self.test_T))

    def __getitem__(self, idx):
        image_T = self.load_image(self.test_T, idx % len(self.test_T)).GRAY().RGB('gray')
        image_N = self.load_image(self.test_N, idx % len(self.test_N)).match_shape(image_T)
        image_D = self.load_image(self.test_D, idx % len(self.test_D))
        return image_T, image_N, image_D

    def load_image(self, path: list[str], idx: int, **kwargs) -> Tensor:
        """
        Load an image from a given path and return it as a Tensor.
        """
        return ImageTensor(path[idx])


class TrainDataset(Dataset):
    """
    Base class for datasets.
    """
    name: str = 'Base'
    root: str = ''
    train_D = ''
    train_T = ''
    train_N = ''
    TN_edges = ''
    D_edges = ''
    D_seg = ''
    TN_seg = ''
    crop_path = ''
    TL_D = []
    TL_T = []
    TL_N = []

    def __init__(self, opt):
        self.num_classes = opt.num_classes
        self.augmentations = hflip
        self.load_size = opt.load_size
        self.resize_and_crop = opt.resize_and_crop
        opt.sampling = opt.sampling if opt.sampling > 0 else 1
        opt.sampling = opt.sampling if opt.sampling > 1 else int(1/opt.sampling)
        self.train_D = [os.path.join(self.train_D, f) for idx, f
                        in enumerate(sorted(os.listdir(self.train_D))) if idx % opt.sampling == 0]
        self.train_T = [os.path.join(self.train_T, f) for idx, f
                        in enumerate(sorted(os.listdir(self.train_T))) if idx % opt.sampling == 0]
        self.train_N = [os.path.join(self.train_N, f) for idx, f
                        in enumerate(sorted(os.listdir(self.train_N))) if idx % opt.sampling == 0]
        self.TN_edges = [os.path.join(self.TN_edges, f) for idx, f
                         in enumerate(sorted(os.listdir(self.TN_edges))) if idx % opt.sampling == 0]if self.TN_edges else []
        self.D_edges = [os.path.join(self.D_edges, f) for idx, f
                        in enumerate(sorted(os.listdir(self.D_edges))) if idx % opt.sampling == 0] if self.D_edges else []
        self.D_seg = [os.path.join(self.D_seg, f) for idx, f
                      in enumerate(sorted(os.listdir(self.D_seg))) if idx % opt.sampling == 0] if self.D_seg else []
        self.TN_seg = [os.path.join(self.TN_seg, f) for idx, f
                      in enumerate(sorted(os.listdir(self.TN_seg))) if idx % opt.sampling == 0] if self.TN_seg else []
        self.TL_collection = {'green': {'T': [ImageTensor(p).RGB('gray') for p in self.TL_T if 'green' in p],
                                        'N': [ImageTensor(p) for p in self.TL_N if 'green' in p],
                                        'D': [ImageTensor(p) for p in self.TL_D if 'green' in p]},
                              'orange': {'T': [ImageTensor(p).RGB('gray') for p in self.TL_T if 'orange' in p],
                                         'N': [ImageTensor(p) for p in self.TL_N if 'orange' in p],
                                         'D': [ImageTensor(p) for p in self.TL_D if 'orange' in p]},
                              'red': {'T': [ImageTensor(p).RGB('gray') for p in self.TL_T if 'red' in p],
                                      'N': [ImageTensor(p) for p in self.TL_N if 'red' in p],
                                      'D': [ImageTensor(p) for p in self.TL_D if 'red' in p]}}

        assert len(self.train_T) == len(self.train_N), "Number of thermal and night images must be equal."
        assert len(self.train_D) == len(self.D_seg), "Number of day images and segmentation masks must be equal."
        assert len(self.train_D) == len(self.D_edges), "Number of day images and day edges must be equal."
        try:
            with open(self.crop_path, 'r') as f:
                self.crop_xxyy = oyaml.safe_load(f)['crop']
        except:
            self.crop_xxyy = []
        self.normalize = Normalize(0.5, 0.5)

    def __len__(self):
        return max(len(self.train_D), len(self.train_T))

    def __getitem__(self, idx):
        image_D = self.normalize(self.load_image(self.train_D, idx % len(self.train_D), fac=1.1))
        image_T = self.normalize(self.load_image(self.train_T, idx % len(self.train_T), True).GRAY().RGB('gray'))
        image_N = self.normalize(self.load_image(self.train_N, idx % len(self.train_N), True))
        image_D_seg = (self.load_image(self.D_seg, idx % (len(self.D_seg) or 1), seg=True).to_tensor()).to(torch.uint8)
        image_TN_seg = (self.load_image(self.TN_seg, idx % (len(self.TN_seg) or 1), seg=True, crop=True).to_tensor()).to(torch.uint8)
        image_D_edges = self.load_image(self.D_edges, idx % (len(self.D_edges) or 1)).to_tensor()
        image_TN_edges = self.load_image(self.TN_edges, idx % (len(self.TN_edges) or 1), True).to_tensor()
        # if torch.rand(1) < 0.5:
        #     image_D, image_D_seg, image_D_edges = self.augmentations(torch.cat([image_D, image_D_seg, image_D_edges], dim=1)).split([3, 1, 1], 1)
        #     image_T, image_N, image_TN_edges, image_TN_seg = self.augmentations(torch.cat([image_T, image_N, image_TN_edges, image_TN_seg], dim=1)).split([3, 3, 1, 1], 1)
        return image_D, image_T, image_N, image_D_seg, image_TN_seg, image_D_edges, image_TN_edges, self.TL_collection

    def load_image(self, path: list[str], idx: int, crop: bool = False, seg=False, **kwargs) -> Tensor:
        """
        Load an image from a given path and return it as a Tensor.
        """
        if path == []:
            return torch.zeros((1, 3, self.load_size[0], self.load_size[1])) if not seg else torch.ones((1, 1, self.load_size[0], self.load_size[1])) *255
        image = ImageTensor(path[idx]) * 255 if seg else ImageTensor(path[idx])
        if crop and self.crop_xxyy:
            crop = self.crop_xxyy[idx]
        else:
            crop = (0, 0, 0, 0)
        image = image.crop(crop, mode='lrtb')
        h, w = image.shape[-2:]
        if self.resize_and_crop:
            min_scale = min(h/self.load_size[0], w/self.load_size[1])
            new_h = int(h/min_scale)
            new_w = int(w/min_scale)
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


class MasterDataset(Dataset):
    """
    Master dataset class to combine multiple datasets.
    """

    def __init__(self, datasets: list[TrainDataset] | list[TestDataset]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = [sum(self.lengths[:i + 1]) for i in range(len(self.lengths))]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                sample_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[dataset_idx][sample_idx]
        raise IndexError("Index out of range")
