import os
import socket
from .DatasetBase import TrainDataset, TestDataset


class LYNRED(TrainDataset):
    """
    Dataset class for the LYNRED night dataset.
    """
    name = 'LYNRED'
    root = '/silenus/PROJECTS/pr-remote-sensing-1a/godeta/datasets/LYNRED/' \
        if 'laptop' not in socket.gethostname() else "/home/godeta/PycharmProjects/TIR2VIS/datasets/LYNRED/"

    def __init__(self, opt):
        self.train_D = self.root + "LYNRED_datasets/trainA"
        self.train_D_T = None
        self.train_T = self.root + "LYNRED_datasets/trainB"
        self.train_N = self.root + "LYNRED_datasets/trainC"
        self.TN_edges = self.root + "LYNRED_datasets/LYNRED_IR_edge_map"
        self.TN_seg = self.root + "LYNRED_datasets/LYNRED_IR_seg_mask"
        self.D_edges = self.root + "LYNRED_datasets/LYNRED_Vis_edge_map"
        self.D_seg = self.root + "LYNRED_datasets/LYNRED_Vis_seg_mask"
        self.TL_D = self.root + "LYNRED_datasets/FG_sample_D_TN/"
        self.TL_D = [self.TL_D + f for f in sorted(os.listdir(self.TL_D))]
        self.TL_T = self.root + "LYNRED_datasets/FG_sample_T/"
        self.TL_T = [self.TL_T + f for f in sorted(os.listdir(self.TL_T))]
        self.TL_N = self.root + "LYNRED_datasets/FG_sample_N/"
        self.TL_N = [self.TL_N + f for f in sorted(os.listdir(self.TL_N))]
        super().__init__(opt)


class LYNRED_DAY(TestDataset):
    """
    Dataset class for the LYNRED day dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/Lynred_day/"

    def __init__(self, opt):
        self.test_N = self.root_dir + "vis/"
        self.test_T = self.root_dir + "ir/"

        super().__init__(opt)


class LYNRED_DAY_SAMPLES(TestDataset):
    """
    Dataset class for the LYNRED day dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/LYNRED_DAY/"

    def __init__(self, opt):
        self.test_N = self.root_dir + "vis/"
        self.test_T = self.root_dir + "ir/"

        super().__init__(opt)


class LYNRED_NIGHT_SAMPLES(TestDataset):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/LYNRED_NIGHT/"

    def __init__(self, opt):
        self.test_N = self.root_dir + "vis/"
        self.test_T = self.root_dir + "ir/"

        super().__init__(opt)
