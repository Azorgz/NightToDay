import torch
from torch.utils.data import DataLoader
from ImagesCameras import ImageTensor

from .DatasetBase import MasterDataset
from .FLIR import FLIR
from .LYNRED import LYNRED

# from .LYNRED import LYNRED

DATASETS = {
    'FLIR': FLIR,
    'LYNRED': LYNRED,
    # 'FLIR_reg_day': FLIR_reg_day,
    # 'FLIR_reg_night': FLIR_reg_night,
    # 'FLIR_day': FLIR_DAY,
    # 'FLIR_night': FLIR_NIGHT,
    # 'FLIR_night_SAMPLES': FLIR_NIGHT_SAMPLES,
    # 'FLIR_day_SAMPLES': FLIR_DAY_SAMPLES,
    # 'LYNRED_day': LYNRED_DAY,
    # 'LYNRED_night': LYNRED_NIGHT,
    # 'LYNRED_night_SAMPLES': LYNRED_NIGHT_SAMPLES,
    # 'LYNRED_day_SAMPLES': LYNRED_DAY_SAMPLES,
}


def collate_ImageTensor(batch):
    """
    Custom collate function to handle ImageTensor objects in a batch.
    """
    image_D = [item[0] for item in batch]
    image_T = [item[1] for item in batch]
    image_N = [item[2] for item in batch]
    image_D_seg = [item[3] for item in batch]
    image_TN_seg = [item[4] for item in batch]
    image_D_edges = [item[5] for item in batch]
    image_TN_edges = [item[6] for item in batch]
    return {'D': torch.cat(image_D),
            'T': torch.cat(image_T),
            'N': torch.cat(image_N),
            'seg_D': torch.cat(image_D_seg),
            'seg_TN': torch.cat(image_TN_seg),
            'edges_D': torch.cat(image_D_edges),
            'edges_TN': torch.cat(image_TN_edges)}


def get_dataloaders(opt):
    """
    Get dataloaders for the specified datasets.
    """
    datasets = opt.datasets
    shuffle = opt.loader.shuffle
    num_workers = opt.loader.num_workers
    batch_size = opt.loader.batch_size

    if not isinstance(datasets, list):
        datasets = [datasets]
    datasets_loaded = []
    for dataset_opt in datasets:
        if dataset_opt.name in DATASETS:

            datasets_loaded.append(DATASETS[dataset_opt.name](dataset_opt))
        else:
            raise ValueError(f"Dataset {dataset_opt.name} is not supported. Choose among: {', '.join(list(DATASETS.keys()))}")

    dataset = MasterDataset(datasets_loaded)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            collate_fn=collate_ImageTensor)
    return dataloader
