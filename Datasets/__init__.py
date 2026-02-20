import torch
from torch.utils.data import DataLoader
from .DatasetBase import MasterDataset
from .FLIR import FLIR
from .LYNRED import LYNRED, LYNRED_NIGHT_SAMPLES, LYNRED_DAY_SAMPLES

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
    'LYNRED_NIGHT_SAMPLES': LYNRED_NIGHT_SAMPLES,
    'LYNRED_DAY_SAMPLES': LYNRED_DAY_SAMPLES,
}


def collate_ImageTensor_train(batch):
    """
    Custom collate function to handle ImageTensor objects in a batch.
    """
    image_D = [item[0] for item in batch]
    image_D_T = [item[1] for item in batch] if batch[0][1] is not None else None
    image_T = [item[2] for item in batch]
    image_N = [item[3] for item in batch]
    image_D_seg = [item[4] for item in batch]
    image_TN_seg = [item[5] for item in batch]
    image_D_edges = [item[6] for item in batch]
    image_TN_edges = [item[7] for item in batch]
    TL_collection = batch[0][8]
    return {'D': torch.cat(image_D),
            'D_T': torch.cat(image_D_T) if image_D_T is not None else None,
            'T': torch.cat(image_T),
            'N': torch.cat(image_N),
            'seg_D': torch.cat(image_D_seg),
            'seg_TN': torch.cat(image_TN_seg),
            'edges_D': torch.cat(image_D_edges),
            'edges_TN': torch.cat(image_TN_edges),
            'TL_collection': TL_collection}


def collate_ImageTensor_test(batch):
    """
    Custom collate function to handle ImageTensor objects in a batch.
    """
    image_T = [item[0] for item in batch]
    image_N = [item[1] for item in batch]
    return {'T': torch.cat(image_T),
            'N': torch.cat(image_N)}


def get_dataloaders(opt):
    """
    Get dataloaders for the specified datasets.
    """
    train_datasets = opt.train_datasets
    shuffle = opt.loader.shuffle
    num_workers = opt.loader.num_workers
    batch_size = opt.loader.batch_size

    if not isinstance(train_datasets, list):
        train_datasets = [train_datasets]
    datasets_loaded = []
    for dataset_opt in train_datasets:
        if dataset_opt.name in DATASETS:

            datasets_loaded.append(DATASETS[dataset_opt.name](dataset_opt))
        else:
            raise ValueError(
                f"Dataset {dataset_opt.name} is not supported. Choose among: {', '.join(list(DATASETS.keys()))}")

    train_datasets = MasterDataset(datasets_loaded)
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle,
                                  collate_fn=collate_ImageTensor_train)
    #  -------------------------------------------------------------
    test_datasets = opt.test_datasets

    if not isinstance(test_datasets, list):
        test_datasets = [test_datasets]
    datasets_loaded = []
    for dataset_opt in test_datasets:
        if dataset_opt is None:
            continue
        if dataset_opt.name in DATASETS:

            datasets_loaded.append(DATASETS[dataset_opt.name](dataset_opt))
        else:
            raise ValueError(
                f"Dataset {dataset_opt.name} is not supported. Choose among: {', '.join(list(DATASETS.keys()))}")

    test_datasets = MasterDataset(datasets_loaded)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False,
                                 collate_fn=collate_ImageTensor_test)
    return train_dataloader, test_dataloader
