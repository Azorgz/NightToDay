import json
import os
from dataclasses import dataclass
from os.path import isfile
from pathlib import Path
from typing import Literal, Tuple, Union, Iterable

import torch
import yaml
from torch import device

from Datasets import get_dataloaders


@dataclass
class SegScheduleConfig:
    start_epoch: int
    end_epoch: None | int
    updateGT_D_start_epoch: int
    updateGT_TN_start_epoch: int


@dataclass
class SegConfig:
    num_classes: int
    type: Literal['LETNet', 'Segmentor']
    base_dim: int
    n_layers: int
    training_schedule: SegScheduleConfig
    fusion_first: bool = False


@dataclass
class ThermalPreprocessConfig:
    bins: int
    scene: int


@dataclass
class FusConfig:
    preprocess_thermal: ThermalPreprocessConfig
    hidden_dim: 256
    n_enc_layers: 4
    dropout: 0.25
    n_downscaling: 2
    type: Literal['IAware', 'UResNet']


@dataclass
class GenConfig:
    downscaling: int
    fus: FusConfig
    input_size: int | Tuple[int, int] = 256
    hidden_dim: int = 256
    n_enc_layers: int = 4
    n_shared_layers: int = 2
    n_dec_layers: int = 4
    dropout: float = 0.1
    fusion_first: bool = True


@dataclass
class DiscrConfig:
    base_dim: int = 64
    n_layers: int = 4
    fusion_first: bool = False


@dataclass
class ModelConfig:
    name: str
    names_domains: list[str]
    mode: Literal['train', 'test']
    build_from_checkpoint: bool
    fusion_first: bool
    gen: GenConfig
    discr: DiscrConfig
    seg: SegConfig


@dataclass
class SchedulerConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def keys():
        return list(SchedulerConfig.__dict__.keys())


@dataclass
class TrainConfig:
    checkpoint_dir: str | Path
    checkpoint_freq: int
    checkpoint_save_latest: int
    pedestrian_color: Tuple[Literal['red', 'black', 'green', 'blue'], Literal['red', 'black', 'green', 'blue']]
    split_optimizers: bool
    split_weights: bool
    visualize_dir: str | Path
    visualize_freq: int
    input_size: Tuple[int, int]
    start_epoch: int
    test_freq: int
    resume: bool
    resume_epoch: dict | int | str
    partial_train: dict
    total_epochs: int
    lr_G: float
    betas_G: tuple
    lr_D: float
    betas_D: tuple
    lr_S: float
    betas_S: tuple
    loss_scheduler: SchedulerConfig
    gan_type: Literal['lsgan', 'hinge', 'wgan-gp', 'ralsgan']


@dataclass
class LoaderConfig:
    load_size: Tuple[int, int]
    resize_and_crop: bool
    shuffle: bool
    num_workers: int
    batch_size: int


@dataclass
class DatasetConfig:
    load_size: Tuple[int, int]
    num_classes: int
    resize_and_crop: bool
    name: str
    augmentations: dict
    sampling: float


@dataclass
class DataConfig:
    loader: LoaderConfig
    train_datasets: list[DatasetConfig]
    test_datasets: list[DatasetConfig] = None


@dataclass
class OptImage2ImageGATConfig:
    device: device | list[device]
    model: ModelConfig
    training: TrainConfig
    data: DataConfig


@dataclass
class SemClassMapping:
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


def get_config(path=None) -> OptImage2ImageGATConfig:
    if path is not None and isfile(path):
        with open(path, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        BASE_DIR = Path(__file__).resolve().parent
        path = BASE_DIR / 'configs' / 'conf.yaml'
        with open(path, 'r') as f:
            conf = yaml.safe_load(f)

    # Model Config
    input_size = tuple([conf['model']['gen']['input_size'], conf['model']['gen']['input_size']]) if not isinstance(
        conf['model']['gen']['input_size'], Iterable) else conf['model']['gen']['input_size']
    conf['model']['gen']['input_size'] = input_size
    seg_scheduleConfig = conf['model']['seg']['training_schedule']
    assert seg_scheduleConfig['start_epoch'] >= 0, "segmentation training schedule start_epoch must be >= 0"
    seg_scheduleConfig['end_epoch'] = conf['training']['total_epochs'] if seg_scheduleConfig['end_epoch'] is None else \
    seg_scheduleConfig['end_epoch']
    assert seg_scheduleConfig['end_epoch'] > seg_scheduleConfig[
        'start_epoch'], "segmentation training schedule end_epoch must be None or > start_epoch"
    assert seg_scheduleConfig['end_epoch'] > seg_scheduleConfig['updateGT_D_start_epoch'] >= seg_scheduleConfig[
        'start_epoch'], \
        "segmentation training schedule updateGT_start_epoch must be >= start_epoch and < end_epoch"
    assert seg_scheduleConfig['end_epoch'] > seg_scheduleConfig['updateGT_TN_start_epoch'] >= seg_scheduleConfig[
        'start_epoch'], \
        "segmentation training schedule updateGT_start_epoch must be >= start_epoch and < end_epoch"
    conf['model']['seg']['training_schedule'] = SegScheduleConfig(**seg_scheduleConfig)
    conf['model']['gen']['fus']['preprocess_thermal'] = ThermalPreprocessConfig(**conf['model']['gen']['fus']['preprocess_thermal'])
    conf['model']['gen']['fus'] = FusConfig(**conf['model']['gen']['fus'])
    modelConfig = ModelConfig(name=conf['model']['name'],
                              names_domains=conf['model']['names_domains'],
                              mode=conf['model']['mode'],
                              build_from_checkpoint=conf['model']['build_from_checkpoint'],
                              fusion_first=conf['model']['fusion_first'],
                              gen=GenConfig(**conf['model']['gen']),
                              discr=DiscrConfig(**conf['model']['discr']),
                              seg=SegConfig(**conf['model']['seg']))

    # Training Config
    split_weights = conf['training']['split_weights'] and not conf['model']['build_from_checkpoint']
    trainConfig = TrainConfig(
        checkpoint_dir=os.getcwd() + '/' + conf['training']['checkpoint_dir'] + '/' + modelConfig.name,
        checkpoint_freq=conf['training']['checkpoint_freq'],
        checkpoint_save_latest=conf['training']['checkpoint_save_latest'],
        pedestrian_color=conf['training']['pedestrian_color'],
        split_optimizers=conf['training']['split_optimizers'],
        split_weights=split_weights,
        visualize_dir=os.getcwd() + '/' + conf['training']['visualize_dir'] + '/' + modelConfig.name,
        visualize_freq=conf['training']['visualize_freq'],
        input_size=input_size,
        test_freq=conf['training']['test_freq'],
        start_epoch=conf['training']['start_epoch'],
        resume=conf['training']['resume'],
        resume_epoch=validate_epoch_load(conf['training']['resume_epoch'], n_domains=len(modelConfig.names_domains),
                                         split=split_weights),
        partial_train=validate_partial_train(conf['training']['partial_train'],
                                             n_domains=len(modelConfig.names_domains)) if conf['training'][
            'split_optimizers'] else None,
        total_epochs=conf['training']['total_epochs'],
        lr_G=conf['training']['lr']['G'],
        betas_G=tuple(conf['training']['betas']['G']),
        lr_D=conf['training']['lr']['D'],
        betas_D=tuple(conf['training']['betas']['D']),
        lr_S=conf['training']['lr']['S'],
        betas_S=tuple(conf['training']['betas']['S']),
        loss_scheduler=SchedulerConfig(**conf['training']['loss_scheduler']),
        gan_type=conf['training']['gan_type'])

    # Device Config
    devices_opt = [conf['devices']]
    if len(devices_opt) == 1:
        devices = torch.device(f'cuda:{devices_opt[0]}')
    else:
        device_count = torch.cuda.device_count()
        devices = [torch.device(f'cuda:{i}' if device_count > i else 'cpu') for i in devices_opt]

    # Data Config
    loader_conf = conf['data']['loader']
    loader_conf['load_size'] = input_size
    datasets_conf_train = [DatasetConfig(**d | {'load_size': loader_conf['load_size'],
                                                'num_classes': conf['model']['seg']['num_classes'],
                                                'resize_and_crop': loader_conf['resize_and_crop']})
                           for d in conf['data']['train_datasets']]
    datasets_conf_test = [DatasetConfig(**d | {'load_size': loader_conf['load_size'],
                                               'num_classes': conf['model']['seg']['num_classes'],
                                               'resize_and_crop': loader_conf['resize_and_crop']})
                          for d in conf['data']['test_datasets']] if 'test_datasets' in conf['data'] else None
    data = DataConfig(loader=LoaderConfig(**loader_conf),
                      train_datasets=datasets_conf_train,
                      test_datasets=datasets_conf_test)
    return OptImage2ImageGATConfig(device=devices,
                                   model=modelConfig,
                                   training=trainConfig,
                                   data=data)


def validate_epoch_load(epoch_load: Union[str, dict, int], n_domains: int, split: bool) -> dict | str | int:
    if not split:
        assert isinstance(epoch_load, (str, int)), "epoch_load must be 'latest', or an integer."
        return epoch_load
    if isinstance(epoch_load, str):
        if epoch_load.isdigit():
            return {f'G{i}': int(epoch_load) for i in range(n_domains * 2 + 1)} | \
                {f'D{j}': int(epoch_load) for j in range(n_domains)} | \
                {f'S{k}': int(epoch_load) for k in range(n_domains)}
        elif epoch_load.lower() in ['latest', 'last']:
            return ({f'G{i}': epoch_load.lower() for i in range(n_domains * 2 + 1)} |
                    {f'D{j}': epoch_load.lower() for j in range(n_domains)} |
                    {f'S{k}': epoch_load.lower() for k in range(n_domains)})
        else:
            raise ValueError("epoch_load must be a string, or an integer as str.")
    elif isinstance(epoch_load, int):
        return {f'G{i}': int(epoch_load) for i in range(n_domains * 2 + 1)} | \
            {f'D{j}': int(epoch_load) for j in range(n_domains)} | \
            {f'S{k}': int(epoch_load) for k in range(n_domains)}
    elif isinstance(epoch_load, dict):
        ret = ({f'G{i}': None for i in range(n_domains * 2 + 1)} |
               {f'D{j}': None for j in range(n_domains)} |
               {f'S{k}': None for k in range(n_domains)})
        if 'G' in epoch_load:
            if epoch_load['G'] is not None:
                G = int(epoch_load['G']) if not isinstance(epoch_load['G'], str) else epoch_load['G']
                ret.update({f'G{i}': G for i in range(n_domains * 2 + 1)})
        if 'D' in epoch_load:
            if epoch_load['D'] is not None:
                D = int(epoch_load['D']) if not isinstance(epoch_load['D'], str) else epoch_load['D']
                ret.update({f'D{i}': D for i in range(n_domains)})
        if 'S' in epoch_load:
            if epoch_load['S'] is not None:
                S = int(epoch_load['S']) if not isinstance(epoch_load['S'], str) else epoch_load['S']
                ret.update({f'S{i}': S for i in range(n_domains)})
        for key in epoch_load:
            if key in list(ret.keys()):
                ret[key] = epoch_load[key] if isinstance(epoch_load[key], str) else int(epoch_load[key])
        return ret


def validate_partial_train(partial_train: dict | None, n_domains) -> dict | None:
    ret = {f'G': list(range(n_domains * 2 + 1)), f'D': list(range(n_domains)), f'S': list(range(n_domains))}
    if partial_train is None:
        return ret
    else:
        for key, values in partial_train.items():
            if key in ret:
                if isinstance(values, int) and values in ret[key]:
                    ret[key] = [values]
                elif isinstance(values, Iterable):
                    if values is not None and all([v in ret[key] for v in values]):
                        ret[key] = values
                elif values is None:
                    ret[key] = []
    return ret


def build_train_data_from_config():
    """Builds ImageToImageGAT_Dual + optional LossScheduler + segmentation nets."""
    # --- Model creation ---
    model_params = get_config()
    train_dataloaders, test_dataloaders = get_dataloaders(model_params.data)
    return train_dataloaders, test_dataloaders, model_params
