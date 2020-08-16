"""
Created by Sherry Chen on Jul 3, 2020
Load training and validation data and apply temporal/spatial transformation
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter, ColorDrop,
                                PickFirstChannels, RandomApply)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 TemporalEndCrop, TemporalBeginCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from data_utils import Logger, worker_init_fn, get_lr
from dataset import get_data


train_crop_min_scale = 0.25
train_crop_min_ratio = 0.75

input_type = 'rgb'
file_type = 'jpg'

no_mean_norm=False
no_std_norm=False
mean_dataset = 'kinetics'
value_scale = 1

distributed=False


def get_mean_std(value_scale, dataset):
    assert dataset in ['activitynet', 'kinetics', '0.5']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

def build_spatial_transformation(cfg, split): #TODO: rewrite the normalize function
    mean, std = get_mean_std(value_scale, dataset=mean_dataset)
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                         no_std_norm)
    if split == 'train':
        spatial_transform = []
        spatial_transform.append(
            RandomResizedCrop(cfg.DATA.SAMPLE_SIZE, (train_crop_min_scale, 1.0),
                            (train_crop_min_ratio, 1.0/train_crop_min_ratio))
            )
        spatial_transform.append(RandomHorizontalFlip())

        spatial_transform.append(RandomApply([ColorJitter()], p=0.8))
        spatial_transform.append(ColorDrop(p=0.2))

        spatial_transform.append(ToTensor())
        # spatial_transform.append(normalize) #EDIT

    else: #val/ test
        spatial_transform = [
            Resize(cfg.DATA.SAMPLE_SIZE),
            CenterCrop(cfg.DATA.SAMPLE_SIZE),
            ToTensor()
        ]
        spatial_transform.extend([ScaleValue(value_scale)])#, normalize])

    spatial_transform = Compose(spatial_transform)
    return spatial_transform


def build_temporal_transformation(cfg, triplets=True):
    if triplets:
        TempTransform = {}
        #anchor
        anchor_temporal_transform = []
        anchor_temporal_transform.append(TemporalBeginCrop(cfg.DATA.SAMPLE_DURATION))
        anchor_temporal_transform = TemporalCompose(anchor_temporal_transform)
        TempTransform['anchor'] = anchor_temporal_transform

        #positive
        positive_temporal_transform = []
        positive_temporal_transform.append(TemporalRandomCrop(cfg.DATA.SAMPLE_DURATION, start_index=cfg.DATA.SAMPLE_DURATION))
        positive_temporal_transform = TemporalCompose(positive_temporal_transform)
        TempTransform['positive'] = positive_temporal_transform

        #negative
        temporal_transform = []
        temporal_transform.append(TemporalRandomCrop(cfg.DATA.SAMPLE_DURATION))
        temporal_transform = TemporalCompose(temporal_transform)
        TempTransform['negative'] = temporal_transform

    else:
        temporal_transform = []
        temporal_transform.append(TemporalCenterCrop(cfg.DATA.SAMPLE_DURATION)) #opt.n_val_samples))
        temporal_transform = TemporalCompose(temporal_transform)
        TempTransform = temporal_transform

    return  TempTransform

def get_channel_extention(cfg):
    channel_ext = {}
    if cfg.DATASET.CHANNEL_EXTENSIONS == 'keypoint':
        channel_ext['keypoint_path'] = cfg.DATASET.KEYPOINT_PATH
    return channel_ext

def build_data_loader(split, cfg, is_master_proc=True, triplets=True):
    assert split in ['train', 'val', 'test']

    spatial_transform = build_spatial_transformation(cfg, split)
    TempTransform = build_temporal_transformation(cfg, triplets)

    channel_ext = get_channel_extention(cfg)
    data, collate_fn = get_data(split, cfg.DATASET.VID_PATH, cfg.DATASET.ANNOTATION_PATH,
                cfg.TRAIN.DATASET, input_type, file_type, triplets,
                cfg.DATA.SAMPLE_DURATION, spatial_transform, TempTransform,
                channel_ext=channel_ext, is_master_proc=is_master_proc)

    if (is_master_proc):
        print ('Single video input size:', data[1][0][0].size())

    assert (cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0)
    sampler = DistributedSampler(data) if cfg.NUM_GPUS > 1 else None
    if (sampler is not None and is_master_proc):
        print ('Using distributed sampler')
        print ('Batch size per gpu:', int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS))

    if split == 'train':
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                                                  shuffle=(False if sampler else True),
                                                  num_workers=cfg.TRAIN.NUM_DATA_WORKERS,
                                                  pin_memory=True,
                                                  sampler=sampler,
                                                  worker_init_fn=worker_init_fn)
    elif split == 'val':
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                                                  shuffle=False,
                                                  num_workers=cfg.TRAIN.NUM_DATA_WORKERS,
                                                  pin_memory=True,
                                                  sampler=sampler,
                                                  worker_init_fn=worker_init_fn
                                                  # collate_fn=collate_fn)
                                                  )
    else: #test split
        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                                                  shuffle=False,
                                                  num_workers=cfg.TRAIN.NUM_DATA_WORKERS,
                                                  pin_memory=True,
                                                  sampler=sampler,
                                                  worker_init_fn=worker_init_fn
                                                  # collate_fn=collate_fn)
                                                  )

    return data_loader, data



if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config'))
    from m_parser import load_config, parse_args

    args = parse_args()
    cfg = load_config(args)

    # train_loader, data = build_data_loader('train', cfg)
    val_loader, data = build_data_loader('val', cfg)
    d = data[0]
    # spatial_transform = build_spatial_transformation(cfg, 'train')
    # TempTransform = build_temporal_transformation(cfg)

    # data, _ = get_data('train', cfg.DATASET.VID_PATH,
    #             cfg.DATASET.ANNOTATION_PATH, cfg.TRAIN.DATASET, input_type, False,
    #             file_type, spatial_transform, TempTransform)
    # a = data[1]
    # print(a[0][0].size())
