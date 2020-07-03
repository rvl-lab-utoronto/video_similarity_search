"""
Created by Sherry Chen on Jul 3, 2020
Load training and validation data and apply temporal/spatial transformation
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 TemporalEndCrop, TemporalBeginCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from utils import Logger, worker_init_fn, get_lr
from dataset import get_training_data, get_validation_data, get_inference_data


sample_size = 112
train_crop_min_scale = 0.25
train_crop_min_ratio = 0.75
sample_duration = 16
n_val_samples = 3 # number of validation samples for each activity

video_path = '/media/diskstation/datasets/UCF101/jpg'
annotation_path = '/media/diskstation/datasets/UCF101/json/ucf101_01.json'
dataset='ucf101'
input_type = 'rgb'
file_type = 'jpg'
batch_size= 16
n_threads = 4

no_mean_norm=False
no_std_norm=False
mean_dataset = 'kinetics'
value_scale = 1

ntriplets = 1000
ntesttriplets = 100
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

mean, std = get_mean_std(value_scale, dataset=mean_dataset)



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


def get_train_data():
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                     no_std_norm)

    spatial_transform = []
    spatial_transform.append(
        RandomResizedCrop(sample_size, (train_crop_min_scale, 1.0),
                        (train_crop_min_ratio, 1.0/train_crop_min_ratio))
        )
    spatial_transform.append(RandomHorizontalFlip())
    spatial_transform.append(ToTensor())
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    TempTransform = {}
    #anchor
    begin_temporal_transform = []
    begin_temporal_transform.append(TemporalBeginCrop(sample_duration))
    begin_temporal_transform = TemporalCompose(begin_temporal_transform)
    TempTransform['anchor'] = begin_temporal_transform

    #positive
    end_temporal_transform = []
    end_temporal_transform.append(TemporalEndCrop(sample_duration))
    end_temporal_transform = TemporalCompose(end_temporal_transform)
    TempTransform['positive'] = end_temporal_transform

    #negative
    temporal_transform = []
    temporal_transform.append(TemporalRandomCrop(sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)
    TempTransform['negative'] = temporal_transform

    train_data = get_training_data(video_path, annotation_path,
                                   dataset, input_type, file_type,
                                   ntriplets,
                                   spatial_transform, TempTransform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
    return train_data, train_loader


def get_val_data():
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                     no_std_norm)
    spatial_transform = [
        Resize(sample_size),
        CenterCrop(sample_size),
        ToTensor()
    ]

    spatial_transform.extend([ScaleValue(value_scale), normalize])
    spatial_transform = Compose(spatial_transform)
    #
    # temporal_transform = []
    # # if sample_t_stride > 1:
    # #     temporal_transform.append(TemporalSubsampling(sample_t_stride))
    # temporal_transform.append(TemporalEvenCrop(sample_duration, n_val_samples))
    # temporal_transform = TemporalCompose(temporal_transform)

    TempTransform = {}
    #anchor
    begin_temporal_transform = []
    begin_temporal_transform.append(TemporalBeginCrop(sample_duration))
    begin_temporal_transform = TemporalCompose(begin_temporal_transform)
    TempTransform['anchor'] = begin_temporal_transform

    #positive
    end_temporal_transform = []
    end_temporal_transform.append(TemporalEndCrop(sample_duration))
    end_temporal_transform = TemporalCompose(end_temporal_transform)
    TempTransform['positive'] = end_temporal_transform

    #negative
    temporal_transform = []
    temporal_transform.append(TemporalRandomCrop(sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)
    TempTransform['negative'] = temporal_transform


    val_data, collate_fn = get_validation_data(video_path, annotation_path, dataset,
                                input_type, file_type,
                                ntesttriplets,
                                spatial_transform,
                                TempTransform)
    #TODO: investigate torch.utils.data.distributed.DistributedSampler()
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size = (batch_size // n_val_samples),
                                            # batch_size = batch_size,
                                            shuffle=False,
                                            num_workers=n_threads,
                                            pin_memory=True,
                                            sampler=val_sampler,
                                            worker_init_fn=worker_init_fn
                                            # collate_fn=collate_fn
                                            )

    return val_data, val_loader




if __name__ == '__main__':
    # train_loader = get_train_data()
    # print(train_loader)
    # for i, (inputs, targets) in enumerate(train_loader):
    #     if i>3:
    #         break
    #     print(i, inputs.shape, targets)
    train_data, train_loader = get_train_data()

    # for i, data in enumerate(train_loader):
    #     a, b = data
    #     x, y, z = a
    #     print(x.shape)
