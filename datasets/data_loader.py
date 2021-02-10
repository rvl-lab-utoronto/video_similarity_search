"""
Created by Sherry Chen on Jul 3, 2020
Load training and validation data and apply temporal/spatial transformation
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter, ColorDrop,
                                PickFirstChannels, RandomApply, GaussianBlur)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 TemporalEndCrop, TemporalBeginCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_data
from loader import VideoLoader, BinaryImageLoaderPIL
import datasets.ucf101
import datasets.kinetics

train_crop_min_scale = 0.25
train_crop_min_ratio = 0.75

input_type = 'rgb'
file_type = 'jpg'

no_mean_norm=False
no_std_norm=False
mean_dataset = 'kinetics'
value_scale = 1


# Worker init function passed to pytorch data loader
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


# Return mean and std deviation used for normalization
def get_mean_std(value_scale, dataset):
    if dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std


# Return normalization function used per image in a video
def get_normalize_method(mean, std, no_mean_norm, no_std_norm, num_channels=3, is_master_proc=True):
    if no_mean_norm:
        mean = [0, 0, 0]
    elif no_std_norm:
        std = [1, 1, 1]

    extra_num_channel = num_channels-3
    mean.extend([0] * extra_num_channel)
    std.extend([1] * extra_num_channel)
    if (is_master_proc):
        print('Normalize mean:{}, std:{}'.format(mean, std))
    return Normalize(mean, std)


# Return spatial transformations used per image in a video
def build_spatial_transformation(cfg, split, is_master_proc=True):
    mean, std = get_mean_std(value_scale, dataset=cfg.TRAIN.DATASET)
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                         no_std_norm, num_channels=cfg.DATA.INPUT_CHANNEL_NUM, is_master_proc=is_master_proc)

    if split == 'train':
        spatial_transform = []
        spatial_transform.append(
            RandomResizedCrop(cfg.DATA.SAMPLE_SIZE, (train_crop_min_scale, 1.0),
                            (train_crop_min_ratio, 1.0/train_crop_min_ratio))
            )
        spatial_transform.append(RandomHorizontalFlip(p=0.5))

        spatial_transform.append(RandomApply([ColorJitter()], p=0.8))
        spatial_transform.append(ColorDrop(p=0.2))
        spatial_transform.append(GaussianBlur(p=0.2))
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
    normalize = Compose([normalize])

    return spatial_transform, normalize


# Return transformation transformations used per set of video frames
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


# Return dictionary of channel extension information, with each key containing
# an array: [<mask root path>, VideoLoader object for loading the mask]
def get_channel_extention(cfg):
    channel_ext = {}

    assert cfg.TRAIN.DATASET in ['kinetics', 'ucf101']

    if cfg.TRAIN.DATASET == 'ucf101':
        kp_img_name_formatter = datasets.ucf101.kp_img_name_formatter
        salient_img_name_formatter = datasets.ucf101.salient_img_name_formatter
        optical_img_name_formatter = datasets.ucf101.optical_img_name_formatter

    elif cfg.TRAIN.DATASET == 'kinetics':
        kp_img_name_formatter = datasets.kinetics.kp_img_name_formatter
        salient_img_name_formatter = datasets.kinetics.salient_img_name_formatter

    for channel_extension in cfg.DATASET.CHANNEL_EXTENSIONS.split(','):
        if channel_extension == 'keypoint':
            channel_ext['keypoint'] = [cfg.DATASET.KEYPOINT_PATH,
                                            VideoLoader(kp_img_name_formatter, image_loader=BinaryImageLoaderPIL)]
        elif channel_extension == 'salient':
            channel_ext['salient'] = [cfg.DATASET.SALIENT_PATH,
                                           VideoLoader(salient_img_name_formatter, image_loader=BinaryImageLoaderPIL)]
        elif channel_extension == 'optical_u':
            channel_ext['optical_u'] = [cfg.DATASET.OPTICAL_U_PATH,
                                           VideoLoader(optical_img_name_formatter, image_loader=BinaryImageLoaderPIL)]

    return channel_ext


# Return a pytorch DataLoader
def build_data_loader(split, cfg, is_master_proc=True, triplets=True,
                      negative_sampling=False, req_spatial_transform=None,
                      req_train_shuffle=None, val_sample=1, drop_last=True):

    # ==================== Transforms and parameter Setup ======================

    assert split in ['train', 'val', 'test']
    assert (cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0)

    # Get spatial transforms and overwrite with req_spatial_transform if specified
    spatial_transform, normalize = build_spatial_transformation(cfg, split, is_master_proc=is_master_proc)
    if req_spatial_transform is not None:
        spatial_transform = req_spatial_transform
        if (is_master_proc):
            print('Using requested spatial transforms')

    # Get temporal transforms
    TempTransform = build_temporal_transformation(cfg, triplets)

    # Get input channel extension (e.g. salient object mask, keypoint channel)
    # dictionary and assert that the specified input_channel_num is valid
    channel_ext = {}
    if (triplets and cfg.DATASET.POS_CHANNEL_REPLACE and split == 'train') or not cfg.DATASET.POS_CHANNEL_REPLACE:
        channel_ext = get_channel_extention(cfg)
        assert (cfg.DATASET.POS_CHANNEL_REPLACE or len(channel_ext) + 3 == cfg.DATA.INPUT_CHANNEL_NUM)
        if (is_master_proc):
            print('Channel ext:', channel_ext)

    # Set the target type and path to clustering information
    if split == 'train':
        target_type = cfg.DATASET.TARGET_TYPE_T

        # Only need cluster labels if sampling triplets
        if triplets:
            cluster_path = cfg.DATASET.CLUSTER_PATH
        else:
            cluster_path = None
    else:
        target_type = cfg.DATASET.TARGET_TYPE_V
        cluster_path = None
    if (is_master_proc):
        print('Target_type for {} split: {}'.format(split, target_type))

    # ============================= Build dataset ==============================

    # Warning about sampling positives from same real label for training (supervised)
    if is_master_proc and split == 'train' and target_type == 'label' and cfg.DATASET.POSITIVE_SAMPLING_P != 1.0:
        print('NOTE: Will sample positives from same real label (SUPERVISED) for training with POSITIVE_SAMPLING_P =',
            cfg.DATASET.POSITIVE_SAMPLING_P)
    elif is_master_proc and split == 'train' and triplets:
        print('Probability of sampling positive from same video: {}'.format(cfg.DATASET.POSITIVE_SAMPLING_P))

    if (is_master_proc):
        print ('Loading', cfg.TRAIN.DATASET, split, 'split...')
    data, (collate_fn, _) = get_data(split, cfg.DATASET.VID_PATH, cfg.DATASET.ANNOTATION_PATH,
                cfg.TRAIN.DATASET, input_type, file_type, triplets,
                cfg.DATA.SAMPLE_DURATION, spatial_transform, TempTransform, normalize=normalize,
                channel_ext=channel_ext, cluster_path=cluster_path,
                target_type=target_type, val_sample=val_sample,
                negative_sampling=negative_sampling,
                positive_sampling_p=cfg.DATASET.POSITIVE_SAMPLING_P,
                pos_channel_replace=cfg.DATASET.POS_CHANNEL_REPLACE,
                prob_pos_channel_replace=cfg.DATASET.PROB_POS_CHANNEL_REPLACE,
                is_master_proc=is_master_proc)
    if (is_master_proc):
        print ('Single video input size:', data[1][0][0].size())


    # ============================ Build DataLoader ============================

    # Use a DistributedSampler if using more than 1 GPU
    sampler = DistributedSampler(data) if cfg.NUM_GPUS > 1 else None
    if is_master_proc:
        if sampler is not None:
            print ('Using distributed sampler')
        else:
            print ('Not using distributed sampler')

    if split == 'train' or split == 'val':
        # shuffle = True when GPU_num=1
        if req_train_shuffle is not None:
            shuffle = req_train_shuffle
        else:
            shuffle=(False if sampler else True)

        if split == 'train':
            if triplets:
                batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
            else:  # if not in train mode can support a larger batch size
                batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS) * 7
        else:
            batch_size = int(cfg.VAL.BATCH_SIZE)
        if is_master_proc:
            print (split, 'batch size for this process:', batch_size)

        # if drop_last == True,
        # Drop the last non-full batch of each workers dataset replica.
        # Note: this hides a bug with all_gather in validation which
        # would occur when the last batch had different sizes across
        # different gpu processes.

        data_loader = torch.utils.data.DataLoader(data,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=cfg.TRAIN.NUM_DATA_WORKERS,
                                                  pin_memory=True,
                                                  sampler=sampler,
                                                  worker_init_fn=worker_init_fn,
                                                  drop_last=drop_last)

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
    return data_loader, (data, sampler)


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config'))

    from m_parser import load_config, arg_parser

    args = arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    cfg = load_config(args)


    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s) per node".format(cfg.NUM_GPUS))
    print('gpu', cfg.NUM_GPUS)


    # train_loader, data = build_data_loader('train', cfg)
    # val_loader, data = build_data_loader('val', cfg)
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import misc.distributed_helper as du_helper

    du_helper.launch_processes('train', cfg, func=test, shard_id=0, NUM_SHARDS=1, ip_address_port="tcp://localhost:8081")
