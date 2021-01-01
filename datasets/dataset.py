"""
Modified from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
from torchvision import get_image_backend

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data.dataloader import default_collate

from loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5
from triplets_dataset import TripletsData
from video_dataset import VideoDataset
from ucf101 import UCF101
from kinetics import Kinetics


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


def get_data(split, video_path, annotation_path, dataset_name, input_type,
             file_type, triplets, sample_duration, spatial_transform=None,
             temporal_transform=None, normalize=None, target_transform=None, channel_ext={},
             cluster_path=None, target_type=None, val_sample=1,
             negative_sampling=False, positive_sampling_p=1.0,
             pos_channel_replace=False, modality=False, is_master_proc=True):

    assert split in ['train', 'val', 'test']
    assert dataset_name in ['kinetics', 'ucf101']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

    if split == 'train':
        ret_collate_fn = None
    else: #val, test
        ret_collate_fn = collate_fn

    video_path_formatter = (lambda root_path, label, video_id: root_path + '/' +
                        label + '/' + video_id)

    if dataset_name == 'ucf101':
        Dataset = UCF101(video_path, annotation_path, split, sample_duration,
                        channel_ext, cluster_path,
                        is_master_proc, video_path_formatter, val_sample)

    elif dataset_name == 'kinetics':
        Dataset = Kinetics(video_path, annotation_path, split, sample_duration,
                channel_ext, cluster_path, is_master_proc, video_path_formatter)

    if get_image_backend() == 'accimage':
        from datasets.loader import ImageLoaderAccImage
        loader = VideoLoader(Dataset.image_name_formatter, ImageLoaderAccImage())
        if (is_master_proc):
            print('Image loader:', 'ImageLoaderAccImage')
    else:
        loader = VideoLoader(Dataset.image_name_formatter)
        if (is_master_proc):
            print('Image loader:', 'ImageLoaderPIL')

    cluster_labels = None
    if triplets:
        if (is_master_proc):
            print('Using triplets dataset...')

        if target_type == 'cluster_label':
            cluster_labels = set(Dataset.get_cluster_labels())
        # else:
        #     cluster_labels = None

        # Don't do channel replacements (multiview) for validation
        if split != 'train':
            pos_channel_replace=False

        data = TripletsData(data = Dataset.get_dataset(),
                            class_names = Dataset.get_idx_to_class_map(),
                            cluster_labels = cluster_labels,
                            split=split,
                            channel_ext=channel_ext,
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform,
                            target_transform=target_transform,
                            normalize=normalize,
                            video_loader=loader,
                            target_type=target_type,
                            negative_sampling=negative_sampling,
                            positive_sampling_p=positive_sampling_p,
                            pos_channel_replace=pos_channel_replace,
                            modality=modality)
    else:
        if (is_master_proc):
            print('Using single video dataset...')
        data = VideoDataset(data = Dataset.get_dataset(),
                            class_names = Dataset.get_idx_to_class_map(),
                            split=split,
                            channel_ext=channel_ext,
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform,
                            target_transform=target_transform,
                            normalize=normalize,
                            video_loader=loader,
                            image_name_formatter=Dataset.image_name_formatter)

    if (is_master_proc):
        print('{}_data: {}'.format(split, len(data)))

    return data, (ret_collate_fn, cluster_labels)
