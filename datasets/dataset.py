"""
Modified from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
from torchvision import get_image_backend

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data.dataloader import default_collate

from loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5
from triplets_loader import TripletsData
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
             file_type, spatial_transform=None,
             temporal_transform=None, target_transform=None):

    assert split in ['train', 'val', 'test']
    assert dataset_name in ['kinetics', 'ucf101']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

    if split == 'train':
        subset = 'training'
        ret_collate_fn = None
    elif split == 'val':
        subset = 'validation'
        ret_collate_fn = collate_fn

    video_path_formatter = (lambda root_path, label, video_id: root_path + '/' +
                        label + '/' + video_id)

    print ('Loading', dataset_name)
    if dataset_name == 'ucf101':
        Dataset = UCF101(video_path, annotation_path, subset, video_path_formatter)
    elif dataset_name == 'kinetics':
        Dataset = Kinetics(video_path, annotation_path, split, video_path_formatter)

    if get_image_backend() == 'accimage':
        from datasets.loader import ImageLoaderAccImage
        loader = VideoLoader(Dataset.image_name_formatter, ImageLoaderAccImage())
    else:
        loader = VideoLoader(Dataset.image_name_formatter)

    data = TripletsData(data = Dataset.get_dataset(),
                        class_names = Dataset.get_idx_to_class_map(),
                        subset=subset,
                        spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform,
                        target_transform=target_transform,
                        video_loader=loader) 
    print('{}_data: {}'.format(split, len(data)))

    return data, ret_collate_fn
