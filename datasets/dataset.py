"""
Modified from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
from torchvision import get_image_backend

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from videodataset import VideoDataset
from videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5
from triplets_loader import TripletsData

def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      ntriplets=None,
                      triplets = True,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):


    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (lambda root_path, label, video_id: root_path + '/' +
                            label + '/' + video_id)

    if triplets:
        training_data = TripletsData(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter,
                                     ntriplets=ntriplets)

    print('training_data', len(training_data))
    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        ntriplets=None,
                        triplets = True,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path + '/' + label + '/' + video_id)

    if triplets:
        val_data = TripletsData(video_path,
                                 annotation_path,
                                 'validation',
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform,
                                 video_loader=loader,
                                 video_path_formatter=video_path_formatter,
                                 ntriplets=ntriplets)


    print('val_data:{}'.format(len(val_data)))
    return val_data, collate_fn
