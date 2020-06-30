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
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):
    # assert dataset_name in [
    #     'kinetics', 'ucf101', 'hmdb51', 'mit'
    # ]
    # assert input_type in ['rgb', 'flow']
    # assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (lambda root_path, label, video_id: root_path + '/' +
                            label + '/' + video_id)

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
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    # assert dataset_name in [
    #     'kinetics', 'ucf101', 'hmdb51', 'mit'
    # ]
    # assert input_type in ['rgb', 'flow']
    # assert file_type in ['jpg', 'hdf5']
    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path + '/' + label + '/' + video_id)

    # validation_data = VideoDatasetMultiClips(
    #     video_path,
    #     annotation_path,
    #     'validation',
    #     spatial_transform=spatial_transform,
    #     temporal_transform=temporal_transform,
    #     target_transform=target_transform,
    #     video_loader=loader,
    #     video_path_formatter=video_path_formatter)
    val_data = TripletsData(video_path,
                                 annotation_path,
                                 'validation',
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform,
                                 video_loader=loader,
                                 video_path_formatter=video_path_formatter,
                                 ntriplets=ntriplets)



    return val_data, collate_fn


def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       inference_subset,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):
    assert dataset_name in [
        'kinetics', 'ucf101', 'hmdb51', 'mit'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']
    assert inference_subset in ['train', 'val', 'test']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'val':
        subset = 'validation'
    elif inference_subset == 'test':
        subset = 'testing'

    inference_data = VideoDatasetMultiClips(
        video_path,
        annotation_path,
        subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter,
        target_type=['video_id', 'segment'])

    return inference_data, collate_fn
