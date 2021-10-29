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
from hmdb51 import HMDB51
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


def read_cluster_labels(cluster_path, is_master_proc):
    if not cluster_path:
        if is_master_proc:
            print('cluster_path not defined....')
        return None

    if type(cluster_path) is not tuple:
        with open(cluster_path, 'r') as f:
            cluster_labels = f.readlines()
        cluster_labels = [int(id.replace('\n', '')) for id in cluster_labels]
        if is_master_proc:
            print('retrieved {} cluster id from file: {}'.format(len(cluster_labels), cluster_path))
        return cluster_labels

    else:

        with open(cluster_path[0], 'r') as f:
            cluster_labels1 = f.readlines()
        with open(cluster_path[1], 'r') as f:
            cluster_labels2 = f.readlines()

        cluster_labels = [(int(id1.replace('\n', '')), int(id2.replace('\n', '')))
                            for id1, id2 in zip(cluster_labels1, cluster_labels2)]
        if is_master_proc:
            print('retrieved {} cluster id from files:{},{}'.format(len(cluster_labels), cluster_path[0],
                                                                    cluster_path[1]))
        return cluster_labels


def get_data(split, video_path, annotation_path, dataset_name, input_type,
             file_type, triplets, sample_duration, skip_rate, spatial_transform=None,
             temporal_transform=None, normalize=None, target_transform=None, channel_ext={},
             cluster_path=None, target_type=None, val_sample=1,
             negative_sampling=False, positive_sampling_p=1.0,
             pos_channel_replace=False, prob_pos_channel_replace=None, modality=False, predict_temporal_ds=False, 
             relative_speed_perception=False, 
             local_local_contrast=False, intra_negative=False, flow_only=False, is_master_proc=True):


    '''
    this is to differentiate out validation method and CoCLR validation method. 
    During training, we set split=='val' and randomly sample 1 subclip for evaluation. 
    After training, we set split=='test' to average all possible sequences for each clip to do evaluation
    '''


    assert split in ['train', 'val', 'test']
    assert dataset_name in ['kinetics', 'ucf101', 'hmdb51']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

    if flow_only:
        assert triplets == False, 'flow-only inputs only supported for non-triplet loader'

    if split == 'train':
        ret_collate_fn = None
    else: #val, test
        ret_collate_fn = collate_fn

    video_path_formatter = (lambda root_path, label, video_id: root_path + '/' +
                        label + '/' + video_id)

    # read cluster labels, may return list or list of tuples for multiple assignments
    cluster_labels = read_cluster_labels(cluster_path, is_master_proc)

    if dataset_name == 'ucf101':
        split2 = split if split!='test' else 'val'
        Dataset = UCF101(video_path, annotation_path, split2, sample_duration*skip_rate,
                        channel_ext, cluster_labels,
                        is_master_proc, video_path_formatter, val_sample)

    elif dataset_name == 'hmdb51':
        split2 = split if split!='test' else 'val'
        Dataset = HMDB51(video_path, annotation_path, split2, sample_duration*skip_rate,
                        channel_ext, cluster_labels,
                        is_master_proc, video_path_formatter, val_sample)

    elif dataset_name == 'kinetics':
        Dataset = Kinetics(video_path, annotation_path, split, sample_duration*skip_rate,
                channel_ext, cluster_labels, is_master_proc, video_path_formatter)

    if get_image_backend() == 'accimage':
        from datasets.loader import ImageLoaderAccImage
        loader = VideoLoader(Dataset.image_name_formatter, ImageLoaderAccImage())
        if (is_master_proc):
            print('Image loader:', 'ImageLoaderAccImage')
    else:
        loader = VideoLoader(Dataset.image_name_formatter)
        if (is_master_proc):
            print('Image loader:', 'ImageLoaderPIL')

    if triplets:
        if (is_master_proc):
            print('==> Using TripletsData loader...')

        # Don't do channel replacements (multiview) for validation
        if split != 'train':
            pos_channel_replace=False

        data = TripletsData(data = Dataset.get_dataset(),
                            class_names = Dataset.get_idx_to_class_map(),
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
                            prob_pos_channel_replace=prob_pos_channel_replace,
                            modality=modality,
                            sample_duration=sample_duration,
                            predict_temporal_ds=predict_temporal_ds,
                            relative_speed_perception=relative_speed_perception,
                            local_local_contrast=local_local_contrast,
                            intra_negative=intra_negative)
    else:
        if (is_master_proc):
            print('==> Using VideoDataset Loader...')
        data = VideoDataset(data = Dataset.get_dataset(),
                            class_names = Dataset.get_idx_to_class_map(),
                            split=split,
                            channel_ext=channel_ext,
                            modality=modality,
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform,
                            target_transform=target_transform,
                            normalize=normalize,
                            video_loader=loader,
                            image_name_formatter=Dataset.image_name_formatter,
                            sample_duration=sample_duration,
                            flow_only=flow_only)

    if (is_master_proc):
        print('{}_data: {}'.format(split, len(data)))

    return data, ret_collate_fn
