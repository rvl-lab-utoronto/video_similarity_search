import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from loader import VideoLoader, BinaryImageLoaderPIL
from dataset_utils import construct_net_input

"""
Pulled from https://github.com/kenshohara/3D-ResNets-PyTorch
"""


class VideoDataset(data.Dataset):

    def __init__(self,
                 data,
                 class_names,
                 split='train',
                 channel_ext={},
                 modality=False,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 normalize=None,
                 video_loader=None,
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 sample_duration=16):

        self.data = data
        self.class_names = class_names
        self.split=split
        self.channel_ext = channel_ext
        self.modality = modality
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.normalize=normalize
        self.image_name_formatter = image_name_formatter
        self.sample_duration = sample_duration

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = 'label'
        self.total_labels = []
        self.get_all_labels()

    def get_label_to_class_map(self):
        return self.class_names

    def get_total_labels(self):
        return self.total_labels

    def get_all_labels(self):
        for d in self.data:
            self.total_labels.append(d['label'])

    def kp_img_name_formatter(self, x):
        return f'image_{x:05d}_kp.png'

    def get_test_video_frame_indices(self, frame_indices):
        total_frames = len(frame_indices)
        if total_frames - self.sample_duration <= 0: #pad left, only sample one
            sequence = np.arange(1, self.sample_duration)
            seq_idx = np.zeros_like(sequence)
            sequence = sequence[sequence<total_frames]
            seq_idx[-len(sequence)::] = sequence
        else:
            available = total_frames - self.sample_duration
            start = np.expand_dims(np.arange(1, available+1, self.sample_duration),1)
            seq_idx = np.expand_dims(np.arange(self.sample_duration), 0) + start
            seq_idx = seq_idx.flatten()
        return seq_idx


    def _get_video_custom_temporal(self, index, temporal_transform=None):
        cur = self.data[index]
        path = cur['video']
        if isinstance(self.target_type, list):
            target = [cur[t] for t in self.target_type]
        else:
            target = cur[self.target_type]

        frame_indices = list(range(1, cur['num_frames'] + 1))
        
        if self.split=="test":
            frame_indices = self.get_test_video_frame_indices(frame_indices)
        elif temporal_transform is not None:
            frame_indices = temporal_transform(frame_indices)

        channel_paths = {}
        for key in self.channel_ext:
            channel_paths[key] = cur[key]

        clip = construct_net_input(self.loader, self.channel_ext, self.spatial_transform, 
                                    self.normalize, path, frame_indices, 
                                    channel_paths=channel_paths, modality=self.modality,
                                    split='val')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target, path, index

    def __getitem__(self, index):
        return self._get_video_custom_temporal(index, self.temporal_transform)

    def __len__(self):
        return len(self.data)

    def _loading_img_path(self, index, temporal_transform=None):
        cur = self.data[index]
        path = cur['video']

        frame_indices = list(range(1, cur['num_frames'] + 1))
        if temporal_transform is not None:
            frame_indices = temporal_transform(frame_indices)
        image_path = path + '/' + self.image_name_formatter(frame_indices[0])
        return image_path
