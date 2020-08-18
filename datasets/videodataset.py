import json
from pathlib import Path

import torch
import torch.utils.data as data
from loader import VideoLoader, BinaryImageLoaderPIL

"""
Pulled from https://github.com/kenshohara/3D-ResNets-PyTorch
"""


class VideoDataset(data.Dataset):

    def __init__(self,
                 data,
                 class_names,
                 split='train',
                 channel_ext={},
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 normalize=None,
                 video_loader=None,
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):

        self.data = data
        self.class_names = class_names
        self.split=split
        self.channel_ext = channel_ext
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.normalize=normalize
        self.image_name_formatter = image_name_formatter

        self.kp_loader = VideoLoader(self.kp_img_name_formatter, image_loader=BinaryImageLoaderPIL)

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type


    def kp_img_name_formatter(self, x):
        return f'image_{x:05d}_kp.png'

    def __loading(self, path, frame_indices, channel_paths=[]):
        clip = self.loader(path, frame_indices)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        for channel in channel_paths:
            channel_clip = self.kp_loader(channel, frame_indices)
            if self.spatial_transform is not None:
                channel_clip = [self.spatial_transform(img) for img in channel_clip]
                clip = [torch.cat((clip[i], channel_clip[i]), dim=0) for i in range(len(clip))]

        clip = [self.normalize(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)
        return clip

    def _get_video_custom_temporal(self, index, temporal_transform=None):
        cur = self.data[index]
        path = cur['video']
        if isinstance(self.target_type, list):
            target = [cur[t] for t in self.target_type]
        else:
            target = cur[self.target_type]

        frame_indices = list(range(1, cur['num_frames'] + 1))
        if temporal_transform is not None:
            frame_indices = temporal_transform(frame_indices)

        channel_paths = []
        for key in self.channel_ext:
            channel_paths.append(cur[key])

        clip = self.__loading(path, frame_indices, channel_paths=channel_paths)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target, path

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
