import json
from pathlib import Path

import torch
import torch.utils.data as data
from loader import VideoLoader

"""
Pulled from https://github.com/kenshohara/3D-ResNets-PyTorch
"""


class VideoDataset(data.Dataset):

    def __init__(self,
                 data,
                 class_names,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        # self.data, self.class_names = self.__make_dataset(
        #     root_path, annotation_path, subset, video_path_formatter)
        self.data = data
        self.class_names = class_names
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type


    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
