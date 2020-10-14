import json
from pathlib import Path

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

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        self.total_labels = []
        self.get_all_labels()

    def get_all_labels(self):
        for d in self.data:
            self.total_labels.append(d['label'])

    def kp_img_name_formatter(self, x):
        return f'image_{x:05d}_kp.png'

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

        channel_paths = {}
        for key in self.channel_ext:
            channel_paths[key] = cur[key]

        total_clips = self.construct_net_input(self.loader, self.channel_ext, self.spatial_transform, self.normalize, path, frame_indices, channel_paths=channel_paths)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return total_clips, path

    def __getitem__(self, index):
        return self._get_video_custom_temporal(index, self.temporal_transform), index

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


    def construct_net_input(vid_loader, channel_ext, spatial_transform, normalize_fn, path, frame_indices, channel_paths={}):
        total_clips = []
        clip = vid_loader(path, frame_indices)
        if spatial_transform is not None:
            spatial_transform.randomize_parameters()
            clip = [spatial_transform(img) for img in clip]
        clip = [normalize_fn(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)
        total_clips.append(clip)
        for key in channel_paths:
            channel_path = channel_paths[key]
            channel_loader = channel_ext[key][1]

            channel_clip = channel_loader(channel_path, frame_indices)
            if spatial_transform is not None:
                channel_clip = [spatial_transform(img) for img in channel_clip]
            channel_clip = [normalize_fn(img) for img in clip] #DO we need to normalize??
            channel_clip = torch.stack(channel_clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)
            total_clips.append(channel_clip)


        return total_clips