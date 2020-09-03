"""
Created by Sherry Chen on Jul 3, 2020
Sample triplet pair and apply temporal/spatial transformation
"""

import pprint
import torch
import torch.utils.data as data
import numpy as np
import random
import os, sys, json, csv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from loader import VideoLoader
from dataset_utils import construct_net_input


class TripletsData(data.Dataset):

    def __init__(self,
                 data,
                 class_names,
                 cluster_id=None,
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
        self.cluster_id = cluster_id
        self.split = split
        self.channel_ext = channel_ext
        self.spatial_transform = spatial_transform
        self.normalize=normalize

        if temporal_transform is not None:
            self.anchor_temporal_transform = temporal_transform['anchor']
            self.positive_temporal_transform = temporal_transform['positive']
            self.negative_temporal_transform = temporal_transform['negative']
        else:
            self.anchor_temporal_transform= None
            self.positive_temporal_transform = None
            self.negative_temporal_transform = None

        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        print('target_type', self.target_type)

        self.data_labels = np.array([data[self.target_type] for data in self.data])
        if self.target_type == 'label':
            self.label_to_indices = {label: np.where(self.data_labels == label)[0] for label in self.class_names.keys()}
        else: #target_type == cluster_id
            self.label_to_indices = {label: np.where(self.data_labels == label)[0] for label in self.cluster_id}

    def __getitem__(self, index):
        anchor=self.data[index]
        a_target = anchor[self.target_type]

        # if positive_sampling == 'SameInstance' and self.split == 'train':
        # if self.split=='train':
        s_positive = anchor.copy()

        d_p_idx = np.random.choice(self.label_to_indices[a_target]) #TODO
        while d_p_idx == index and len(self.label_to_indices[a_target]) > 1:
            d_p_idx = np.random.choice(self.label_to_indices[a_target])

        d_positive = self.data[d_p_idx]

        # else: #validation split, use true label
        #     p_idx = np.random.choice(self.label_to_indices[a_target])
        #     positive = self.data[p_idx]
        #
        #     same_inst_



        # if negative_sampling == 'RandomNegativeMining':
        #     while True:
        #         negative_idx = np.random.randint(self.__len__())
        #         if negative_idx != index: break
        #
        # else:
        #     negative_idx=None
        #     print("TODO: NOT YET IMPLEMENTED")

        # negative = self.data[negative_idx]

        sp_target = s_positive[self.target_type]
        dp_target = d_positive[self.target_type]
        # n_target = negative[self.target_type]

        a_clip = self._load_clip(anchor, self.anchor_temporal_transform)
        sp_clip = self._load_clip(s_positive, self.positive_temporal_transform)
        dp_clip = self._load_clip(d_positive, self.positive_temporal_transform)
        # n_clip = self._load_clip(negative, self.negative_temporal_transform)

        # print('anchor', anchor)
        # print('s_idx:{}, d_idx:{}'.format(index, d_p_idx))
        # print("s_positive:{}, d_positive:{}".format(s_positive, d_positive))

        return (a_clip, sp_clip, dp_clip), (a_target, sp_target, dp_target)

    def _load_clip(self, data, temporal_transform):
        path = data['video']
        frame_indices = list(range(1, data['num_frames'] + 1))
        frame_id = temporal_transform(frame_indices)

        channel_paths = {}
        for key in self.channel_ext:
            channel_paths[key] = data[key]

        clip = construct_net_input(self.loader, self.channel_ext, self.spatial_transform, self.normalize, path, frame_id, channel_paths=channel_paths)
        return clip

    def __len__(self):
        return len(self.data)
