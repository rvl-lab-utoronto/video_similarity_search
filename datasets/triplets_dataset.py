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
                 cluster_labels=None,
                 split='train',
                 channel_ext={},
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 normalize=None,
                 video_loader=None,
                 positive_sampling_p=1.0,
                 negative_sampling=False,
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):

        self.data = data
        self.class_names = class_names
        self.negative_sampling=negative_sampling
        self.positive_sampling_p = positive_sampling_p
        self.cluster_labels = cluster_labels
        self.split = split
        self.channel_ext = channel_ext
        self.spatial_transform = spatial_transform
        self.normalize=normalize
        self.positive_types = ['same_inst', 'diff_inst']

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

        self.data_labels = np.array([data[self.target_type] for data in self.data])

        if self.target_type == 'label':
            self.label_to_indices = {label: np.where(self.data_labels == label)[0] for label in self.class_names.keys()}
        else: #target_type == cluster_labels
            self.label_to_indices = {label: np.where(self.data_labels == label)[0] for label in self.cluster_labels}

    def __getitem__(self, index):
        anchor=self.data[index]
        a_target = anchor[self.target_type]

        #assert(~(self.split == 'train' and self.target_type == 'label' and self.positive_sampling_p != 1.0))

        p_type = np.random.choice(self.positive_types, p=[self.positive_sampling_p, 1-self.positive_sampling_p])
        if (p_type == 'same_inst' and self.split =='train'):
            positive = anchor.copy()

        else: #sample positive from same a_target (of type target_type - 'label' or 'cluster_label')
            p_idx = np.random.choice(self.label_to_indices[a_target]) 

            # Pick different video from anchor if there is more than 1 video with target a_target
            while p_idx == index and len(self.label_to_indices[a_target]) > 1:
                p_idx = np.random.choice(self.label_to_indices[a_target])
            positive = self.data[p_idx]

        p_target = positive[self.target_type]

        a_clip = self._load_clip(anchor, self.anchor_temporal_transform)
        p_clip = self._load_clip(positive, self.positive_temporal_transform)

        if self.negative_sampling:
            while True:
                negative_idx = np.random.randint(self.__len__())
                if negative_idx != index: break
            negative = self.data[negative_idx]
            n_target = negative[self.target_type]
            n_clip = self._load_clip(negative, self.negative_temporal_transform)
            return (a_clip, p_clip, n_clip), (a_target, p_target, n_target)
        else:
            return (a_clip, p_clip), (a_target, p_target)

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
