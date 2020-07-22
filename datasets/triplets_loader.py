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


class TripletsData(data.Dataset):

    def __init__(self,
                 data,
                 class_names,
                 split='train',
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        self.data = data
        self.class_names = class_names
        self.split=split
        self.spatial_transform = spatial_transform

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

        self.test_labels = np.array([data[self.target_type] for data in self.data])
        self.label_to_indices = {label: np.where(self.test_labels == label)[0] for label in self.class_names.keys()}
        # print(self.label_to_indices)

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip


    def __getitem__(self, index, negative_sampling='RandomNegativeMining'):
        anchor=self.data[index]
        a_target = anchor[self.target_type]

        if self.split=='train':
            positive = anchor.copy()
        else:
            p_idx = np.random.choice(self.label_to_indices[a_target])
            positive = self.data[p_idx]

        if negative_sampling == 'RandomNegativeMining':
            while True:
                negative_idx = np.random.randint(self.__len__())
                if negative_idx != index: break

        else:
            negative_idx=None
            print("TODO: NOT YET IMPLEMENTED")

        negative = self.data[negative_idx]

        a_path = anchor['video']
        p_path = positive['video']
        n_path = negative['video']

        p_target = positive[self.target_type]
        n_target = negative[self.target_type]

        a_frame_indices = list(range(1, anchor['num_frames'] + 1))
        p_frame_indices = a_frame_indices
        n_frame_indices = list(range(1, negative['num_frames'] + 1))

        a_frame_id = self.anchor_temporal_transform(a_frame_indices)
        p_frame_id = self.positive_temporal_transform(p_frame_indices)
        n_frame_id = self.negative_temporal_transform(n_frame_indices)

        a_clip = self.__loading(a_path, a_frame_id)
        p_clip = self.__loading(p_path, p_frame_id)
        n_clip = self.__loading(n_path, n_frame_id)

        return (a_clip, p_clip, n_clip), (a_target, p_target, n_target)

    # def _get_positive(self, index, target):



    def __len__(self):
        return len(self.data)
