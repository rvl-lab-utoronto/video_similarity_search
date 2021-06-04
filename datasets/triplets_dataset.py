"""
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
from temporal_transforms import Shuffle


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


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
                 pos_channel_replace=False,
                 sample_duration=None,
                 prob_pos_channel_replace=None,
                 relative_speed_perception=False,
                 local_local_contrast=False,
                 intra_negative=False,
                 modality=False,
                 predict_temporal_ds=False,
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
        self.pos_channel_replace = pos_channel_replace
        self.sample_duration = sample_duration
        self.prob_pos_channel_replace = prob_pos_channel_replace
        self.relative_speed_perception = relative_speed_perception
        self.local_local_contrast = local_local_contrast
        self.intra_negative = intra_negative
        self.modality = modality
        self.predict_temporal_ds = predict_temporal_ds
        self.max_sr = 4
        self.shuffle = Shuffle()

        if temporal_transform is not None:
            self.anchor_temporal_transform = temporal_transform['anchor']
            self.positive_temporal_transform = temporal_transform['positive']
            self.negative_temporal_transform = temporal_transform['negative']
            if self.intra_negative:
                self.intra_neg_temporal_transform = temporal_transform['intra_negative']

            if self.relative_speed_perception:
                self.fast_positive_temporal_transform = temporal_transform['fast_positive']
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

        if self.split == 'train' and self.predict_temporal_ds: #only applied to train split
            ds_label = random.randint(1, 4) #TODO: make it configurable
            a_clip = self._load_clip(anchor, self.anchor_temporal_transform,
                        use_channel_ext=(False if self.pos_channel_replace else True), ds=ds_label)
            p_clip = self._load_clip(positive, self.positive_temporal_transform,
                        pos_channel_replace=self.pos_channel_replace, ds=ds_label)
            return (a_clip, p_clip), (a_target, p_target), ds_label, index


        a_clip = self._load_clip(anchor, self.anchor_temporal_transform,
                use_channel_ext=(False if self.pos_channel_replace else True))
        p_clip = self._load_clip(positive, self.positive_temporal_transform,
                pos_channel_replace=self.pos_channel_replace)

        if self.relative_speed_perception:
            p_fast_clip = self._load_clip(positive, self.fast_positive_temporal_transform,
                    pos_channel_replace=self.pos_channel_replace)
        elif self.local_local_contrast:
            a2_clip = self._load_clip(anchor, self.anchor_temporal_transform,
                    pos_channel_replace=self.pos_channel_replace)
        elif self.intra_negative:
            intra_n_clip = self._load_clip(anchor, self.intra_neg_temporal_transform,
                    pos_channel_replace=self.pos_channel_replace, intra_negative=True) 

        if self.negative_sampling:
            while True:
                negative_idx = np.random.randint(self.__len__())
                if negative_idx != index: break
            negative = self.data[negative_idx]
            n_target = negative[self.target_type]
            n_clip = self._load_clip(negative, self.negative_temporal_transform)
            return (a_clip, p_clip, n_clip), (a_target, p_target, n_target), (index, negative_idx)
        
        elif self.relative_speed_perception:
            return (a_clip, p_clip, p_fast_clip), (a_target, p_target), index

        elif self.local_local_contrast:
            return (a_clip, p_clip, a2_clip), (a_target, p_target), index
        
        elif self.intra_negative:
            return (a_clip, p_clip, intra_n_clip), (a_target, p_target), index

        else:
            return (a_clip, p_clip), (a_target, p_target), index

    def _load_clip(self, data, temporal_transform, use_channel_ext=True, pos_channel_replace=False, intra_negative=False, ds=1):
        path = data['video']
        frame_indices = list(range(1, data['num_frames'] + 1))
        if self.predict_temporal_ds:
            total_frame_len = len(frame_indices)
            start_frame = random.randint(1, total_frame_len)
            frame_id = self.get_temporal_ds_frame_indices(self.sample_duration, total_frame_len, start_frame, ds=ds)
        else:
            frame_id = temporal_transform(frame_indices) #EDIT 
        
        if self.intra_negative:
            frame_id = self.shuffle(frame_id)
            # print(frame_id)



        channel_paths = {}
        if use_channel_ext:
            for key in self.channel_ext:
                channel_paths[key] = data[key]

        clip = construct_net_input(self.loader, self.channel_ext,
                self.spatial_transform, self.normalize, path, frame_id,
                channel_paths=channel_paths,
                pos_channel_replace=pos_channel_replace,
                prob_pos_channel_replace=self.prob_pos_channel_replace,
                modality=self.modality)
        return clip

    def __len__(self):
        return len(self.data)

    def get_temporal_ds_frame_indices(self, sample_duration, total_frame_len, start_frame, ds=1):
        frame_indices = []
        for i in range(sample_duration):
            cur_frame_idx = (start_frame + i*ds) % total_frame_len + 1
            frame_indices.append(cur_frame_idx)
        return frame_indices
