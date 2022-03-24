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
                 multi_partition=False, 
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):

        self.data = data
        self.class_names = class_names
        self.negative_sampling=negative_sampling
        self.positive_sampling_p = positive_sampling_p
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
        self.multi_partition = multi_partition
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
        
        self.gt_target_type='label'

        print('** target type', self.data[0][self.target_type])
        self.DUAL_CLUST_SAMPLING = type(self.data[0][self.target_type]) is tuple

        if self.target_type == 'label':
            self.data_labels = np.array([data[self.target_type] for data in self.data])
            self.label_to_indices = {label: np.where(self.data_labels == label)[0] for label in self.class_names.keys()}
        
        else: #target_type == cluster_labels
            if self.multi_partition:
                self.data_labels1 = np.array([data[self.target_type][0] for data in self.data])
                self.data_labels2 = np.array([data[self.target_type][1] for data in self.data])
                self.cluster_labels1 = set(self.data_labels1)
                self.cluster_labels2 = set(self.data_labels2)
                self.label_to_indices1 = {label: np.where(self.data_labels1 ==
                    label)[0] for label in self.cluster_labels1}
                self.label_to_indices2 = {label: np.where(self.data_labels2 ==
                    label)[0] for label in self.cluster_labels2}
            
            #TODO: FIX THIS FOR DUAL CLUST LABELS
            #set of cluster labels
            elif self.DUAL_CLUST_SAMPLING:
                self.data_labels1 = np.array([data[self.target_type][0] for data in self.data])
                self.data_labels2 = np.array([data[self.target_type][1] for data in self.data])
                self.cluster_labels1 = set(self.data_labels1)
                self.cluster_labels2 = set(self.data_labels2)
                self.label_to_indices1 = {label: np.where(self.data_labels1 ==
                    label)[0] for label in self.cluster_labels1}
                self.label_to_indices2 = {label: np.where(self.data_labels2 ==
                    label)[0] for label in self.cluster_labels2}
            else:
                self.data_labels = np.array([data[self.target_type][0] for data in self.data]) #TODO: edited by sherry
                self.cluster_labels = set(self.data_labels)
                self.label_to_indices = {label: np.where(self.data_labels ==
                    label)[0] for label in self.cluster_labels}

    def __getitem__(self, index):
        anchor=self.data[index]
        a_target = anchor[self.target_type]

        #assert(~(self.split == 'train' and self.target_type == 'label' and self.positive_sampling_p != 1.0))

        pos_from_dualclust_intersec = False

        p_type = np.random.choice(self.positive_types, p=[self.positive_sampling_p, 1-self.positive_sampling_p])
        if (p_type == 'same_inst' and self.split =='train'):
            positive = anchor.copy()

        else: #sample positive from same a_target (of type target_type - 'label' or 'cluster_label')
            if self.multi_partition and self.target_type != "label":
                p_idx = np.random.choice(self.label_to_indices1[a_target[0]])
                while p_idx == index and len(self.label_to_indices1[a_target[0]]) > 1:
                    p_idx = np.random.choice(self.label_to_indices1[a_target[0]])


            elif not self.DUAL_CLUST_SAMPLING:
                clust_choices = self.label_to_indices[a_target]

                p_idx = np.random.choice(clust_choices)
                # Pick different video from anchor if there is more than 1 video with target a_target
                while p_idx == index and len(clust_choices) > 1:
                    p_idx = np.random.choice(clust_choices)

            else: #DUAL CLUST SAMPLING
                clust_choices1 = self.label_to_indices1[a_target[0]]
                clust_choices2 = self.label_to_indices2[a_target[1]]
                clust_choices_intersec = np.intersect1d(clust_choices1, clust_choices2)

                # If there are common videos in both clust assign (note always
                # >= 1 since anchor is in both)
                if clust_choices_intersec.size > 1:
                    clust_choices = clust_choices_intersec
                    pos_from_dualclust_intersec = True
                else:
                    # choose flow clust with P(pos channel replace) else rgb clust
                    if np.random.rand() < self.prob_pos_channel_replace:
                        clust_choices = clust_choices2
                    else:
                        clust_choices = clust_choices1

                p_idx = np.random.choice(clust_choices)
                # Pick different video from anchor if there is more than 1 video with target a_target
                while p_idx == index and len(clust_choices) > 1:
                    p_idx = np.random.choice(clust_choices)

            positive = self.data[p_idx]

        p_target = positive[self.target_type]

        a_gt_target = anchor[self.gt_target_type]
        p_gt_target = positive[self.gt_target_type]

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
            return (a_clip, p_clip, p_fast_clip), (a_target, p_target), index, pos_from_dualclust_intersec

        elif self.local_local_contrast:
            return (a_clip, p_clip, a2_clip), (a_target, p_target), (a_gt_target, p_gt_target), index, pos_from_dualclust_intersec

        elif self.intra_negative:
            return (a_clip, p_clip, intra_n_clip), (a_target, p_target), index, pos_from_dualclust_intersec

        else:
            return (a_clip, p_clip), (a_target, p_target), (a_gt_target, p_gt_target), index, pos_from_dualclust_intersec

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
