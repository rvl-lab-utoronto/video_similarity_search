import json
import os
import numpy as np
from pathlib import Path
import itertools

from ucf101 import get_class_labels, get_database

class HMDB51():

    def __init__(self,
                 root_path,
                 annotation_path,
                 split, #training, ...
                 sample_duration,
                 channel_ext={},
                 cluster_path=None,
                 is_master_proc=True,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 val_sample=1
                 ):

        self.split=split
        self.is_master_proc = is_master_proc
        if split == 'train':
            subset = 'training'
        elif split == 'val':
            if self.is_master_proc and val_sample is not None:
                print('Randomly sampling 1 clip from each group for the validation set')
            subset = 'validation'
        else:
            subset = 'test'


        self.channel_ext = channel_ext
        self.cluster_path = cluster_path
        self.cluster_labels = self.read_cluster_labels()

        self.val_sample = val_sample

        self.dataset, self.idx_to_class_map = self.__make_dataset(
                root_path, annotation_path, subset, video_path_formatter,
                sample_duration, is_master_proc)

    def get_dataset(self):
        return self.dataset

    def get_idx_to_class_map(self):
        return self.idx_to_class_map

    def get_cluster_labels(self):
        return self.cluster_labels

    def image_name_formatter(self, x):
        return f'image_{x:05d}.jpg'

    def read_cluster_labels(self):
        if not self.cluster_path:
            if self.is_master_proc:
                print('cluster_path not defined....')
            return None
        with open(self.cluster_path, 'r') as f:
            cluster_labels = f.readlines()
        cluster_labels = [int(id.replace('\n', '')) for id in cluster_labels]
        if self.is_master_proc:
            print('retrieved {} cluster id from file: {}'.format(len(cluster_labels), self.cluster_path))
        return cluster_labels


    def __make_dataset(self, root_path, annotation_path, subset,
            video_path_formatter, sample_duration, is_master_proc):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations, channel_paths = get_database(data, subset, root_path, video_path_formatter, \
                                                        split=self.split, channel_ext=self.channel_ext, val_sample=self.val_sample)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                if (is_master_proc):
                    print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            segment = annotations[i]['segment']
            num_frames = segment[1] - 2 #edit
            if num_frames == 0:
                if (is_master_proc):
                    print ('empty folder', video_paths[i])
                continue
            elif num_frames < 2 * sample_duration:
                #print ('disregarding video with num frames = {} < sample duration = {} : {}'.format(num_frames, sample_duration, video_paths[i]))
                continue

            sample = {
                'video': video_path,
                'num_frames': num_frames,
                'label': label_id,
            }

            if channel_paths:
                for key in channel_paths:
                    sample[key] = channel_paths[key][i]

            if self.cluster_labels:
                cluster_label = self.cluster_labels[len(dataset)-1]
                sample['cluster_label'] = cluster_label

            dataset.append(sample)

        dataset = np.array(dataset)
        return dataset, idx_to_class
