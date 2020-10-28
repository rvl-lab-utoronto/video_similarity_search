import json
import numpy as np
import os
from pathlib import Path
import csv

def kp_img_name_formatter(x):
    return f'{x:06d}_kp.jpg'


def salient_img_name_formatter(x):
    return f'{x:06d}_sal_fuse.png'


def parse_categories(annotation_path):
    idx_to_class = {}
    index = 0

    category_file = os.path.join(annotation_path, 'categories.csv')
    with open (category_file, newline='') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            idx_to_class[index] = row[0]
            index += 1

    return idx_to_class


def parse_database(root_path, annotation_path, split, video_path_formatter, channel_ext={}):
    video_ids = []
    video_paths = []
    frame_counts = []
    labels = []
    channel_paths = {}

    for key in channel_ext:
        if key not in channel_paths:
            channel_paths[key]=[]

    split_annotation_path = os.path.join(annotation_path, '{}.csv'.format(split))
    with open (split_annotation_path, newline='') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            video_ids.append(os.path.basename(row[0]))
            video_paths.append(os.path.join(root_path, row[0]))
            for key in channel_ext:
                channel_paths[key].append(os.path.join(channel_ext[key][0], row[0]))
            frame_counts.append(int(row[1]))
            labels.append(int(row[2]))

    return video_ids, video_paths, frame_counts, labels, channel_paths


class Kinetics():

    def __init__(self,
                 root_path,
                 annotation_path,
                 split, #training, ...
                 sample_duration,
                 channel_ext={},
                 cluster_path=None,
                 is_master_proc=True,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id)
                 ):

        self.is_master_proc = is_master_proc
        self.channel_ext = channel_ext
        self.cluster_path = cluster_path
        self.cluster_labels = self.read_cluster_labels()

        self.dataset, self.idx_to_class_map = self.__make_dataset(
            root_path, annotation_path, split, video_path_formatter, sample_duration, is_master_proc)

    def get_dataset(self):
        return self.dataset

    def get_idx_to_class_map(self):
        return self.idx_to_class_map

    def get_cluster_labels(self):
        return self.cluster_labels

    def image_name_formatter(self, x):
        return f'{x:06d}.jpg'

    def read_cluster_labels(self):
        if not self.cluster_path:
            return None
        with open(self.cluster_path, 'r') as f:
            cluster_labels = f.readlines()
        cluster_labels = [int(id.replace('\n', '')) for id in cluster_labels]
        if self.is_master_proc:
            print('retrieved {} cluster id from file: {}'.format(len(cluster_labels), self.cluster_path))
        return cluster_labels

    def __make_dataset(self, root_path, annotation_path, split, video_path_formatter, sample_duration, is_master_proc):
        video_ids, video_paths, frame_counts, labels, channel_paths = parse_database(
            root_path, annotation_path, split, video_path_formatter, channel_ext=self.channel_ext)

        idx_to_class = parse_categories(annotation_path)

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 10) == 0:
                if (is_master_proc):
                    print('dataset loading [{}/{}]'.format(i, n_videos))

            if (frame_counts[i] == 0):
                if (is_master_proc):
                    print ('empty folder', video_paths[i])
                continue
            elif frame_counts[i] < 2 * sample_duration:
                # print ('disregarding video with num frames = {} < sample duration = {} : {}'.format(frame_counts[i], sample_duration, video_paths[i]))
                continue

            sample = {
                'video': video_paths[i],
                'num_frames': frame_counts[i],
                'label': labels[i]
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
