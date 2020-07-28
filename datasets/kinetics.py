import json
import numpy as np
import os
from pathlib import Path
import csv


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


def parse_database(root_path, annotation_path, split, video_path_formatter):
    video_ids = []
    video_paths = []
    frame_counts = []
    labels = []

    split_annotation_path = os.path.join(annotation_path, '{}.csv'.format(split))
    with open (split_annotation_path, newline='') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            video_ids.append(os.path.basename(row[0]))
            video_paths.append(os.path.join(root_path, row[0]))
            frame_counts.append(int(row[1]))
            labels.append(int(row[2]))

    return video_ids, video_paths, frame_counts, labels


class Kinetics():

    def __init__(self,
                 root_path,
                 annotation_path,
                 split, #training, ...
                 sample_duration,
                 is_master_proc=True,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id)
                 ):

        self.dataset, self.idx_to_class_map = self.__make_dataset(
            root_path, annotation_path, split, video_path_formatter, sample_duration, is_master_proc)

    def get_dataset(self):
        return self.dataset

    def get_idx_to_class_map(self):
        return self.idx_to_class_map

    def image_name_formatter(self, x):
        return f'{x:06d}.jpg'

    def __make_dataset(self, root_path, annotation_path, split, video_path_formatter, sample_duration, is_master_proc):
        video_ids, video_paths, frame_counts, labels = parse_database(
            root_path, annotation_path, split, video_path_formatter)

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
            elif frame_counts[i] < sample_duration:
                # print ('disregarding video with num frames = {} < sample duration = {} : {}'.format(frame_counts[i], sample_duration, video_paths[i]))
                continue

            sample = {
                'video': video_paths[i],
                'num_frames': frame_counts[i],
                'label': labels[i]
            }
            dataset.append(sample)
        dataset = np.array(dataset)
        return dataset, idx_to_class
