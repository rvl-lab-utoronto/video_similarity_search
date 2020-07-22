import json
import os
import numpy as np
from pathlib import Path


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class UCF101():

    def __init__(self,
                 root_path,
                 annotation_path,
                 split, #training, ...
                 sample_duration,
                 is_master_proc=True,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id)
                 ):

        if split == 'train':
            subset = 'training'
        elif split == 'val':
            subset = 'validation'

        self.dataset, self.idx_to_class_map = self.__make_dataset(
                root_path, annotation_path, subset, video_path_formatter,
                sample_duration, is_master_proc)

    def get_dataset(self):
        return self.dataset

    def get_idx_to_class_map(self):
        return self.idx_to_class_map

    def image_name_formatter(self, x):
        return f'image_{x:05d}.jpg'

    def __make_dataset(self, root_path, annotation_path, subset,
            video_path_formatter, sample_duration, is_master_proc):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(data, subset, root_path, video_path_formatter)
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

            num_frames = segment[1] - 1
            if num_frames == 0:
                if (is_master_proc):
                    print ('empty folder', video_paths[i])
                continue
            elif num_frames < sample_duration:
                #print ('disregarding video with num frames = {} < sample duration = {} : {}'.format(num_frames, sample_duration, video_paths[i]))
                continue

            sample = {
                'video': video_path,
                'num_frames': num_frames,
                'label': label_id
            }
            dataset.append(sample)
        dataset = np.array(dataset)
        return dataset, idx_to_class
