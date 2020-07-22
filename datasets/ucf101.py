import json
import os
import numpy as np
from pathlib import Path
import itertools

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter, split='train'):
    video_groups = {}
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            group = '_'.join(key.split('_')[:-1])
            if group not in video_groups.keys():
                video_groups[group] = []
            video_groups[group].append(key)


    if split == 'train':
        video_ids = list(itertools.chain(*video_groups.values()))
    else:
        video_ids = []
        for name in video_groups:
            video_ids.append(np.random.choice(video_groups[name]))

    video_paths = []
    for id in video_ids:
        annotations.append(data['database'][id]['annotations'])
        if 'video_path' in data['database'][id]:
            video_paths.append(Path(data['database'][id]['video_path']))
        else:
            label = data['database'][id]['annotations']['label']
            video_paths.append(video_path_formatter(root_path, label, id))
    return video_ids, video_paths, annotations


class UCF101():

    def __init__(self,
                 root_path,
                 annotation_path,
                 split, #training, ...
                 sample_duration,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id)
                 ):

        self.split=split
        if split == 'train':
            subset = 'training'
        elif split == 'val':
            subset = 'validation'

        self.dataset, self.idx_to_class_map = self.__make_dataset(
                root_path, annotation_path, subset, video_path_formatter,
                sample_duration)

    def get_dataset(self):
        return self.dataset

    def get_idx_to_class_map(self):
        return self.idx_to_class_map

    def image_name_formatter(self, x):
        return f'image_{x:05d}.jpg'

    def __make_dataset(self, root_path, annotation_path, subset,
            video_path_formatter, sample_duration):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(data, subset, root_path, video_path_formatter, split=self.split)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
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
