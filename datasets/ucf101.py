import json
import os
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
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 sample_duration
                 ):

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
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
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
            #if not os.path.exists(video_path):
            #    print('not exists', video_path)
            #    continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            if segment[1] - 1 < sample_duration:
                print ('disregarding video with num frames = {} < sample
                duration = {} : {}'.format(segment[1] - 1, sample_duration,
                    video_paths[i]))
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class






    
