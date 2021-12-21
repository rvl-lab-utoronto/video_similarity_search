import json
import os
import numpy as np
from pathlib import Path
import itertools

def kp_img_name_formatter(x):
    return f'image_{x:05d}_kp.jpg'


def salient_img_name_formatter(x):
    return f'image_{x:05d}_sal_fuse.png'

def optical_img_name_formatter(x): #frame000077.jpg
    return f'frame{x:06d}.jpg'


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter, split='train', channel_ext={}, val_sample=1):
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

    if subset == 'training':
        video_ids = list(itertools.chain(*video_groups.values()))
    else: # if validation/test, only select videos from different groups.
        if val_sample is not None:
            video_ids = []
            for name in video_groups:
                video_ids.extend(np.random.choice(video_groups[name], size=val_sample))
        else:
            video_ids = list(itertools.chain(*video_groups.values()))

    video_paths = []
    for id in video_ids:
        annotations.append(data['database'][id]['annotations'])
        label = data['database'][id]['annotations']['label']
        video_paths.append(video_path_formatter(root_path, label, id))

    channel_paths = {}
    optical_video_path_formatter = (lambda root_path, video_id: root_path + '/' + video_id)
    for key in channel_ext:
        channel_ext_path = channel_ext[key][0]
        if key not in channel_paths:
            channel_paths[key]=[]
        for id in video_ids:
            label = data['database'][id]['annotations']['label']
            if key == 'optical_u' or key == 'optical_v':
                vids = id.split('_')
                if vids[1] == 'HandStandPushups':
                    id = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
                channel_paths[key].append(optical_video_path_formatter(channel_ext_path, id))
            else:
                channel_paths[key].append(video_path_formatter(channel_ext_path, label, id))

    return video_ids, video_paths, annotations, channel_paths


class UCF101():

    def __init__(self,
                 root_path,
                 annotation_path,
                 split, #training, ...
                 sample_duration,
                 channel_ext={},
                 cluster_labels=None,
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
        self.cluster_labels = cluster_labels

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
