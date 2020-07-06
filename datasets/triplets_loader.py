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
                 output_path,
                 subset, #training, ...
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label',
                 ntriplets=None,
                 same_instance=False):

        self.data = data
        self.class_names = class_names

        self.spatial_transform = spatial_transform
        self.anchor_temporal_transform = temporal_transform['anchor']
        self.positive_temporal_transform = temporal_transform['positive']
        self.negative_temporal_transform = temporal_transform['negative']

        # self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

        self.output_dir = os.path.join(os.path.dirname(output_path), 'tmp_triplet_files')
        self.triplet_label_file = os.path.join(self.output_dir,
                '{subset}_triplet_labels.csv'.format(subset=subset))
        self.triplet_file = os.path.join(self.output_dir,
                '{subset}_triplets.csv'.format(subset=subset))

        if ntriplets:
            if same_instance:
                print('sampling anchor and positive from the same instance')
            else:
                print('sampling anchor and positive from different instances of the same class')

            print('making [{}] triplets...'.format(ntriplets))
            self.make_triplet_list(ntriplets, same_instance)

        self.triplets = self._gettriplets()


    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip


    def _gettriplets(self):
        with open(self.triplet_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            lines = [line for line in csv_reader]
        return lines
        

    def __getitem__(self, index):
        (anchor, positive, negative) = self.triplets[index]

        anchor = json.loads(anchor)
        positive = json.loads(positive)
        negative = json.loads(negative)

        if isinstance(self.target_type, list):
            anchor_target = [anchor[t] for t in self.target_type]
            positive_target = [positive[t] for t in self.target_type]
            negative_target = [negative[t] for t in self.target_type]
        else:
            anchor_target = anchor[self.target_type]
            positive_target = positive[self.target_type]
            negative_target = negative[self.target_type]

        anchor_clip = self.__loading(anchor['video'], anchor['frame_indices'])
        positive_clip = self.__loading(positive['video'], positive['frame_indices'])
        negative_clip = self.__loading(negative['video'], negative['frame_indices'])

        return (anchor_clip, positive_clip, negative_clip), (anchor_target, positive_target, negative_target)


    def __len__(self):
        return len(self.triplets)


    def make_triplet_list(self, ntriplets, same_instance=True):
        triplets = []
        triplet_labels = []
        for i in range(ntriplets):
            pairs = random.sample(self.data, 2)
            anchor = pairs[0]

            negative = pairs[1]

            #anchor
            path = anchor['video']
            target = anchor['label']
            anchor_id = anchor['video_id']
            frame_indices = anchor['frame_indices']
            anchor['frame_indices'] = self.anchor_temporal_transform(frame_indices)
            # print('anchor path:{}, target:{}, frame_indices:{}'.format(path, target, anchor['frame_indices']))


            #positive
            if same_instance:
                positive = anchor.copy()
            else:
                #get the pool of the same class
                anchor_label = target
                pool = [d for d in self.data if d['label'] == anchor_label]
                positive = random.sample(pool, 1)[0]

            path = positive['video']
            target = positive['label']
            positive_id = positive['video_id']
            frame_indices = positive['frame_indices']
            positive['frame_indices'] = self.positive_temporal_transform(frame_indices)
            # print('positive_clip path:{}, target:{}, frame_indices:{}'.format(path, target, positive['frame_indices']))


            #negative
            path = negative['video']
            target = negative['label']
            negative_id = negative['video_id']
            frame_indices = negative['frame_indices']
            negative['frame_indices'] = self.negative_temporal_transform(frame_indices)
            # print('negative_clip path:{}, target:{}, frame_indices:{}, negative_clip:{}'.format(path, target, negative['frame_indices'], negative_clip.size()))

            anchor = json.dumps(anchor)
            positive = json.dumps(positive)
            negative = json.dumps(negative)

            triplet_labels.append([anchor_id, positive_id, negative_id])
            triplets.append([anchor, positive, negative])

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.triplet_label_file, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(triplet_labels)

        with open(self.triplet_file,  'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(triplets)
        
        print('\ttriplet_label_file:{}'.format(self.triplet_label_file))
        print('\ttriplet_file:{}'.format(self.triplet_file))
