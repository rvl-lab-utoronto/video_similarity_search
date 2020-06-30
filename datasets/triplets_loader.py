import pprint
import torch
import torch.utils.data as data
import numpy as np
import random
import os, sys, json, csv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from loader import VideoLoader
from videodataset import get_database, get_class_labels



class TripletsData(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset, #training, ...
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label',
                 ntriplets=None):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

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

        self.triplet_label_file = os.path.join(os.path.dirname(root_path), '{subset}_triplet_labels.csv'.format(subset=subset))
        self.triplet_file = os.path.join(os.path.dirname(root_path), '{subset}_triplets.csv'.format(subset=subset))

        if ntriplets:
            print('making [{}] triplets...'.format(ntriplets))

            self.make_triplet_list(ntriplets)


            print('\ttriplet_label_file:{}'.format(self.triplet_label_file))
            print('\ttriplet_file:{}'.format(self.triplet_file))

        self._gettriplets()


    def __make_dataset(self, root_path, annotation_path, subset, video_path_formatter):
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
            if not os.path.exists(video_path):
                print('not exists', video_path)
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
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

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip
    #
    # def __getitem__(self, index):
    #     path = self.data[index]['video']
    #     if isinstance(self.target_type, list):
    #         target = [self.data[index][t] for t in self.target_type]
    #     else:
    #         target = self.data[index][self.target_type]
    #
    #     frame_indices = self.data[index]['frame_indices']
    #     if self.temporal_transform is not None:
    #         frame_indices = self.temporal_transform(frame_indices)
    #
    #     clip = self.__loading(path, frame_indices)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     return clip, target

    def _gettriplets(self):
        with open(self.triplet_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            lines = [line for line in csv_reader]
        self.triplets = lines
        
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

    def make_triplet_list(self, ntriplets):
        triplets = []
        triplet_labels = []
        for i in range(ntriplets):
            pairs = random.sample(self.data, 2)

            positive = pairs[0]
            anchor = positive.copy()
            negative = pairs[1]

            #anchor
            path = anchor['video']
            target = anchor['label']
            anchor_id = anchor['video_id']
            frame_indices = anchor['frame_indices']
            anchor['frame_indices'] = self.anchor_temporal_transform(frame_indices)
            anchor = json.dumps(anchor)
            # print('anchor_clip path:{}, target:{}, frame_indices:{}, anchor_clip:{}'.format(path, target, anchor['frame_indices'], anchor_clip.size()))

            #positive
            path = positive['video']
            target = positive['label']
            positive_id = positive['video_id']
            frame_indices = positive['frame_indices']
            positive['frame_indices'] = self.positive_temporal_transform(frame_indices)
            positive = json.dumps(positive)
            # print('positive_clip path:{}, target:{}, frame_indices:{}, positive_clip:{}'.format(path, target, positive['frame_indices'], positive_clip.size()))

            #negative
            path = negative['video']
            target = negative['label']
            negative_id = negative['video_id']
            frame_indices = negative['frame_indices']
            negative['frame_indices'] = self.negative_temporal_transform(frame_indices)
            negative = json.dumps(negative)
            # print('negative_clip path:{}, target:{}, frame_indices:{}, negative_clip:{}'.format(path, target, negative['frame_indices'], negative_clip.size()))

            triplet_labels.append([anchor_id, positive_id, negative_id])
            triplets.append([anchor, positive, negative])


        with open(self.triplet_label_file, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(triplet_labels)

        with open(self.triplet_file,  'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(triplets)





#
# if __name__ == '__main__':
#     train_loader = get_train_data()
#     print(train_loader)
#     # train_loader.make_triplet_list(10)
#     # for i, (inputs, targets) in enumerate(train_loader):
#         # if i>3:
#             # break
#         # print(i, inputs.shape, targets)
