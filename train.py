import torch
from torch import nn
from models.resnet import generate_model
from models.triplet_net import Tripletnet



from datasets.spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from datasets.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from datasets.temporal_transforms import Compose as TemporalCompose
from datasets.dataset import get_training_data, get_validation_data, get_inference_data
from datasets.utils import Logger, worker_init_fn, get_lr


model_depth=200
n_classes=1039
n_input_channels=3
resnet_shortcut = 'B'
conv1_t_size = 7 #kernel size in t dim of conv1
conv1_t_stride = 1 #stride in t dim of conv1
no_max_pool = True #max pooling after conv1 is removed
resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value
sample_size = 112
train_crop_min_scale = 0.25
train_crop_min_ratio = 0.75
sample_duration = 16

video_path = '/media/diskstation/datasets/UCF101/UCF101/jpg'
annotation_path = '/media/diskstation/datasets/UCF101/UCF101/json/ucf101_01.json'
dataset='ucf101'
input_type = 'rgb'
file_type = 'jpg'
batch_size=128
n_threads = 4

no_mean_norm=False
no_std_norm=False
mean_dataset = 'kinetics'
value_scale = 1

def get_mean_std(value_scale, dataset):
    assert dataset in ['activitynet', 'kinetics', '0.5']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std

mean, std = get_mean_std(value_scale, dataset=mean_dataset)



def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_data():
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                     no_std_norm)

    spatial_transform = []
    spatial_transform.append(
        RandomResizedCrop(sample_size, (train_crop_min_scale, 1.0),
                        (train_crop_min_ratio, 1.0/train_crop_min_ratio))
        )
    spatial_transform.append(RandomHorizontalFlip())
    spatial_transform.append(ToTensor())
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    temporal_transform.append(TemporalRandomCrop(sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(video_path, annotation_path,
                                   dataset, input_type, file_type,
                                   spatial_transform, temporal_transform)

    train_sampler = None
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
    return train_loader


def load_pretrained_model(model, pretrain_path):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model

    return model




if __name__ == '__main__':
    pretrain_path = '/home/sherry/pretrained/r3d200_KM_200ep.pth'
    model=generate_model(model_depth=model_depth, n_classes=n_classes,
                        n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)
    # print(model)
    print('finished generating model')

    model = load_pretrained_model(model, pretrain_path)
    # print(model)

    tripletnet = Tripletnet(model)
    # print(tripletnet)

    train_loader = get_train_data()
    print(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        if i>3:
            break
        print(i, inputs.shape, targets)
