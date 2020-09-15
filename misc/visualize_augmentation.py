import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn
import os, sys
SOURCE_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOURCE_CODE_DIR)

from datasets.spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter, ColorDrop,
                                PickFirstChannels, RandomApply)
from datasets.data_loader import get_mean_std, get_normalize_method
from torchvision.utils import save_image
import torchvision

no_mean_norm=False
no_std_norm=False

train_crop_min_scale = 0.25
train_crop_min_ratio = 0.75


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path,'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def build_spatial_transformation():
    mean, std = get_mean_std(1, dataset='ucf101')
    normalize = get_normalize_method(mean, std, no_mean_norm,
                                         no_std_norm, num_channels=3)

    spatial_transform = []
    spatial_transform.append(
        RandomResizedCrop(112, (train_crop_min_scale, 1.0),
                        (train_crop_min_ratio, 1.0/train_crop_min_ratio))
        )
    spatial_transform.append(RandomHorizontalFlip(p=0.5))

    # spatial_transform.append(RandomApply([ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)], p=1)) #0.8
    # spatial_transform.append(ColorDrop(p=0.2))
    # spatial_transform.append(torchvision.transforms.Lambda(gaussian_blur))
    spatial_transform.append(ToTensor())
    # spatial_transform.append(normalize) #EDIT
    spatial_transform = Compose(spatial_transform)
    normalize = Compose([normalize])
    return spatial_transform, normalize


def gaussian_blur(img, p=0.5):
    if np.random.uniform() <= p:
        image = np.array(img)
        image_blur = cv2.GaussianBlur(image,(15,15),2)
        new_image = image_blur
        cv2.imwrite('gaussian_blur.png', image_blur)
    else:
        return img
    return new_image

image = 'images/original.png'
image_loader = ImageLoaderPIL()
img = image_loader(image)

spatial_transform, normalize = build_spatial_transformation()
res_img = spatial_transform(img)
# cv2.imwrite('res.png', res_img)
save_image(res_img, 'before_norm.png')

res_img = normalize(res_img)
save_image(res_img, 'test4.png')

# gb_image = gaussian_blur(img)
# save_image(torch.from_numpy(gb_image), 'gaussian_blur.png')
