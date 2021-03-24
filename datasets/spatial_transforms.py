"""
Modified from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
import random
import numpy as np
import cv2
import torchvision
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image


class Compose(transforms.Compose):

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        pass


class ScaleValue(object):

    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):

    def randomize_parameters(self):
        pass


class Scale(transforms.Scale):

    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self,
                 size,
                 crop_position=None,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.size = size
        self.crop_position = crop_position
        self.crop_positions = crop_positions

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.size, self.size)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - self.size
        elif self.crop_position == 'bl':
            i = image_height - self.size
            j = 0
        elif self.crop_position == 'br':
            i = image_height - self.size
            j = image_width - self.size

        img = F.crop(img, i, j, h, w)

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(
            self.size, self.crop_position, self.randomize)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()


class MultiScaleCornerCrop(object):

    def __init__(self,
                 size,
                 scales,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
                 interpolation=Image.BILINEAR):
        self.size = size
        self.scales = scales
        self.interpolation = interpolation
        self.crop_positions = crop_positions

        self.randomize_parameters()

    def __call__(self, img):
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * self.scale)
        self.corner_crop.size = crop_size

        img = self.corner_crop(img)
        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]

        self.corner_crop = CornerCrop(None, crop_position)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, scales={1}, interpolation={2})'.format(
            self.size, self.scales, self.interpolation)


class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.randomize = True


class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5, p=0.8):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p
        self.apply = False
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.apply = self.random_p < self.p
            self.transform = self.get_params(self.brightness, self.contrast,
                                              self.saturation, self.hue)
            self.randomize = False

        if self.apply:
            return self.transform(img)
        else:
            return img

    def randomize_parameters(self):
        self.randomize = True
        self.random_p = random.random()


class ColorDrop(object):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.apply = False
        self.randomize_parameters()
        self.transform_rgb = transforms.Grayscale(num_output_channels=3)
        self.transform_gray = transforms.Grayscale(num_output_channels=1)

    def __call__(self, img):
        if self.randomize:
            self.apply = self.random_p < self.p
            self.randomize = False

        if self.apply:
            if img.mode == "L":
                return self.transform_gray(img)
            else:
                return self.transform_rgb(img)
        else:
            return img

    def randomize_parameters(self):
        self.randomize=True
        self.random_p = random.random()


class PickFirstChannels(object):

    def __init__(self, n):
        self.n = n

    def __call__(self, tensor):
        return tensor[:self.n, :, :]

    def randomize_parameters(self):
        pass


class RandomApply(transforms.RandomApply):

    def __init__(self, transforms, p=0.8):
        super().__init__(transforms, p)

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class GaussianBlur(transforms.Lambda):
    def __init__(self, p=0.2):
        # super().__init__(None)
        self.p=p
        self.randomize_parameters()

    def __call__(self, img):
        if self.random_p < self.p:
            image = np.array(img)
            image_blur = cv2.GaussianBlur(image,(15,15),2)
            return image_blur
        return img

    def randomize_parameters(self):
        self.random_p = random.random()
