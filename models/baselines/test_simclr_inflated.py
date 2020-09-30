import copy

import torch
import torchvision

from inflate_src.i3res import I3ResNet
from simclr_pytorch.resnet_wider import resnet50x1, resnet50x2, resnet50x4


def test():
    num_frames = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sd = 'simclr_pytorch/resnet50-1x.pth'
    resnet = resnet50x1()
    sd = torch.load(sd, map_location='cpu')
    resnet.load_state_dict(sd['state_dict'])
    print ('Finished loading simclr-pretrained resnet50 (2D)')

    i3resnet = I3ResNet(copy.deepcopy(resnet), num_frames)
    i3resnet = i3resnet.to(device)
    print ('Constructed inflated simclr-pretrained resnet50')

    frames = torch.randn(1, 3, num_frames, 224, 224)
    frames = frames.to(device)
    out3d = i3resnet(frames)
    print(out3d.size())


if __name__ == "__main__":
    test()
