import copy

import torch
import torchvision

from inflate_src.i3res import I3ResNet

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def test():
    num_frames = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet = torchvision.models.resnet50(pretrained=True)
    i3resnet = I3ResNet(copy.deepcopy(resnet), num_frames)
    # Discard FC (last kept is avg pool)
    i3resnet = torch.nn.Sequential(*(list(i3resnet.children())[:-1]), Flatten())
    i3resnet = i3resnet.to(device)
    print (i3resnet)

    print ('Constructed inflated imagenet-pretrained resnet50')

    frames = torch.randn(1, 3, num_frames, 224, 224)
    frames = frames.to(device)
    out3d = i3resnet(frames)
    print(out3d.size())


if __name__ == "__main__":
    test()
