import copy
from collections import OrderedDict

import torch
import torchvision

from inflate_src.i3res import I3ResNet

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def test():
    num_frames = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.__dict__['resnet50']()

    sd = 'mocov2_pytorch/moco_v2_200ep_pretrain.pth.tar'
    checkpoint = torch.load(sd, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # return only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print ('Finished loading moco-pretrained resnet50 (2D)')

    i3resnet = I3ResNet(copy.deepcopy(model), num_frames)
    # Discard FC (last kept is avg pool)
    i3resnet = torch.nn.Sequential(*(list(i3resnet.children())[:-1]), Flatten())
    i3resnet = i3resnet.to(device)
    print ('Constructed inflated moco-pretrained resnet50')

    frames = torch.randn(1, 3, num_frames, 224, 224)
    frames = frames.to(device)
    out3d = i3resnet(frames)
    print(out3d.size())


if __name__ == "__main__":
    test()
