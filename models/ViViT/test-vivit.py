
from vivit import ViViT

import torch
import torch.nn as nn

import numpy as np


def rand_input(batch_size=1, channels=3, num_frames=16, height=224, width=224):
    frames = torch.randn(batch_size, channels, num_frames, height, width)
    return frames


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def test():


    #def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
    #             emb_dropout = 0., scale_dim = 4, ):
    #model = ViViT(image_size=224, patch_size=16, num_classes=400,
    #        num_frames=32, in_channels=3).cuda()

    #ViViT-Base factorized encoder model
    #model = ViViT(image_size=224, patch_size=16, num_classes=400,
    #        num_frames=32, dim=768, depth=12, heads=12, in_channels=3).cuda()

    #ViViT-Large factorized encoder model
    #model = ViViT(image_size=224, patch_size=16, num_classes=400,
    #        num_frames=32, dim=1024, depth=24, heads=16, in_channels=3).cuda()

    #ViViT-Base factorized encoder model - slic version - replace classifier with projection head
    model = ViViT(image_size=128, patch_size=16, num_classes=400,
            num_frames=32, dim=768, depth=12, heads=12, pool='mean',
            in_channels=3, prepend_cls_token=False, projection_head=True,
            classifier_head=False, proj_head_hidden_layer= 2048,
            out_dim = 128).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000

    #print(model)
    print('Trainable Parameters: %.3fM' % parameters)

    # batch size, frames, channels, width, height
    img = torch.ones([2, 32, 3, 128, 128]).cuda()
    out = model(img)
    print("Shape of out :", out.shape)      # [B, num_classes]



if __name__ == "__main__":
    test()
