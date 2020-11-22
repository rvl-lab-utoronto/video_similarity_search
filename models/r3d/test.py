from r3d import R3DNet

import torch
import torch.nn as nn


def rand_input(batch_size=1, channels=3, num_frames=32, height=224, width=224):
    frames = torch.randn(batch_size, channels, num_frames, height, width)
    return frames


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def test():
    #args = parse_args()
    #cfg = load_config(args)
    #cfg = get_cfg()
    #cfg.NUM_GPUS = 1

    dim=128
    feature_size = 512
    backbone = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    model = nn.Sequential(backbone,
                          nn.Linear(feature_size, feature_size),
                          nn.ReLU(),
                          nn.Linear(feature_size, dim))

    print(model)

    #cur_device = torch.cuda.current_device()
    #model = model.cuda(device=cur_device)

    frame_list = rand_input(2,3,16,128,128)
    y1 = model(frame_list)
    print('Network output size:    ', y1.size(), '\n')

    #frame_list = rand_input(cfg, 2,3,32,224,224)
    #y = model.forward_no_head(frame_list)
    #print('Network output size without pooling:', y.size(), '\n')

    #frame_list = rand_input(cfg, 3,128,224,224)
    #y_long = model.forward_no_head(frame_list)
    #print('Network output size without pooling:', y_long.size(), '\n')

    #frame_list = rand_input(cfg, 2,3,32,224,224)
    #y2 = model(frame_list)
    #cos_sim = torch.nn.CosineSimilarity(dim=1)
    #sim_y1_y2 = cos_sim(y1, y2)
    #print('Similarity output size:', sim_y1_y2.size())


if __name__ == "__main__":
    test()
