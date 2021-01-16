from resnet import generate_model

import torch
import torch.nn as nn


def rand_input(batch_size=1, channels=3, num_frames=16, height=224, width=224):
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

    model=generate_model(model_depth=18,
                    hidden_layer=2048,
                    out_dim=128,
                    n_input_channels=4,
                    shortcut_type='B',
                    conv1_t_size=7,
                    conv1_t_stride=1,
                    no_max_pool=True,
                    widen_factor=1,
                    projection_head=True)

    print(model)

    #cur_device = torch.cuda.current_device()
    #model = model.cuda(device=cur_device)

    frame_list = rand_input(2,4,16,128,128)
    print('inp size:', frame_list.size())
    y1 = model(frame_list)
    print('Network output size:    ', y1.size(), '\n')


if __name__ == "__main__":
    test()
