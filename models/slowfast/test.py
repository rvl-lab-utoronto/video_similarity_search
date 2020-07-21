from slowfast.utils.parser import load_config, parse_args
#from slowfast.models.build import build_model
from slowfast.models.video_model_builder import SlowFastRepresentation
from slowfast.models.video_model_builder import SlowFast

from slowfast.config.defaults import get_cfg

import torch


def rand_input(cfg, channels=3, num_frames=32, height=224, width=224):
    frames = torch.randn(channels, num_frames, height, width)
    fast_pathway = frames
    slow_pathway = torch.index_select(frames, 1, torch.linspace(0,
        frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA).long(),)
    # unsqueeze for batch size of 1 in this test
    frame_list = [slow_pathway.unsqueeze(0), fast_pathway.unsqueeze(0)]
    print('Slow pathway input size:', frame_list[0].size())
    print('Fast pathway input size:', frame_list[1].size())
    for i in range(len(frame_list)):
        frame_list[i] = frame_list[i].cuda(non_blocking=True)
    return frame_list


def test():
    #args = parse_args()
    #cfg = load_config(args)
    cfg = get_cfg()
    cfg.merge_from_file('configs/Kinetics/SLOWFAST_8x8_R50.yaml')
    cfg.NUM_GPUS = 1

    #model = SlowFast(cfg)
    model = SlowFastRepresentation(cfg)
    #model = build_model(cfg)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    frame_list = rand_input(cfg, 3,32,224,224)
    frame_list_copy = frame_list.copy()
    y = model(frame_list)
    print('Network output size:    ', y.size())
    y = model.forward_no_head(frame_list_copy)
    print('Network output size without pooling:', y.size())

    print()
    frame_list_copy = rand_input(cfg, 3,128,224,224)
    y = model.forward_no_head(frame_list_copy)
    print('Network output size without pooling:', y.size())


if __name__ == "__main__":
    test()
