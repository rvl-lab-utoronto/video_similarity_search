from slowfast.utils.parser import load_config, parse_args
#from slowfast.models.build import build_model
from slowfast.models.video_model_builder import SlowFastRepresentation
from slowfast.models.video_model_builder import SlowFast

from slowfast.config.defaults import get_cfg

import torch


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

    # `channel` x `num frames` x `height` x `width` 
    frames = torch.randn(3,32,224,224)
    fast_pathway = frames
    slow_pathway = torch.index_select(frames, 1, torch.linspace(0,
        frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA).long(),)
    # unsqueeze for batch size of 1 in this test
    frame_list = [slow_pathway.unsqueeze(0), fast_pathway.unsqueeze(0)]
    print('Slow pathway input size:', frame_list[0].size())
    print('Fast pathway input size:', frame_list[1].size())
    for i in range(len(frame_list)):
        frame_list[i] = frame_list[i].cuda(non_blocking=True)

    y = model(frame_list)
    print('Network output size:    ', y.size())


if __name__ == "__main__":
    test()
