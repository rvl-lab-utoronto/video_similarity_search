from slowfast.utils.parser import load_config, parse_args
#from slowfast.models.build import build_model
from slowfast.models.video_model_builder import SlowFastRepresentation
from slowfast.models.video_model_builder import SlowFast

from slowfast.config.defaults import get_cfg

import torch


def rand_input(cfg, batch_size=1, channels=3, num_frames=32, height=224, width=224):
    frames = torch.randn(batch_size, channels, num_frames, height, width)
    frame_idx = 2
    fast_pathway = frames
    slow_pathway = torch.index_select(frames, frame_idx, torch.linspace(0,
        frames.shape[frame_idx] - 1, frames.shape[frame_idx] // cfg.SLOWFAST.ALPHA).long(),)
    # unsqueeze for batch size of 1 in this test
    frame_list = [slow_pathway, fast_pathway]
    print('Slow pathway input size:', frame_list[0].size())
    print('Fast pathway input size:', frame_list[1].size())
    #for i in range(len(frame_list)):
    #    frame_list[i] = frame_list[i].cuda(non_blocking=True)
    return frame_list


def test():
    #args = parse_args()
    #cfg = load_config(args)
    cfg = get_cfg()
    cfg.merge_from_file('configs/Kinetics/SLOWFAST_8x8_R50.yaml')
    cfg.NUM_GPUS = 1

    #model = SlowFast(cfg)
    model = SlowFastRepresentation(cfg)
    print(model)
    #model = build_model(cfg)
    
    #cur_device = torch.cuda.current_device()
    #model = model.cuda(device=cur_device)

    frame_list = rand_input(cfg, 2,3,32,224,224)
    y1 = model(frame_list)
    print('Network output size:    ', y1.size(), '\n')

    #frame_list = rand_input(cfg, 2,3,32,224,224)
    #y = model.forward_no_head(frame_list)
    #print('Network output size without pooling:', y.size(), '\n')

    #frame_list = rand_input(cfg, 3,128,224,224)
    #y_long = model.forward_no_head(frame_list)
    #print('Network output size without pooling:', y_long.size(), '\n')

    frame_list = rand_input(cfg, 2,3,32,224,224)
    y2 = model(frame_list)
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    sim_y1_y2 = cos_sim(y1, y2)
    print('Similarity output size:', sim_y1_y2.size())


if __name__ == "__main__":
    test()
