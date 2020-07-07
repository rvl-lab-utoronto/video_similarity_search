import torch

from models.resnet import generate_model
from slowfast.models.build import build_model
from slowfast.config.defaults import get_cfg


# Resnet params
model_depth=18
n_classes=1039
n_input_channels=3
resnet_shortcut = 'B'
conv1_t_size = 7 #kernel size in t dim of conv1
conv1_t_stride = 1 #stride in t dim of conv1
no_max_pool = True #max pooling after conv1 is removed
resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value


def model_selector(cfg):
    assert cfg.MODEL.ARCH in ['3dresnet', 'slowfast']

    if cfg.MODEL.ARCH == '3dresnet':
        model=generate_model(model_depth=model_depth, n_classes=n_classes,
                        n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)

    elif cfg.MODEL.ARCH == 'slowfast':
        slowfast_cfg = get_cfg()
        slowfast_cfg.merge_from_file(cfg.SLOWFAST.CFG_PATH)

        slowfast_cfg.NUM_GPUS = cfg.NUM_GPUS
        slowfast_cfg.DATA.NUM_FRAMES = cfg.DATA.SAMPLE_DURATION
        slowfast_cfg.DATA.CROP_SIZE = cfg.DATA.SAMPLE_SIZE
        slowfast_cfg.DATA.INPUT_CHANNEL_NUM = [cfg.DATA.INPUT_CHANNEL_NUM, cfg.DATA.INPUT_CHANNEL_NUM]

        model = build_model(slowfast_cfg)
    
    return model


def multipathway_input(frames):
    # assume batchsize already in tensor dimension
    frame_idx = 2
    SLOWFAST_ALPHA = 4

    fast_pathway = frames
    slow_pathway = torch.index_select(frames, frame_idx, torch.linspace(0,
        frames.shape[frame_idx] - 1, frames.shape[frame_idx] // SLOWFAST_ALPHA).long(),)
    frame_list = [slow_pathway, fast_pathway]

    return frame_list

