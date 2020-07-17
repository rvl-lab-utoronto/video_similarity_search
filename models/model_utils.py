import torch

from models.resnet import generate_model
#from models.slowfast.slowfast.models.build import build_model
from models.slowfast.slowfast.models.video_model_builder import SlowFast
from models.slowfast.slowfast.config.defaults import get_cfg

def model_selector(cfg):
    assert cfg.MODEL.ARCH in ['3dresnet', 'slowfast']

    if cfg.MODEL.ARCH == '3dresnet':
        model=generate_model(model_depth=cfg.RESNET.MODEL_DEPTH,
                        n_classes=cfg.RESNET.N_CLASSES,
                        n_input_channels=cfg.RESNET.N_INPUT_CHANNELS,
                        shortcut_type=cfg.RESNET.SHORTCUT,
                        conv1_t_size=cfg.RESNET.CONV1_T_SIZE,
                        conv1_t_stride=cfg.RESNET.CONV1_T_STRIDE,
                        no_max_pool=cfg.RESNET.NO_MAX_POOl,
                        widen_factor=cfg.RESNET.WIDEN_FACTOR)

    elif cfg.MODEL.ARCH == 'slowfast':
        slowfast_cfg = get_cfg()
        slowfast_cfg.merge_from_file(cfg.SLOWFAST.CFG_PATH)

        slowfast_cfg.NUM_GPUS = cfg.NUM_GPUS
        slowfast_cfg.DATA.NUM_FRAMES = cfg.DATA.SAMPLE_DURATION
        slowfast_cfg.DATA.CROP_SIZE = cfg.DATA.SAMPLE_SIZE
        slowfast_cfg.DATA.INPUT_CHANNEL_NUM = [cfg.DATA.INPUT_CHANNEL_NUM, cfg.DATA.INPUT_CHANNEL_NUM]

        #model = build_model(slowfast_cfg)
        model = SlowFast(slowfast_cfg)

    return model


def multipathway_input(frames, cfg):
    # assume batchsize already in tensor dimension
    frame_idx = 2

    fast_pathway = frames
    slow_pathway = torch.index_select(frames, frame_idx, torch.linspace(0,
        frames.shape[frame_idx] - 1, frames.shape[frame_idx] // cfg.SLOWFAST.ALPHA).long(),)
    frame_list = [slow_pathway, fast_pathway]

    return frame_list
