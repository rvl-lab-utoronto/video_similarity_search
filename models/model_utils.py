import torch

from models.resnet import generate_model
from slowfast.utils.parser import load_config
from slowfast.models.build import build_model


# Resnet params
model_depth=18
n_classes=1039
n_input_channels=3
resnet_shortcut = 'B'
conv1_t_size = 7 #kernel size in t dim of conv1
conv1_t_stride = 1 #stride in t dim of conv1
no_max_pool = True #max pooling after conv1 is removed
resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value


def model_selector(arch_name, args=None):
    assert arch_name in ['3dresnet', 'slowfast']

    if arch_name == '3dresnet':
        model=generate_model(model_depth=model_depth, n_classes=n_classes,
                        n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)
    elif arch_name == 'slowfast':
        cfg = load_config(args)
        model = build_model(cfg)
    
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

