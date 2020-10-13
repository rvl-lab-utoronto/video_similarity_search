import torch
import shutil
import os

from models.resnet import generate_model
#from models.slowfast.slowfast.models.build import build_model
#from models.slowfast.slowfast.models.video_model_builder import SlowFast
from models.slowfast.slowfast.models.video_model_builder import SlowFastRepresentation
from models.slowfast.slowfast.config.defaults import get_cfg

import copy
import torchvision
from models.baselines.inflate_src.i3res import I3ResNet
from models.baselines.simclr_pytorch.resnet_wider import resnet50x1


def create_output_dirs(cfg):
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints')):
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints'))


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def imagenet_inflated(num_frames, center_init=False):
    resnet = torchvision.models.resnet50(pretrained=True)

    # Inflate with time dimensions of kernels initialized with copied weights /
    # size_time_dim
    i3resnet = I3ResNet(copy.deepcopy(resnet), num_frames, center=center_init)

    # Discard FC (last kept is avg pool)
    i3resnet = torch.nn.Sequential(*(list(i3resnet.children())[:-1]), Flatten())
    return i3resnet


def simclr_inflated(num_frames, center_init=False):
    sd = 'models/baselines/simclr_pytorch/resnet50-1x.pth'
    resnet = resnet50x1()
    sd = torch.load(sd, map_location='cpu')
    resnet.load_state_dict(sd['state_dict'])

    # Inflate with time dimensions of kernels initialized with copied weights /
    # size_time_dim
    i3resnet = I3ResNet(copy.deepcopy(resnet), num_frames, center=center_init)

    # Discard FC (last kept is avg pool)
    i3resnet = torch.nn.Sequential(*(list(i3resnet.children())[:-1]), Flatten())
    return i3resnet


def mocov2_inflated(num_frames, center_init=True):
    model = torchvision.models.__dict__['resnet50']()

    sd = 'models/baselines/mocov2_pytorch/moco_v2_200ep_pretrain.pth.tar'
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

    # Inflate with time dimensions of kernels initialized with copied weights /
    # size_time_dim
    i3resnet = I3ResNet(copy.deepcopy(model), num_frames, center=center_init)

    # Discard FC (last kept is avg pool)
    i3resnet = torch.nn.Sequential(*(list(i3resnet.children())[:-1]), Flatten())
    return i3resnet


# Select the appropriate model with the specified cfg parameters
def model_selector(cfg):
    assert cfg.MODEL.ARCH in ['3dresnet', 'slowfast',
            'simclr_pretrained_inflated_res50',
            'imagenet_pretrained_inflated_res50',
            'mocov2_pretrained_inflated_res50']

    if cfg.MODEL.ARCH == '3dresnet':
        model=generate_model(model_depth=cfg.RESNET.MODEL_DEPTH,
                        n_classes=cfg.RESNET.N_CLASSES,
                        n_input_channels=cfg.DATA.INPUT_CHANNEL_NUM,
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

        # Use a custom SlowFast with a head that doesn't include the FC
        # layers after the global avg pooling and dropout
        model = SlowFastRepresentation(slowfast_cfg)

        # Unused Model with FC:
        #model = build_model(slowfast_cfg)
        #model = SlowFast(slowfast_cfg)

    elif cfg.MODEL.ARCH == 'simclr_pretrained_inflated_res50':
        model = simclr_inflated(cfg.DATA.SAMPLE_DURATION)

    elif cfg.MODEL.ARCH == 'imagenet_pretrained_inflated_res50':
        model = imagenet_inflated(cfg.DATA.SAMPLE_DURATION)

    elif cfg.MODEL.ARCH == 'mocov2_pretrained_inflated_res50':
        model = mocov2_inflated(cfg.DATA.SAMPLE_DURATION)

    return model


def multipathway_input(frames, cfg):
    # assume batchsize already in tensor dimension
    frame_idx = 2

    fast_pathway = frames
    slow_pathway = torch.index_select(frames, frame_idx, torch.linspace(0,
        frames.shape[frame_idx] - 1, frames.shape[frame_idx] // cfg.SLOWFAST.ALPHA).long(),)
    frame_list = [slow_pathway, fast_pathway]

    return frame_list


# Load pretrained model from the specified checkpoint path
def load_pretrained_model(model, pretrain_path, is_master_proc=True):
    if pretrain_path:
        if (is_master_proc):
            print('Loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])
    return model


# Saved model checkpoint to the specified path (only do this for the master
# process if in distributed training)
def save_checkpoint(state, is_best, model_name, output_path, is_master_proc=True, filename='checkpoint.pth.tar'):
    if not is_master_proc:
        return
    """Saves checkpoint to disk"""
    directory = "tnet_checkpoints/%s/"%(model_name)
    directory = os.path.join(output_path, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if (is_master_proc):
        print('\n=> checkpoint:{} saved...'.format(filename))
    if is_best:
        shutil.copyfile(filename,  os.path.join(directory, 'model_best.pth.tar'))
        if (is_master_proc):
            print('=> best_model saved as:{}'.format(os.path.join(directory, 'model_best.pth.tar')))


# Load model checkpoint from the specified path
def load_checkpoint(model, checkpoint_path, is_master_proc=True):
    if os.path.isfile(checkpoint_path):
        if (is_master_proc):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if (is_master_proc):
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        if (is_master_proc):
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return start_epoch, best_prec1
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(dista, distb):
    margin = 0
    pred = (distb - dista - margin)
    return (pred > 0).sum() * 1.0 / (dista.size()[0])
