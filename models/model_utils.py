import torch
import torch.nn as nn
import shutil
import os

from models.resnet import generate_model
from models.infoNCE import InfoNCE, UberNCE
#from models.slowfast.slowfast.models.build import build_model
#from models.slowfast.slowfast.models.video_model_builder import SlowFast
from models.slowfast.slowfast.models.video_model_builder import SlowFastRepresentation
from models.slowfast.slowfast.config.defaults import get_cfg
from models.s3d.select_backbone import select_backbone
from models.r3d.r3d import R3DNet

import copy
import torchvision
from models.baselines.inflate_src.i3res import I3ResNet
from models.baselines.simclr_pytorch.resnet_wider import resnet50x1
from models.multiview import Multiview

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
def model_selector(cfg, projection_head=True, hyperbolic=False, classifier=False, num_classes=101, is_master_proc=True):
    assert cfg.MODEL.ARCH in ['3dresnet', 'slowfast', 'info_nce', "uber_nce", 's3d', 'r3d',
            'simclr_pretrained_inflated_res50',
            'imagenet_pretrained_inflated_res50',
            'mocov2_pretrained_inflated_res50']
    print('n input channel: ', cfg.DATA.INPUT_CHANNEL_NUM)
    if cfg.MODEL.ARCH == '3dresnet':
        model=generate_model(model_depth=cfg.RESNET.MODEL_DEPTH,
                        hidden_layer=cfg.RESNET.HIDDEN_LAYER,
                        out_dim=cfg.RESNET.OUT_DIM,
                        num_classes=num_classes,
                        n_input_channels=cfg.DATA.INPUT_CHANNEL_NUM,
                        shortcut_type=cfg.RESNET.SHORTCUT,
                        conv1_t_size=cfg.RESNET.CONV1_T_SIZE,
                        conv1_t_stride=cfg.RESNET.CONV1_T_STRIDE,
                        no_max_pool=cfg.RESNET.NO_MAX_POOl,
                        widen_factor=cfg.RESNET.WIDEN_FACTOR,
                        projection_head=projection_head,
                        predict_temporal_ds=cfg.MODEL.PREDICT_TEMPORAL_DS,
                        spatio_temporal_attention=cfg.RESNET.ATTENTION,
                        hyperbolic=hyperbolic,
                        classifier=classifier)

        #only resnet supports multiview for now
        if cfg.DATASET.MODALITY == True:
            encoder1 = model
            encoder2 = generate_model(model_depth=cfg.RESNET.MODEL_DEPTH,
                        hidden_layer=cfg.RESNET.HIDDEN_LAYER,
                        out_dim=cfg.RESNET.OUT_DIM,
                        n_input_channels=cfg.DATA.INPUT_CHANNEL_NUM,
                        shortcut_type=cfg.RESNET.SHORTCUT,
                        conv1_t_size=cfg.RESNET.CONV1_T_SIZE,
                        conv1_t_stride=cfg.RESNET.CONV1_T_STRIDE,
                        no_max_pool=cfg.RESNET.NO_MAX_POOl,
                        widen_factor=cfg.RESNET.WIDEN_FACTOR,
                        projection_head=projection_head,
                        predict_temporal_ds=cfg.MODEL.PREDICT_TEMPORAL_DS,
                        spatio_temporal_attention=cfg.RESNET.ATTENTION)

            model = Multiview(encoder1, encoder2, cfg.RESNET.OUT_DIM)


    elif cfg.MODEL.ARCH == 's3d':
        dim = 128
        backbone, param = select_backbone('s3d', first_channel=cfg.DATA.INPUT_CHANNEL_NUM)
        feature_size = param['feature_size']
        model = nn.Sequential(backbone,
                              nn.AdaptiveAvgPool3d((1,1,1)),
                              nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                              nn.ReLU(),
                              nn.Conv3d(feature_size, dim, kernel_size=1, bias=True),
                              Flatten())

    elif cfg.MODEL.ARCH == 'r3d':
        dim=128
        feature_size = 512
        backbone = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
        model = nn.Sequential(backbone,
                              nn.Linear(feature_size, feature_size),
                              nn.ReLU(),
                              nn.Linear(feature_size, dim))

    elif cfg.MODEL.ARCH == 'slowfast':
        slowfast_cfg = get_cfg()
        slowfast_cfg.merge_from_file(cfg.SLOWFAST.CFG_PATH)

        slowfast_cfg.NUM_GPUS = cfg.NUM_GPUS
        slowfast_cfg.DATA.NUM_FRAMES = cfg.DATA.SAMPLE_DURATION
        slowfast_cfg.DATA.CROP_SIZE = cfg.DATA.SAMPLE_SIZE

        if cfg.SLOWFAST.FAST_MASK:
            if is_master_proc:
                print ("Fast pathway of SlowFast will use mask input")
            # 4th input channel will be sent to the fast pathway while RGB is
            # sent to slow pathway
            assert (cfg.DATA.INPUT_CHANNEL_NUM == 4)
            slowfast_cfg.DATA.INPUT_CHANNEL_NUM = [3, 3]
        else:
            slowfast_cfg.DATA.INPUT_CHANNEL_NUM = [cfg.DATA.INPUT_CHANNEL_NUM, cfg.DATA.INPUT_CHANNEL_NUM]

        # Use a custom SlowFast with a head that doesn't include the FC
        # layers after the global avg pooling and dropout
        model = SlowFastRepresentation(slowfast_cfg, projection_head=True)

        # Unused Model with FC:
        #model = build_model(slowfast_cfg)
        #model = SlowFast(slowfast_cfg)
    elif cfg.MODEL.ARCH == 'info_nce':
        model = InfoNCE('s3d', dim=128, K=2048, m=0.999, T=0.07) #TODO: use config parameters

    elif cfg.MODEL.ARCH == 'uber_nce':
        model = UberNCE('s3d', dim=128, K=2048, m=0.999, T=0.07)

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

    if cfg.SLOWFAST.FAST_MASK:
        # Use salient obj channel only for fast path
        slow_pathway = slow_pathway[:,:3,:,:,:]
        fast_pathway = fast_pathway[:,3,:,:,:]
        fast_pathway = torch.stack((fast_pathway, fast_pathway, fast_pathway), dim=1)

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
def load_checkpoint(model, checkpoint_path, classifier=False, is_master_proc=True):
    if os.path.isfile(checkpoint_path):
        if (is_master_proc):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        state_dict = checkpoint['state_dict']

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items(): #edit
            if 'module.' in k:
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            elif classifier and ('fc' in k or 'bn_proj' in k):
                continue
            else:
                new_state_dict[k] = v
        # load params
        if classifier:
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict)

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
