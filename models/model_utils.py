import torch
import shutil
import os

from models.resnet import generate_model
#from models.slowfast.slowfast.models.build import build_model
#from models.slowfast.slowfast.models.video_model_builder import SlowFast
from models.slowfast.slowfast.models.video_model_builder import SlowFastRepresentation
from models.slowfast.slowfast.config.defaults import get_cfg

def create_output_dirs(cfg):
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

    if not os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints')):
        os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'tnet_checkpoints'))

# Select the appropriate model with the specified cfg parameters
def model_selector(cfg, projection_head=True):
    assert cfg.MODEL.ARCH in ['3dresnet', 'slowfast']

    if cfg.MODEL.ARCH == '3dresnet':
        model=generate_model(model_depth=cfg.RESNET.MODEL_DEPTH,
                        n_classes=cfg.RESNET.N_CLASSES,
                        n_input_channels=cfg.DATA.INPUT_CHANNEL_NUM,
                        shortcut_type=cfg.RESNET.SHORTCUT,
                        conv1_t_size=cfg.RESNET.CONV1_T_SIZE,
                        conv1_t_stride=cfg.RESNET.CONV1_T_STRIDE,
                        no_max_pool=cfg.RESNET.NO_MAX_POOl,
                        widen_factor=cfg.RESNET.WIDEN_FACTOR,
                        projection_head=projection_head)

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
