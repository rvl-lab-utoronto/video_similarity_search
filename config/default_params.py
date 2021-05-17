"""Configs."""
from fvcore.common.config import CfgNode

# Config definition
_C = CfgNode()


# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 300
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.DATASET = "ucf101"
#_C.TRAIN.DATASET = "kinetics"
_C.TRAIN.NUM_DATA_WORKERS = 4
_C.TRAIN.LOG_INTERVAL = 5 #for print statements

_C.TRAIN.EVAL_BATCH_SIZE = False

# -----------------------------------------------------------------------------
# Testing options
# -----------------------------------------------------------------------------
_C.VAL = CfgNode()
_C.VAL.METRIC = 'global' #local_batch
_C.VAL.BATCH_SIZE = 80 #note that local_batch metric is sensitive to the batch_size
_C.VAL.LOG_INTERVAL = 5

# -----------------------------------------------------------------------------
# Testing options
# -----------------------------------------------------------------------------
_C.TEST = CfgNode()
#_C.TEST.DATASET = "ucf101"
#_C.TEST.BATCH_SIZE = 8


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.ARCH = "slowfast"
#_C.MODEL.ARCH = "3dresnet"
_C.MODEL.PREDICT_TEMPORAL_DS = False

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
#_C.DATASET.VID_PATH = '/media/diskstation/datasets/UCF101/jpg'
#_C.DATASET.ANNOTATION_PATH = '/media/diskstation/datasets/UCF101/json/ucf101_01.json'
_C.DATASET.VID_PATH = '/media/diskstation/datasets/kinetics400/frames_shortedge320px_25fps'
_C.DATASET.ANNOTATION_PATH = '/media/diskstation/datasets/kinetics400/vid_paths_and_labels/frame_paths'

_C.DATASET.CLUSTER_PATH = ''
_C.DATASET.TARGET_TYPE_T = 'label' #[label, cluster_label], where label refer to the true label
_C.DATASET.TARGET_TYPE_V = 'label'

_C.DATASET.SAMPLING_STRATEGY = 'random_semi_hard' #random_negative
_C.DATASET.POSITIVE_SAMPLING_P = 0.8

_C.DATASET.CHANNEL_EXTENSIONS = ''
_C.DATASET.KEYPOINT_PATH = ''
_C.DATASET.SALIENT_PATH = ''
_C.DATASET.OPTICAL_U_PATH = ''
_C.DATASET.OPTICAL_V_PATH = ''

_C.DATASET.MODALITY=False
_C.DATASET.POS_CHANNEL_REPLACE = False
_C.DATASET.PROB_POS_CHANNEL_REPLACE=0.25
_C.DATASET.RECONSTRUCTION = False


# -----------------------------------------------------------------------------
# Slowfast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()
_C.SLOWFAST.CFG_PATH = 'models/slowfast/configs/Kinetics/SLOWFAST_8x8_R50.yaml'
_C.SLOWFAST.ALPHA = 4
_C.SLOWFAST.FAST_MASK = False


# -----------------------------------------------------------------------------
# 3dResNet options
# -----------------------------------------------------------------------------
_C.RESNET=CfgNode()
_C.RESNET.MODEL_DEPTH = 18
_C.RESNET.N_CLASSES=101
_C.RESNET.PROJECTION_HEAD = True
_C.RESNET.HIDDEN_LAYER = 2048
_C.RESNET.OUT_DIM = 128
_C.RESNET.SHORTCUT = 'B'
_C.RESNET.CONV1_T_SIZE = 7
_C.RESNET.CONV1_T_STRIDE = 1
_C.RESNET.NO_MAX_POOl = True
_C.RESNET.WIDEN_FACTOR = 1

_C.RESNET.ATTENTION = False

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The spatial crop size of the input clip.
_C.DATA.SAMPLE_SIZE = 224

# The number of frames of the input clip.
_C.DATA.SAMPLE_DURATION = 8

# Input frame channel dimension.
_C.DATA.INPUT_CHANNEL_NUM = 3

#select ['center', 'random', 'avg'] for temporal cropping in data preprocessing
_C.DATA.TEMPORAL_CROP='center'

# -----------------------------------------------------------------------------
# Loss Options
# -----------------------------------------------------------------------------
_C.LOSS = CfgNode()
_C.LOSS.TYPE = 'triplet'
_C.LOSS.MARGIN = 0.2
_C.LOSS.DIST_METRIC = 'cosine'
#_C.LOSS.DIST_METRIC = 'euclidean'

# NCE loss params
_C.LOSS.K = 1024 #num of negatives
_C.LOSS.T = 0.07 #temperature
_C.LOSS.M = 0.5 #momentum
_C.LOSS.FEAT_DIM = 128

#Relative speed perception
_C.LOSS.RELATIVE_SPEED_PERCEPTION = False

#Local local contrast
_C.LOSS.LOCAL_LOCAL_CONTRAST = False
_C.LOSS.LOCAL_LOCAL_WEIGHT = 1.0
_C.LOSS.LOCAL_LOCAL_MARGIN = 0.04

#intra negative
_C.LOSS.INTRA_NEGATIVE = False

# -----------------------------------------------------------------------------
# Optimizer options
# -----------------------------------------------------------------------------
_C.OPTIM = CfgNode()
_C.OPTIM.OPTIMIZER = 'sgd'
_C.OPTIM.WD = 0.00001
_C.OPTIM.LR = 0.01
_C.OPTIM.MOMENTUM = 0.5
_C.OPTIM.SCHEDULE = []

# -----------------------------------------------------------------------------
# Iterative clustering options
# -----------------------------------------------------------------------------
_C.ITERCLUSTER = CfgNode()
#_C.ITERCLUSTER.METHOD = 'spherical_kmeans'
_C.ITERCLUSTER.METHOD = 'kmeans'
_C.ITERCLUSTER.INTERVAL = 5
_C.ITERCLUSTER.K = 1000
_C.ITERCLUSTER.ADAPTIVEP = False
_C.ITERCLUSTER.WARMUP_EPOCHS = 0
_C.ITERCLUSTER.L2_NORMALIZE = True
_C.ITERCLUSTER.FINCH_PARTITION = 0

# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
_C.NUM_GPUS = 1
_C.OUTPUT_PATH = "."
_C.SYNC_BATCH_NORM = False


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


# Check assertions for the cfg parameters and return the cfg
def _assert_and_infer_cfg(cfg):
    #assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    #assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
