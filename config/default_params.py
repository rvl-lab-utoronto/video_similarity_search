"""Configs."""
from fvcore.common.config import CfgNode

# Config definition
_C = CfgNode()


# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.DATASET = "ucf101"


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
#_C.MODEL.ARCH = "slowfast"
_C.MODEL.ARCH = "3dresnet"


# -----------------------------------------------------------------------------
# Slowfast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()
_C.SLOWFAST.CFG_PATH = 'models/slowfast/configs/Kinetics/SLOWFAST_8x8_R50.yaml'


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


# -----------------------------------------------------------------------------
# Loss Options
# -----------------------------------------------------------------------------
_C.LOSS = CfgNode()
_C.LOSS.MARGIN = 0.2


# -----------------------------------------------------------------------------
# Optimizer options
# -----------------------------------------------------------------------------
_C.OPTIM = CfgNode()
_C.OPTIM.LR = 0.05
_C.OPTIM.MOMENTUM = 0.5


# ----------------------------------------------------------------------------- 
# Misc options
# -----------------------------------------------------------------------------
_C.NUM_GPUS = 1
_C.OUTPUT_PATH = "."


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
